from __future__ import annotations

import re
import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.models import (
    ChatRequest,
    ChatResponse,
    ClearCacheResponse,
    ConversationsResponse,
    DocumentsResponse,
    UploadResponse,
)
from app.services.document_parser import DocumentParsingError, validate_extension
from app.services.rag_service import RAGService
from app.services.state_store import StateStore, utc_now_iso
from app.services.tracing import log_dict, log_metric, log_params, setup_mlflow, traced_run

BASE_DIR = Path(__file__).resolve().parent.parent


def _resolve_path(path_value: str) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return BASE_DIR / path


settings = get_settings()
upload_dir = _resolve_path(settings.upload_dir)
upload_dir.mkdir(parents=True, exist_ok=True)

state_file = _resolve_path(settings.state_file)
state_store = StateStore(
    file_path=str(state_file),
    cache_size=settings.cache_size,
    conversation_history_size=settings.conversation_history_size,
)

rag_service = RAGService(settings)
setup_mlflow(settings.mlflow_tracking_uri, settings.mlflow_experiment)

app = FastAPI(title="BasicRAG", version="0.1.0")

allowed_origins = [item.strip() for item in settings.allow_origins.split(",") if item.strip()]
if not allowed_origins:
    allowed_origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def build_scope(document_ids: list[str]) -> str:
    if not document_ids:
        return "all"
    return ",".join(sorted(document_ids))


def build_cache_key(question: str, scope: str) -> str:
    return f"{normalize_text(question)}||{scope}"


def extract_topic(question: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", question.lower())
    keywords = []
    for word in words:
        if len(word) < 3 or word in STOP_WORDS:
            continue
        if word not in keywords:
            keywords.append(word)

        if len(keywords) == 4:
            break

    if not keywords:
        return "general"
    return " ".join(keywords)


def record_conversation(
    question: str,
    answer: str,
    topic: str,
    scope: str,
    cache_hit: bool,
) -> None:
    state_store.add_conversation(
        {
            "question": question,
            "answer": answer,
            "topic": topic,
            "scope": scope,
            "cache_hit": cache_hit,
            "asked_at": utc_now_iso(),
        }
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(static_dir / "index.html")


@app.get("/api/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "pinecone_ready": rag_service.pinecone_ready,
        "pinecone_status": rag_service.pinecone_status,
        "cache_entries": state_store.cache_count(),
    }


@app.get("/api/documents", response_model=DocumentsResponse)
def documents() -> DocumentsResponse:
    return DocumentsResponse(documents=state_store.list_documents())


@app.get("/api/conversations", response_model=ConversationsResponse)
def conversations() -> ConversationsResponse:
    return ConversationsResponse(conversations=state_store.list_conversations(limit=40))


@app.post("/api/upload", response_model=UploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    user_api_key: str = Form(""),
) -> UploadResponse:
    if not user_api_key.strip():
        raise HTTPException(
            status_code=400,
            detail="Provide your OpenAI API key to upload and index a document.",
        )

    original_name = file.filename or "uploaded_file"

    try:
        validate_extension(original_name)
    except DocumentParsingError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    safe_name = re.sub(r"[^A-Za-z0-9_.-]", "_", original_name)
    stored_name = f"{uuid4().hex}_{safe_name}"
    stored_path = upload_dir / stored_name

    try:
        with stored_path.open("wb") as target:
            shutil.copyfileobj(file.file, target)
    finally:
        await file.close()

    with traced_run("upload_document", tags={"endpoint": "/api/upload"}):
        log_params(
            {
                "file_name": original_name,
                "file_type": Path(original_name).suffix.lower().lstrip("."),
            }
        )

        try:
            metadata = rag_service.ingest_document(stored_path, original_name, user_api_key)
        except DocumentParsingError as exc:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Upload failed: {exc}") from exc

        state_store.add_document(metadata)
        log_metric("chunk_count", float(metadata["chunk_count"]))
        log_dict(metadata, "upload_metadata.json")

    return UploadResponse(**metadata)


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest) -> ChatResponse:
    question = payload.question.strip()
    user_api_key = payload.user_api_key.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if not user_api_key:
        raise HTTPException(
            status_code=400,
            detail="Provide your OpenAI API key before asking a question.",
        )

    known_document_ids = state_store.get_document_ids()
    if not known_document_ids:
        raise HTTPException(
            status_code=400,
            detail="No documents available yet. Upload PDF, CSV, or TXT first.",
        )

    selected_ids = [doc_id for doc_id in payload.document_ids if doc_id]
    unknown_ids = sorted(set(selected_ids) - known_document_ids)
    if unknown_ids:
        raise HTTPException(
            status_code=400,
            detail="Some selected documents are no longer available. Refresh and try again.",
        )

    scope = build_scope(selected_ids)
    topic = extract_topic(question)
    cache_key = build_cache_key(question, scope)

    cached = state_store.get_cached_answer(cache_key)
    if cached:
        response = ChatResponse(
            answer=str(cached.get("answer", "")),
            references=cached.get("references", []),
            cache_hit=True,
            topic=str(cached.get("topic", topic)),
            scope=scope,
        )
        record_conversation(question, response.answer, response.topic, scope, cache_hit=True)
        return response

    with traced_run("chat_question", tags={"endpoint": "/api/chat"}):
        log_params(
            {
                "scope": scope,
                "selected_doc_count": len(selected_ids),
                "cache_hit": "false",
            }
        )

        try:
            rag_output = rag_service.answer_question(
                question=question,
                user_api_key=user_api_key,
                document_ids=selected_ids,
            )
        except RuntimeError as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Chat failed: {exc}") from exc

        log_metric("retrieved_count", float(rag_output["retrieved_count"]))
        log_dict(
            {
                "question": question,
                "topic": topic,
                "scope": scope,
                "retrieved_count": rag_output["retrieved_count"],
            },
            "chat_trace.json",
        )

    response = ChatResponse(
        answer=rag_output["answer"],
        references=rag_output["references"],
        cache_hit=False,
        topic=topic,
        scope=scope,
    )

    state_store.set_cached_answer(
        cache_key,
        {
            "answer": response.answer,
            "references": [item.model_dump() for item in response.references],
            "topic": topic,
            "scope": scope,
        },
    )

    record_conversation(question, response.answer, topic, scope, cache_hit=False)
    return response


@app.post("/api/cache/clear", response_model=ClearCacheResponse)
def clear_cache() -> ClearCacheResponse:
    cleared_count = state_store.clear_cache()
    return ClearCacheResponse(
        message="Cache cleared successfully.",
        cleared_count=cleared_count,
    )
