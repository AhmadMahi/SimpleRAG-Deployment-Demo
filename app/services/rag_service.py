from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

from app.config import Settings
from app.services.document_parser import DocumentParsingError, parse_file


class RAGService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=900,
            chunk_overlap=140,
            separators=["\n\n", "\n", " ", ""],
        )

        self._pinecone_error = ""
        self._index = None

        self._init_pinecone()

    @property
    def pinecone_ready(self) -> bool:
        return self._index is not None

    @property
    def pinecone_status(self) -> str:
        if self._index is not None:
            return "ready"
        return self._pinecone_error or "not_configured"

    def _init_pinecone(self) -> None:
        if not self.settings.pinecone_api_key:
            self._pinecone_error = "Set PINECONE_API_KEY to enable retrieval."
            return

        try:
            pc = Pinecone(api_key=self.settings.pinecone_api_key)
            existing_names = self._extract_index_names(pc.list_indexes())

            if self.settings.pinecone_index not in existing_names:
                pc.create_index(
                    name=self.settings.pinecone_index,
                    dimension=self.settings.embedding_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=self.settings.pinecone_cloud,
                        region=self.settings.pinecone_region,
                    ),
                )

            self._index = pc.Index(self.settings.pinecone_index)
        except Exception as exc:  # pragma: no cover - external dependency behavior
            self._pinecone_error = f"Pinecone setup failed: {exc}"
            self._index = None

    def _extract_index_names(self, list_result: Any) -> set[str]:
        if hasattr(list_result, "names"):
            names = list_result.names()
            return set(str(name) for name in names)

        index_items = list_result
        if isinstance(list_result, dict):
            index_items = list_result.get("indexes", [])

        names: set[str] = set()
        for item in index_items:
            if isinstance(item, dict):
                name = item.get("name")
            else:
                name = getattr(item, "name", None)

            if name:
                names.add(str(name))

        return names

    def _require_index(self) -> None:
        if self._index is None:
            raise RuntimeError(self.pinecone_status)

    def _openai_client(self, api_key: str) -> OpenAI:
        key = api_key.strip()
        if not key:
            raise ValueError("OpenAI API key is required.")
        return OpenAI(api_key=key)

    def _embed_texts(self, client: OpenAI, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        batch_size = 64

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = client.embeddings.create(
                model=self.settings.openai_embedding_model,
                input=batch,
            )
            vectors.extend(item.embedding for item in response.data)

        return vectors

    def ingest_document(
        self,
        file_path: Path,
        original_filename: str,
        user_api_key: str,
    ) -> dict[str, Any]:
        self._require_index()

        raw_text = parse_file(file_path)
        chunks = [chunk.strip() for chunk in self.splitter.split_text(raw_text) if chunk.strip()]

        if not chunks:
            raise DocumentParsingError("No usable text chunks found after splitting.")

        client = self._openai_client(user_api_key)
        vectors = self._embed_texts(client, chunks)

        doc_id = str(uuid4())
        uploaded_at = datetime.now(timezone.utc).isoformat()

        pinecone_vectors: list[dict[str, Any]] = []
        for index, (chunk_text, embedding) in enumerate(zip(chunks, vectors)):
            pinecone_vectors.append(
                {
                    "id": f"{doc_id}:{index}",
                    "values": embedding,
                    "metadata": {
                        "doc_id": doc_id,
                        "source": original_filename,
                        "chunk_id": index,
                        "uploaded_at": uploaded_at,
                        "text": chunk_text,
                    },
                }
            )

        upsert_batch_size = 100
        for i in range(0, len(pinecone_vectors), upsert_batch_size):
            batch = pinecone_vectors[i : i + upsert_batch_size]
            self._index.upsert(vectors=batch, namespace=self.settings.pinecone_namespace)

        return {
            "id": doc_id,
            "name": original_filename,
            "file_type": file_path.suffix.lower().lstrip("."),
            "uploaded_at": uploaded_at,
            "chunk_count": len(chunks),
        }

    def answer_question(
        self,
        question: str,
        user_api_key: str,
        document_ids: list[str] | None,
        top_k: int = 6,
    ) -> dict[str, Any]:
        self._require_index()

        client = self._openai_client(user_api_key)
        query_vector = self._embed_texts(client, [question])[0]

        doc_filter = None
        if document_ids:
            doc_filter = {"doc_id": {"$in": document_ids}}

        query_response = self._index.query(
            vector=query_vector,
            top_k=top_k,
            include_metadata=True,
            namespace=self.settings.pinecone_namespace,
            filter=doc_filter,
        )

        matches_payload: list[dict[str, Any]] = []
        for match in query_response.matches or []:
            metadata = dict(match.metadata or {})
            chunk_text = str(metadata.get("text", "")).strip()
            if not chunk_text:
                continue

            matches_payload.append(
                {
                    "doc_id": str(metadata.get("doc_id", "")),
                    "source": str(metadata.get("source", "unknown")),
                    "chunk_id": int(metadata.get("chunk_id", 0)),
                    "score": float(match.score) if match.score is not None else None,
                    "text": chunk_text,
                }
            )

        if not matches_payload:
            return {
                "answer": "I could not find enough context in the selected documents. Try selecting more documents or asking a narrower question.",
                "references": [],
                "retrieved_count": 0,
            }

        context_parts = []
        for i, item in enumerate(matches_payload, start=1):
            context_parts.append(
                f"[{i}] Source: {item['source']} (chunk {item['chunk_id']})\n{item['text']}"
            )

        system_prompt = (
            "You are a concise RAG assistant. Use only the provided context. "
            "If context is insufficient, say that clearly. "
            "Cite supporting source numbers in square brackets like [1], [2]."
        )

        user_prompt = (
            f"Question:\n{question}\n\n"
            f"Context:\n{chr(10).join(context_parts)}\n\n"
            "Provide a direct answer first, then a short evidence note."
        )

        response = client.chat.completions.create(
            model=self.settings.openai_chat_model,
            temperature=0.2,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        answer = (response.choices[0].message.content or "").strip()
        if not answer:
            answer = "I could not generate an answer for this question."

        references = []
        for item in matches_payload:
            snippet = item["text"]
            if len(snippet) > 260:
                snippet = f"{snippet[:260].rstrip()}..."

            references.append(
                {
                    "doc_id": item["doc_id"],
                    "source": item["source"],
                    "chunk_id": item["chunk_id"],
                    "score": round(item["score"], 4)
                    if item["score"] is not None
                    else None,
                    "text": snippet,
                }
            )

        return {
            "answer": answer,
            "references": references,
            "retrieved_count": len(references),
        }
