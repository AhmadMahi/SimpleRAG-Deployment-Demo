# BasicRAG (FastAPI + Pinecone + MLflow)

Minimal Retrieval-Augmented Generation app with:

- GPT model: `gpt-4o-mini`
- Embedding model: `text-embedding-3-small`
- Vector DB: Pinecone
- Tracing: MLflow
- Frontend: single-page HTML/CSS/JS UI
- File support: PDF, CSV, TXT only

## Features

- Upload and index PDF/CSV/TXT files.
- Bring your own OpenAI API key from the UI.
- Document history with selectable scope:
  - talk to one or more specific docs
  - or talk to all docs
- RAG answers with source references per response.
- Simple cache for recent Q+A by question + scope.
  - cache-hit responses are explicitly marked in UI
  - clear cache button provided
- Recent conversation/topic history.
- MLflow tracing for upload and chat flows.

## Quick Start

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and fill Pinecone values.

4. Run the app:

```bash
uvicorn app.main:app --reload
```

5. Open:

```text
http://127.0.0.1:8000
```

## Environment Variables

Required for backend retrieval:

- `PINECONE_API_KEY`

Commonly configured:

- `PINECONE_INDEX` (default: `basic-rag-index`)
- `PINECONE_NAMESPACE` (default: `default`)
- `PINECONE_CLOUD` (default: `aws`)
- `PINECONE_REGION` (default: `us-east-1`)
- `MLFLOW_TRACKING_URI` (default: `mlruns`)
- `MLFLOW_EXPERIMENT` (default: `basic-rag`)

OpenAI API key is provided per request in the UI.

## Render Deployment

This repo includes `render.yaml` for blueprint deployment.

1. Push this repo to GitHub.
2. In Render, create a new Blueprint from the repo.
3. Set `PINECONE_API_KEY` in Render secrets.
4. Deploy.

Render runs:

- Build: `pip install -r requirements.txt`
- Start: `uvicorn app.main:app --host 0.0.0.0 --port $PORT`

## Notes

- Uploaded files are stored in `uploads/`.
- App state (documents, cache, conversation history) is stored in `data/state.json`.
- For production, consider persistent storage volume and stricter API key handling.
# SimpleRAG-Deployment-Demo
