from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.generator import generate_answer
from app.parser import chunk_documents, load_pdf
from app.retriever import HybridRetriever, build_or_load_vector_store

# Global pipeline state (Server startup par ek dafa memory mein load ho jayega)
pipeline = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup par Vector Store aur BM25 Retriever load karta hai."""
    print("\n[Startup] Initializing Apple 10-K RAG Engine...")
    raw_docs = load_pdf()
    chunks = chunk_documents(raw_docs)
    vector_db = build_or_load_vector_store()

    pipeline["retriever"] = HybridRetriever(vector_db, chunks)
    print("[Startup] RAG Engine ready to accept API queries!\n")
    yield
    pipeline.clear()


app = FastAPI(
    title="Apple 10-K RAG API",
    description="Production-grade Financial RAG Pipeline using FastAPI, ChromaDB, and Google Gemini",
    version="1.0.0",
    lifespan=lifespan,
)


# Pydantic Data Models
class QueryRequest(BaseModel):
    query: str = Field(
        ..., example="What was Apple's total net sales for iPhone in 2025?"
    )
    top_k: int = Field(default=3, ge=1, le=10, example=3)


class SourceCitation(BaseModel):
    page: int | None
    file_name: str
    snippet: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: list[SourceCitation]


# Endpoints
@app.get("/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    return {"status": "ok", "retriever_loaded": "retriever" in pipeline}


@app.post("/query", response_model=QueryResponse, tags=["RAG Query"])
def query_rag(request: QueryRequest):
    """Query receive karta hai, hybrid retrieval + reranking run karta hai, aur Gemini se answer generate karwata hai."""
    if "retriever" not in pipeline:
        raise HTTPException(
            status_code=500, detail="RAG Engine not initialized."
        )

    try:
        retriever = pipeline["retriever"]
        retrieved_docs = retriever.retrieve_and_rerank(
            query=request.query, top_k=request.top_k
        )
        result = generate_answer(request.query, retrieved_docs)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error processing query: {str(e)}"
        )