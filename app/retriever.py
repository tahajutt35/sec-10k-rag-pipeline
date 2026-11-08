import os
import time
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sentence_transformers import CrossEncoder
from app.config import settings

# Global Lazy Load for CrossEncoder Reranker
_reranker_model = None


def get_reranker():
    global _reranker_model
    if _reranker_model is None:
        print("Loading Cross-Encoder Reranker model...")
        _reranker_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _reranker_model


def get_embedding_function():
    return GoogleGenerativeAIEmbeddings(
        model="models/gemini-embedding-001",
        google_api_key=settings.GOOGLE_API_KEY,
    )


def build_or_load_vector_store(chunks: list[Document] = None) -> Chroma:
    embeddings = get_embedding_function()

    # Agar DB disk par maujood hai, direct load karo
    if os.path.exists(settings.CHROMA_DB_DIR) and len(
        os.listdir(settings.CHROMA_DB_DIR)
    ) > 0:
        print("Loading existing Chroma Vector Store from disk...")
        return Chroma(
            persist_directory=settings.CHROMA_DB_DIR,
            embedding_function=embeddings,
            collection_name=settings.COLLECTION_NAME,
        )

    if not chunks:
        raise ValueError("No chunks provided and Vector Store does not exist on disk.")

    batch_size = 10
    delay_seconds = 35
    vector_store = None
    total_batches = (len(chunks) + batch_size - 1) // batch_size

    print(f"\nBuilding Vector Store: {len(chunks)} chunks in {total_batches} batches...")

    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        print(f"Embedding Batch {batch_num}/{total_batches} ({len(batch)} chunks)...")

        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=batch,
                embedding=embeddings,
                persist_directory=settings.CHROMA_DB_DIR,
                collection_name=settings.COLLECTION_NAME,
            )
        else:
            vector_store.add_documents(documents=batch)

        if batch_num < total_batches:
            time.sleep(delay_seconds)

    return vector_store


class HybridRetriever:
    """BM25 + Vector DB + Cross-Encoder Reranker pipeline."""

    def __init__(self, vector_db: Chroma, chunks: list[Document]):
        self.vector_db = vector_db
        print("Indexing chunks for BM25 Sparse Search...")
        self.bm25_retriever = BM25Retriever.from_documents(chunks)
        self.bm25_retriever.k = 10

    def retrieve_and_rerank(
        self, query: str, top_k: int = 3, fetch_k: int = 10
    ) -> list[Document]:
        # 1. Vector Search
        vector_results = self.vector_db.similarity_search(query, k=fetch_k)

        # 2. BM25 Keyword Search
        bm25_results = self.bm25_retriever.invoke(query)

        # 3. Deduplicate Chunks
        combined_docs = []
        seen_contents = set()

        for doc in vector_results + bm25_results:
            if doc.page_content not in seen_contents:
                combined_docs.append(doc)
                seen_contents.add(doc.page_content)

        # 4. Cross-Encoder Reranking
        reranker = get_reranker()
        pairs = [[query, doc.page_content] for doc in combined_docs]
        scores = reranker.predict(pairs)

        # Sort by reranker score descending
        doc_score_pairs = list(zip(combined_docs, scores))
        doc_score_pairs.sort(key=lambda x: x[1], reverse=True)

        # Top K selection
        reranked_docs = [doc for doc, score in doc_score_pairs[:top_k]]
        return reranked_docs


if __name__ == "__main__":
    from app.parser import chunk_documents, load_pdf

    # Load docs locally for BM25 index setup
    raw_docs = load_pdf()
    chunks = chunk_documents(raw_docs)

    # Fast load persisted vector store from disk
    vector_db = build_or_load_vector_store()

    # Initialize Hybrid Retriever
    hybrid_retriever = HybridRetriever(vector_db, chunks)

    # Test Query
    query = "What is Apple's total revenue or net sales for 2025?"
    print(f"\nExecuting Hybrid Search + Reranking for Query: '{query}'...")
    top_docs = hybrid_retriever.retrieve_and_rerank(query, top_k=3)

    print("\n--- Top Reranked Retrieval Results ---")
    for idx, doc in enumerate(top_docs, 1):
        print(f"\n[Rank {idx}] Page {doc.metadata.get('page')}:")
        print(f"{doc.page_content[:300]}...")