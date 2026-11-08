import json
from app.parser import chunk_documents, load_pdf
from app.retriever import HybridRetriever, build_or_load_vector_store


def run_evaluation():
    # Load Golden Dataset
    with open("eval/golden_dataset.json", "r") as f:
        golden_data = json.load(f)

    # Initialize Retriever Pipeline
    raw_docs = load_pdf()
    chunks = chunk_documents(raw_docs)
    vector_db = build_or_load_vector_store()
    retriever = HybridRetriever(vector_db, chunks)

    hits = 0
    total_queries = len(golden_data)

    print("\n--- Running RAG Retrieval Evaluation ---")

    for idx, item in enumerate(golden_data, 1):
        query = item["query"]
        expected_page = item["expected_page"]

        retrieved_docs = retriever.retrieve_and_rerank(query, top_k=3)
        retrieved_pages = [doc.metadata.get("page") for doc in retrieved_docs]

        is_hit = expected_page in retrieved_pages
        if is_hit:
            hits += 1

        print(f"\n[Test {idx}/{total_queries}] Query: '{query}'")
        print(f"  • Expected Page: {expected_page}")
        print(f"  • Retrieved Pages: {retrieved_pages}")
        print(f"  • Status: {'✅ HIT' if is_hit else '❌ MISS'}")

    hit_rate = (hits / total_queries) * 100
    print("\n================ EVALUATION SUMMARY ================")
    print(f"Total Test Cases: {total_queries}")
    print(f"Retrieval Recall@3 (Hit Rate): {hit_rate:.2f}%")
    print("====================================================\n")


if __name__ == "__main__":
    run_evaluation()