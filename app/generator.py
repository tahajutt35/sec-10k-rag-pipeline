from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from app.config import settings


def get_llm():
    """Gemini 3.5 Flash Lite initialize karta hai (Warning-free setup)."""
    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=settings.GOOGLE_API_KEY,
    )


PROMPT_TEMPLATE = """You are an expert financial analyst assistant specializing in Apple Inc.'s 10-K SEC filings.
Answer the question based STRICTLY on the provided context below.

Guidelines:
1. Base your answer ONLY on the provided context chunks. Do NOT assume or hallucinate information.
2. If the context does not contain enough information to answer the question, clearly state "Information not found in the provided 10-K filing."
3. Always extract exact financial figures, dollar amounts, and percentages when available.
4. Reference page numbers explicitly when citing facts.

Context:
{context}

Question: {question}

Detailed Answer:"""


def generate_answer(query: str, retrieved_docs: list[Document]) -> dict:
    """Retrieved docs ko context string mein assemble karke Gemini LLM se clean structured answer leta hai."""
    context_str = ""
    for idx, doc in enumerate(retrieved_docs, 1):
        page_num = doc.metadata.get("page", "N/A")
        context_str += (
            f"\n--- Document Chunk {idx} (Page {page_num}) ---\n{doc.page_content}\n"
        )

    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE, input_variables=["context", "question"]
    )

    llm = get_llm()
    chain = prompt | llm

    print(f"Generating answer using Gemini 3.5 Flash Lite for query: '{query}'...")
    response = chain.invoke({"context": context_str, "question": query})

    # Clean array/dict output from Gemini response into a standard string
    if isinstance(response.content, list):
        answer_text = "".join(
            [
                item.get("text", "")
                for item in response.content
                if isinstance(item, dict) and item.get("type") == "text"
            ]
        )
    else:
        answer_text = str(response.content)

    sources = [
        {
            "page": doc.metadata.get("page"),
            "file_name": doc.metadata.get("file_name", "sample.pdf"),
            "snippet": doc.page_content[:150] + "...",
        }
        for doc in retrieved_docs
    ]

    return {
        "query": query,
        "answer": answer_text,
        "sources": sources,
    }


if __name__ == "__main__":
    from app.parser import chunk_documents, load_pdf
    from app.retriever import HybridRetriever, build_or_load_vector_store

    raw_docs = load_pdf()
    chunks = chunk_documents(raw_docs)
    vector_db = build_or_load_vector_store()

    retriever = HybridRetriever(vector_db, chunks)

    test_query = "What was Apple's total net sales for iPhone in 2025 and 2024?"
    retrieved_chunks = retriever.retrieve_and_rerank(test_query, top_k=3)

    result = generate_answer(test_query, retrieved_chunks)

    print("\n================ FINAL GENERATED ANSWER ================")
    print(result["answer"])
    print("\n================ SOURCES CITED ================")
    for src in result["sources"]:
        print(f"• Page {src['page']} ({src['file_name']})")