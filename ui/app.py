import requests
import streamlit as st

st.set_page_config(
    page_title="Apple 10-K RAG Assistant", page_icon="🍎", layout="wide"
)

st.title("🍎 Apple Inc. 10-K Financial Assistant")
st.markdown(
    "Ask natural language financial questions powered by **Hybrid Retrieval (BM25 + ChromaDB)** & **Gemini 3.5 Flash Lite**."
)

API_URL = "http://127.0.0.1:8000/query"

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    top_k = st.slider(
        "Top Chunks to Retrieve (top_k)", min_value=1, max_value=10, value=3
    )
    st.markdown("---")
    st.markdown("### 🏗️ System Architecture")
    st.markdown("""
    - **Parser:** PyMuPDF (`fitz`)
    - **Vector DB:** ChromaDB
    - **Sparse Search:** BM25
    - **Reranker:** Cross-Encoder (`ms-marco-MiniLM-L-6-v2`)
    - **LLM:** Gemini 3.5 Flash Lite
    """)

# User Query Input
user_query = st.text_input(
    "Enter your question about Apple's 10-K filing:",
    placeholder="e.g., What was Apple's total net sales for iPhone in 2025?",
)

if st.button("Submit Query", type="primary"):
    if not user_query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching document & generating grounded answer..."):
            try:
                response = requests.post(
                    API_URL,
                    json={"query": user_query, "top_k": top_k},
                    timeout=60,
                )

                if response.status_code == 200:
                    data = response.json()

                    st.subheader("💡 Answer")
                    st.markdown(data["answer"])

                    st.markdown("---")
                    st.subheader("📌 Cited Source Contexts")

                    for idx, src in enumerate(data.get("sources", []), 1):
                        page_num = src.get("page", "N/A")
                        file_name = src.get("file_name", "sample.pdf")
                        snippet = src.get("snippet", "")

                        with st.expander(f"Source {idx}: Page {page_num} ({file_name})"):
                            st.write(f"**Snippet Preview:**\n{snippet}")
                else:
                    st.error(
                        f"API Error ({response.status_code}): {response.text}"
                    )
            except requests.exceptions.ConnectionError:
                st.error(
                    "❌ Could not connect to FastAPI backend. Ensure `uvicorn app.main:app` is running on port 8000."
                )
            except Exception as e:
                st.error(f"An unexpected error occurred: {str(e)}")