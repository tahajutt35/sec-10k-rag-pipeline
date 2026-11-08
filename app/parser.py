import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import settings


def load_pdf(file_path: str = settings.PDF_PATH) -> list[Document]:
    """PyMuPDF se fast parsing karta hai aur metadata dynamically attach karta hai."""
    print(f"Loading document with PyMuPDF from: {file_path}")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()

    file_name = os.path.basename(file_path)

    for doc in documents:
        doc.metadata["file_name"] = file_name
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1

    print(f"Successfully loaded {len(documents)} pages from {file_name}.")
    return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = settings.CHUNK_SIZE,
    chunk_overlap: int = settings.CHUNK_OVERLAP,
) -> list[Document]:
    """Loaded documents ko overlap ke sath split karta hai."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"Generated {len(chunks)} chunks from {len(documents)} pages.")
    return chunks


if __name__ == "__main__":
    raw_docs = load_pdf()
    chunks = chunk_documents(raw_docs)

    print("\n--- Sample Chunk Output ---")
    print(f"Metadata: {chunks[0].metadata}")
    print(f"Content Preview:\n{chunks[0].page_content[:300]}...")