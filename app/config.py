import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Updated File Path
    PDF_PATH: str = "data/sample.pdf"
    
    # Chunking Configs
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # Vector DB Config
    CHROMA_DB_DIR: str = "data/chroma_db"
    COLLECTION_NAME: str = "apple_10k"

settings = Settings()