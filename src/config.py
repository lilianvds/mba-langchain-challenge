import os
from dotenv import load_dotenv

load_dotenv()

# Database configuration
DB_URL = os.getenv("DATABASE_URL")
if not DB_URL:
    raise ValueError("A variável DATABASE_URL não está configurada no .env")

# LLM Provider configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower() # Default to gemini

# Model names
EMBEDDING_MODEL = ""
LLM_MODEL = ""

if LLM_PROVIDER == "gemini":
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("A variável GOOGLE_API_KEY não está configurada no .env para o provedor Gemini.")
    EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-001") 
    LLM_MODEL = os.getenv("GEMINI_LLM_MODEL", "gemini-2.5-flash-lite")
elif LLM_PROVIDER == "openai":
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("A variável OPENAI_API_KEY não está configurada no .env para o provedor OpenAI.")
    EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    LLM_MODEL = os.getenv("OPENAI_LLM_MODEL", "gpt-3.5-turbo")
else:
    raise ValueError("LLM_PROVIDER deve ser 'gemini' ou 'openai'.")

# Ingestion constants
COLLECTION_NAME = "mba_pdf_collection"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150