import os

from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODELO_LLM = os.getenv("MODELO_LLM", "gemini-3.5-flash-lite")
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "gemini-embedding-001")

if not GOOGLE_API_KEY:
    raise RuntimeError("Falta GOOGLE_API_KEY en el entorno (.env)")
