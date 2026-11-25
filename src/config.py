"""Configuracion central del agente RecetaYa.

Lee el archivo .env y expone:
- rutas del proyecto (fichas, Chroma y API sanitaria simulada);
- parametros del agente (iteraciones, replanificaciones, inventario caido);
- fabricas crear_llm() y crear_embeddings() que respetan PROVEEDOR.

Las fabricas existen para no crear clientes al importar el modulo: asi los
tests y la carga de datos no necesitan red ni una API key.
"""

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel

# Ruta raiz del repo calculada desde este archivo, para que funcione igual si
# se ejecuta desde la raiz, desde tests/ o desde cualquier otro directorio.
RAIZ = Path(__file__).resolve().parent.parent
load_dotenv(RAIZ / ".env")

DIR_FICHAS = RAIZ / "data" / "fichas"
DIR_CHROMA = RAIZ / "data" / "chroma"
RUTA_API_SANITARIA = RAIZ / "data" / "externo" / "api_sanitaria.json"

PROVEEDOR = os.getenv("PROVEEDOR", "gemini").strip().lower()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

# Falla temprano si el .env tiene un proveedor mal escrito, en vez de fallar
# mas adelante al llamar al LLM.
if PROVEEDOR not in {"gemini", "ollama"}:
    raise ValueError(
        f"PROVEEDOR invalido: {PROVEEDOR!r}. Usa 'gemini' u 'ollama'."
    )

MODELO_LLM = os.getenv(
    "MODELO_LLM",
    "gemini-3.5-flash-lite" if PROVEEDOR == "gemini" else "llama3.2",
)
MODELO_EMBEDDINGS = os.getenv("MODELO_EMBEDDINGS", "gemini-embedding-001")


def _entero(nombre: str, defecto: int) -> int:
    """Lee una variable entera del .env y explica el error si no es valida."""
    valor = os.getenv(nombre, str(defecto))
    try:
        return int(valor)
    except ValueError as error:
        raise ValueError(f"{nombre} debe ser un numero entero, no {valor!r}") from error


DIM_EMBEDDINGS = _entero("DIM_EMBEDDINGS", 768)
MAX_ITERACIONES = _entero("MAX_ITERACIONES", 6)
MAX_REPLANIFICACIONES = _entero("MAX_REPLANIFICACIONES", 1)
# El .env usa 0/1; se convierte a bool para que las herramientas lean claro.
INVENTARIO_CAIDO = os.getenv("INVENTARIO_CAIDO", "0").strip() == "1"


def _exigir_api_key() -> None:
    """Solo Gemini necesita key; Ollama corre local y no usa esta variable."""
    if not GOOGLE_API_KEY:
        raise RuntimeError(
            "Falta GOOGLE_API_KEY en el entorno. Copia .env.example a .env "
            "y completa tu clave de Google AI Studio."
        )


def crear_llm() -> BaseChatModel:
    """Crea el LLM del proveedor configurado en el .env.

    Gemini: temperature=0 y max_retries=3 como pide el enunciado.
    Ollama: plan B local; no se pasa max_retries porque no aplica a un servidor local.
    """
    if PROVEEDOR == "gemini":
        _exigir_api_key()
        return init_chat_model(
            model=MODELO_LLM,
            model_provider="google_genai",
            api_key=GOOGLE_API_KEY,
            temperature=0,
            max_retries=3,
        )

    # Plan B: el modelo local se define en MODELO_LLM (por defecto llama3.2).
    return init_chat_model(
        model=MODELO_LLM,
        model_provider="ollama",
        temperature=0,
    )


def crear_embeddings() -> Embeddings:
    """Crea los embeddings del proveedor configurado.

    Gemini: gemini-embedding-001 con dimension configurable (768 por defecto).
    Ollama: nomic-embed-text, tal como fija el enunciado para el plan B.
    """
    if PROVEEDOR == "gemini":
        _exigir_api_key()
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            model=MODELO_EMBEDDINGS,
            google_api_key=GOOGLE_API_KEY,
            output_dimensionality=DIM_EMBEDDINGS,
        )

    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(model="nomic-embed-text")
