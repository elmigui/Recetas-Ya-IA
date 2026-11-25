# Recetas-Ya-IA

Proyecto de IA construido con un entorno virtual de Python, LangGraph, LangChain, Google Gemini (GenAI) y ChromaDB.

## Requisitos

- Python 3.13
- Una `GOOGLE_API_KEY` de Google AI Studio

## Instalación

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Variables de entorno

Crear un archivo `.env` en la raíz (no se sube a git):

```env
GOOGLE_API_KEY=tu_api_key
MODELO_LLM=gemini-3.5-flash-lite
MODELO_EMBEDDINGS=gemini-embedding-001
```

## Estructura del proyecto

```text
Recetas-Ya-IA/
├── data/
│   ├── db/          # FUENTE INTERNA: datos locales / base vectorial Chroma
│   └── externo/     # FUENTE EXTERNA: simulacion de API
├── src/             # Codigo fuente
│   ├── config.py    # Carga de variables de entorno
│   └── llm.py       # Cliente LLM (ChatGoogleGenerativeAI)
├── test/            # Pruebas con pytest
├── .env             # Variables de entorno (ignorado por git)
├── requirements.txt
└── venv/            # Entorno virtual (ignorado por git)
```

## Comprobación de conexión

```powershell
.\venv\Scripts\python.exe -m pytest -m integration -s
```

## Pruebas

```powershell
.\venv\Scripts\python.exe -m pytest
```
