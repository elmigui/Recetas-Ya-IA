from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import GOOGLE_API_KEY, MODELO_LLM


def get_llm() -> ChatGoogleGenerativeAI:
    return ChatGoogleGenerativeAI(
        model=MODELO_LLM,
        google_api_key=GOOGLE_API_KEY,
    )
