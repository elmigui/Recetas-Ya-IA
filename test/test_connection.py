import pytest

from src.llm import get_llm


@pytest.mark.integration
def test_llm_connection():
    respuesta = get_llm().invoke("Responde unicamente con la palabra: hola")
    assert respuesta.content
    print(f"\nRespuesta del modelo: {respuesta.content}")
