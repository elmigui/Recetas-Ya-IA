"""Pruebas de la logica del grafo (Fase 5) que no llaman al LLM.

Cubren el enmascaramiento de datos personales y la decision de replanificar.
La ejecucion completa del agente se prueba con `python src/grafo.py`.
"""

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from src import config
from src.grafo import (
    _enmascarar,
    _parsear_hechos,
    decidir_tras_herramientas,
    decidir_tras_revisar,
)


def test_enmascarar_rut_y_correo():
    texto = "RUT 11.111.111-1, otro 22222222-2, correo ana@example.cl"
    limpio = _enmascarar(texto)

    assert "11.111.111-1" not in limpio
    assert "22222222-2" not in limpio
    assert "ana@example.cl" not in limpio
    assert limpio.count("[DATO OCULTO]") == 3


def test_enmascarar_no_cambia_texto_normal():
    texto = "Stock de Z en Centro: 15 unidades."
    assert _enmascarar(texto) == texto


def _estado_con_error(replanificaciones: int) -> dict:
    return {
        "messages": [
            HumanMessage(content="¿Puedo dispensar la R-1050?"),
            ToolMessage(
                content="Servicio de Inventario no disponible",
                tool_call_id="1",
                status="error",
            ),
        ],
        "replanificaciones": replanificaciones,
    }


def test_decidir_tras_revisar_replanifica_la_primera_vez(monkeypatch):
    monkeypatch.setattr(config, "MAX_REPLANIFICACIONES", 1)
    assert decidir_tras_revisar(_estado_con_error(0)) == "planificar"


def test_decidir_tras_revisar_responde_al_limite(monkeypatch):
    monkeypatch.setattr(config, "MAX_REPLANIFICACIONES", 1)
    assert decidir_tras_revisar(_estado_con_error(1)) == "responder"


def test_decidir_tras_revisar_sin_error_responde():
    estado = {"messages": [HumanMessage(content="Hola")], "replanificaciones": 0}
    assert decidir_tras_revisar(estado) == "responder"


def _estado_con_tool(estado_tool: str) -> dict:
    return {
        "messages": [
            HumanMessage(content="¿Puedo dispensar la R-1050?"),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "consultar_stock",
                        "args": {"principio_activo": "V", "sucursal": "Centro"},
                        "id": "llamada-1",
                    }
                ],
            ),
            ToolMessage(
                content="Servicio de Inventario no disponible",
                tool_call_id="llamada-1",
                status=estado_tool,
            ),
        ],
    }


def test_decidir_tras_herramientas_va_a_revisar_con_error():
    assert decidir_tras_herramientas(_estado_con_tool("error")) == "revisar"


def test_decidir_tras_herramientas_vuelve_al_agente_sin_error():
    assert decidir_tras_herramientas(_estado_con_tool("success")) == "agente"


def test_parsear_hechos_limpia_ninguno_y_duplicados():
    texto = (
        "NINGUNO\n"
        "- Trabajo en la sucursal Centro\n"
        "2. Trabajo en la sucursal Centro\n"
        "Prefiero los turnos de mañana"
    )
    assert _parsear_hechos(texto) == [
        "Trabajo en la sucursal Centro",
        "Prefiero los turnos de mañana",
    ]
