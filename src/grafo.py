"""Grafo del agente RecetaYa (Fase 6).

Flujo:
START -> recuperar_memoria -> planificar -> agente
      -> (herramientas -> agente)* -> revisar
      -> (planificar | responder) -> guardar_memoria -> END
Si una herramienta devuelve error, se sale del bucle hacia revisar de
inmediato; asi el agente no reintenta la misma herramienta caida y puede
replanificar con la informacion del error.

- recuperar_memoria: trae hasta 3 hechos del usuario desde Chroma.
- planificar: pide al LLM un plan de 2 a 5 pasos con herramientas.
- agente: ejecuta el plan conversando con el LLM y las herramientas.
- herramientas: ToolNode; los errores se convierten en ToolMessage con
  status="error" gracias a handle_tool_errors=True.
- revisar: detecta el error del turno y decide si se replanifica.
- responder: publica la respuesta final con RUT y correos enmascarados.
- guardar_memoria: extrae y guarda hechos operativos estables del usuario.
"""

import operator
import re
import sys
from pathlib import Path
from typing import Annotated, TypedDict

# Permite ejecutar `python src/grafo.py` desde la raiz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langgraph.checkpoint.memory import InMemorySaver  # noqa: E402
from langgraph.graph import END, START, StateGraph, add_messages  # noqa: E402
from langgraph.prebuilt import ToolNode  # noqa: E402

from src import config, memoria  # noqa: E402
from src.herramientas import HERRAMIENTAS  # noqa: E402
from src.prompts import (  # noqa: E402
    PROMPT_MEMORIA,
    PROMPT_PLANIFICACION,
    PROMPT_SISTEMA,
)

# Patrones para datos personales que no deben salir en la respuesta final.
PATRON_RUT = re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-[\dkK]\b")
PATRON_RUT_SIN_PUNTOS = re.compile(r"\b\d{7,8}-[\dkK]\b")
PATRON_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
TEXTO_OCULTO = "[DATO OCULTO]"


class Estado(TypedDict, total=False):
    """Estado compartido por todos los nodos del grafo."""

    # Historial del hilo; add_messages agrega en vez de reemplazar.
    messages: Annotated[list[AnyMessage], add_messages]
    # Contexto del farmaceutico (no datos de pacientes).
    user_id: str
    sucursal: str
    # Plan del turno y hechos de memoria de largo plazo.
    plan: list[str]
    memoria: list[str]
    # Contadores y explicacion de lo que hizo cada nodo.
    iteraciones: int
    replanificaciones: int
    decisiones: Annotated[list[str], operator.add]


def _texto_respuesta(mensaje) -> str:
    """Extrae texto de un mensaje.

    Gemini puede devolver content como str o como lista de bloques; esta
    funcion unifica ambos casos para imprimir y para enmascarar.
    """
    contenido = mensaje.content
    if isinstance(contenido, str):
        return contenido
    partes = []
    for bloque in contenido or []:
        if isinstance(bloque, str):
            partes.append(bloque)
        elif isinstance(bloque, dict) and bloque.get("text"):
            partes.append(bloque["text"])
    return "\n".join(partes).strip()


def _mensajes_sistema(estado: Estado) -> SystemMessage:
    """Arma el prompt de sistema con el contexto y el plan del turno."""
    lineas = [PROMPT_SISTEMA]
    if estado.get("user_id"):
        lineas.append(f"Farmaceutico actual: {estado['user_id']}.")
    if estado.get("sucursal"):
        lineas.append(f"Sucursal habitual: {estado['sucursal']}.")
    if estado.get("plan"):
        lineas.append("Plan del turno:\n" + "\n".join(estado["plan"]))
    if estado.get("memoria"):
        lineas.append("Memoria del usuario:\n" + "\n".join(estado["memoria"]))
    return SystemMessage(content="\n\n".join(lineas))


def _ultima_pregunta(mensajes: list[AnyMessage]) -> str:
    """Devuelve el texto de la ultima pregunta del farmaceutico."""
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, HumanMessage):
            return _texto_respuesta(mensaje)
    return ""


def _error_del_turno(mensajes: list[AnyMessage]) -> ToolMessage | None:
    """Busca el ultimo ToolMessage con status error del turno actual.

    Solo se mira desde la ultima pregunta humana; asi un error de un turno
    anterior no dispara una replanificacion en el turno nuevo.
    """
    inicio = 0
    for indice in range(len(mensajes) - 1, -1, -1):
        if isinstance(mensajes[indice], HumanMessage):
            inicio = indice
            break
    for mensaje in reversed(mensajes[inicio:]):
        if isinstance(mensaje, ToolMessage) and mensaje.status == "error":
            return mensaje
    return None


def _enmascarar(texto: str) -> str:
    """Reemplaza RUT y correos por un texto seguro."""
    texto = PATRON_RUT.sub(TEXTO_OCULTO, texto)
    texto = PATRON_RUT_SIN_PUNTOS.sub(TEXTO_OCULTO, texto)
    return PATRON_EMAIL.sub(TEXTO_OCULTO, texto)


def _parsear_plan(texto: str) -> list[str]:
    """Extrae los pasos numerados del plan; si no hay, usa el texto completo."""
    pasos = []
    for linea in texto.splitlines():
        coincidencia = re.match(r"\s*\d+[.)]\s*(.+)", linea)
        if coincidencia:
            pasos.append(coincidencia.group(1).strip())
    if not pasos and texto.strip():
        pasos = [texto.strip()]
    return pasos[:5]


def _parsear_hechos(texto: str) -> list[str]:
    """Limpia la salida del extractor de memoria: un hecho por linea.

    Ignora NINGUNO y los duplicados exactos dentro del mismo turno.
    """
    hechos = []
    for linea in texto.splitlines():
        limpio = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", linea).strip()
        if not limpio or limpio.upper() == "NINGUNO":
            continue
        if limpio not in hechos:
            hechos.append(limpio)
    return hechos


def decidir_siguiente(estado: Estado) -> str:
    """Decide si el agente sigue usando herramientas o pasa a revisar.

    Se implementa a mano (en vez de tools_condition) para que la regla de
    MAX_ITERACIONES quede visible y sea facil de explicar.
    """
    ultimo = estado["messages"][-1]
    if isinstance(ultimo, AIMessage) and ultimo.tool_calls:
        if estado.get("iteraciones", 0) < config.MAX_ITERACIONES:
            return "herramientas"
    return "revisar"


def _error_en_ultimas_herramientas(mensajes: list[AnyMessage]) -> bool:
    """Indica si la ultima tanda de herramientas termino con error."""
    for mensaje in reversed(mensajes):
        if isinstance(mensaje, AIMessage):
            break
        if isinstance(mensaje, ToolMessage) and mensaje.status == "error":
            return True
    return False


def decidir_tras_herramientas(estado: Estado) -> str:
    """Si la herramienta fallo, va a revisar; si no, vuelve al agente.

    Se corta el bucle en el primer error para que el agente no insista con un
    servicio caido; revisar decidira si corresponde replanificar.
    """
    if _error_en_ultimas_herramientas(estado["messages"]):
        return "revisar"
    return "agente"


def decidir_tras_revisar(estado: Estado) -> str:
    """Replanifica una vez si hubo error; si no, responde."""
    error = _error_del_turno(estado["messages"])
    ultimo = estado["messages"][-1]
    if (
        error is not None
        # El ultimo mensaje debe ser el resultado de una herramienta; si fuera
        # un AIMessage con tool_calls sin resolver, Gemini rechaza el turno.
        and isinstance(ultimo, ToolMessage)
        and estado.get("replanificaciones", 0) < config.MAX_REPLANIFICACIONES
    ):
        return "planificar"
    return "responder"


def revisar(estado: Estado) -> dict:
    """Registra el resultado del turno y si hubo un error de herramienta."""
    error = _error_del_turno(estado["messages"])
    if error is None:
        return {"decisiones": ["Revisar: sin errores en las herramientas."]}

    detalle = _texto_respuesta(error)[:200]
    nombre = getattr(error, "name", None) or "herramienta"
    if estado.get("replanificaciones", 0) < config.MAX_REPLANIFICACIONES:
        decision = f"Revisar: error en {nombre}: {detalle} Se replanifica."
    else:
        decision = (
            f"Revisar: error en {nombre}: {detalle} "
            "Sin replanificaciones disponibles; se responde."
        )
    return {"decisiones": [decision]}


def recuperar_memoria(estado: Estado) -> dict:
    """Recupera hasta 3 hechos del usuario y los deja en el estado."""
    hechos = memoria.recuperar(
        estado.get("user_id", ""),
        _ultima_pregunta(estado["messages"]),
        k=3,
    )
    if hechos:
        decision = f"Memoria: {len(hechos)} hechos recuperados."
    else:
        decision = "Memoria: sin hechos previos para este usuario."
    return {"memoria": hechos, "decisiones": [decision]}


def crear_grafo():
    """Construye y compila el grafo con memoria de corto y largo plazo."""
    llm = config.crear_llm()
    llm_con_herramientas = llm.bind_tools(HERRAMIENTAS)

    def planificar(estado: Estado) -> dict:
        """Genera el plan del turno (o el plan alternativo tras un error)."""
        mensajes = estado["messages"]

        # Un turno nuevo reinicia el contador; una replanificacion lo aumenta.
        reinicio = isinstance(mensajes[-1], HumanMessage)
        replanificaciones = (
            0 if reinicio else estado.get("replanificaciones", 0) + 1
        )

        contexto = [f"Pregunta del farmaceutico: {_ultima_pregunta(mensajes)}"]
        if estado.get("user_id"):
            contexto.append(f"Farmaceutico actual: {estado['user_id']}.")
        if estado.get("sucursal"):
            contexto.append(f"Sucursal habitual: {estado['sucursal']}.")
        if estado.get("memoria"):
            contexto.append("Memoria del usuario:\n" + "\n".join(estado["memoria"]))

        error = _error_del_turno(mensajes)
        if error is not None:
            detalle_error = _texto_respuesta(error)
            contexto.append(
                "El plan anterior fallo con este error: "
                f"{detalle_error}. Propon un plan alternativo."
            )
            # Si lo que fallo fue Inventario, el plan alternativo debe usar
            # las herramientas que el enunciado define para ese caso.
            if "Inventario" in detalle_error:
                contexto.append(
                    "No vuelvas a usar consultar_stock. Usa estado_reserva "
                    "para confirmar el estado y buscar_alternativa para el stock."
                )

        respuesta = llm.invoke(
            [
                SystemMessage(content=PROMPT_PLANIFICACION),
                HumanMessage(content="\n\n".join(contexto)),
            ]
        )
        pasos = _parsear_plan(_texto_respuesta(respuesta))
        if not pasos:
            pasos = ["Consultar la receta y responder con los datos disponibles."]

        plan_texto = "\n".join(
            f"{numero}. {paso}" for numero, paso in enumerate(pasos, 1)
        )
        etiqueta = (
            f"Replanificacion {replanificaciones}/{config.MAX_REPLANIFICACIONES}"
            if replanificaciones
            else "Planificacion"
        )
        return {
            "plan": pasos,
            "replanificaciones": replanificaciones,
            "decisiones": [f"{etiqueta}:\n{plan_texto}"],
        }

    def agente(estado: Estado) -> dict:
        """Llama al LLM con el historial y registra lo que decidio."""
        mensajes = estado["messages"]

        # El checkpointer conserva el estado entre invocaciones del mismo hilo.
        # Si el turno empieza con una pregunta nueva, se reinician los contadores
        # para que MAX_ITERACIONES se aplique a este turno y no al historial.
        reinicio = isinstance(mensajes[-1], HumanMessage)
        iteraciones = 0 if reinicio else estado.get("iteraciones", 0)
        replanificaciones = 0 if reinicio else estado.get("replanificaciones", 0)

        respuesta = llm_con_herramientas.invoke(
            [_mensajes_sistema(estado)] + mensajes
        )
        if respuesta.tool_calls:
            iteraciones += 1
            nombres = ", ".join(llamada["name"] for llamada in respuesta.tool_calls)
            decision = f"Agente: solicita {nombres} (iteracion {iteraciones})."
        else:
            decision = "Agente: responde sin solicitar herramientas."

        return {
            "messages": [respuesta],
            "iteraciones": iteraciones,
            "replanificaciones": replanificaciones,
            "decisiones": [decision],
        }

    def responder(estado: Estado) -> dict:
        """Publica la ultima respuesta del agente con datos personales ocultos."""
        ultima_ai = None
        for mensaje in reversed(estado["messages"]):
            if isinstance(mensaje, AIMessage):
                ultima_ai = mensaje
                break

        if ultima_ai is None:
            return {"decisiones": ["Responder: no hay respuesta del agente."]}

        texto = _texto_respuesta(ultima_ai)
        if not texto:
            # Si el agente se quedo sin iteraciones con un tool_call pendiente,
            # se usa el ultimo resultado exitoso para no devolver algo vacio.
            for mensaje in reversed(estado["messages"]):
                if isinstance(mensaje, ToolMessage) and mensaje.status != "error":
                    texto = (
                        "No se pudo generar una respuesta final del agente. "
                        f"Ultimo dato disponible: {_texto_respuesta(mensaje)}"
                    )
                    break
        texto = texto or "No se genero una respuesta final."
        texto_seguro = _enmascarar(texto)

        # model_copy conserva el id; add_messages reemplaza el mensaje original
        # en vez de dejar la version con datos personales en el historial.
        respuesta = ultima_ai.model_copy(
            update={"content": texto_seguro, "tool_calls": []}
        )
        return {
            "messages": [respuesta],
            "decisiones": [
                "Responder: respuesta final con RUT y correos enmascarados."
            ],
        }

    def guardar_memoria(estado: Estado) -> dict:
        """Extrae hechos operativos estables del turno y los guarda en Chroma."""
        user_id = estado.get("user_id")
        if not user_id:
            return {"decisiones": ["Memoria: sin user_id, no se guarda nada."]}

        # Solo se usa el mensaje del usuario: la respuesta del agente puede
        # contener sucursales inferidas de una receta y no son hechos del usuario.
        contexto = (
            f"Farmaceutico: {user_id}\n"
            f"Mensaje: {_ultima_pregunta(estado['messages'])}"
        )
        salida = llm.invoke(
            [
                SystemMessage(content=PROMPT_MEMORIA),
                HumanMessage(content=contexto),
            ]
        )
        hechos = _parsear_hechos(_texto_respuesta(salida))
        guardados = memoria.guardar(user_id, hechos)
        if guardados:
            decision = f"Memoria: se guardaron {guardados} hechos."
        else:
            decision = "Memoria: no se guardaron hechos nuevos."
        return {"decisiones": [decision]}

    builder = StateGraph(Estado)
    builder.add_node("recuperar_memoria", recuperar_memoria)
    builder.add_node("planificar", planificar)
    builder.add_node("agente", agente)
    builder.add_node(
        "herramientas", ToolNode(HERRAMIENTAS, handle_tool_errors=True)
    )
    builder.add_node("revisar", revisar)
    builder.add_node("responder", responder)
    builder.add_node("guardar_memoria", guardar_memoria)

    builder.add_edge(START, "recuperar_memoria")
    builder.add_edge("recuperar_memoria", "planificar")
    builder.add_edge("planificar", "agente")
    builder.add_conditional_edges(
        "agente",
        decidir_siguiente,
        {"herramientas": "herramientas", "revisar": "revisar"},
    )
    builder.add_conditional_edges(
        "herramientas",
        decidir_tras_herramientas,
        {"agente": "agente", "revisar": "revisar"},
    )
    builder.add_conditional_edges(
        "revisar",
        decidir_tras_revisar,
        {"planificar": "planificar", "responder": "responder"},
    )
    builder.add_edge("responder", "guardar_memoria")
    builder.add_edge("guardar_memoria", END)

    # En LangGraph 1.x el checkpointer en memoria se llama InMemorySaver
    # (antes MemorySaver). Cumple la misma funcion: estado por thread_id.
    return builder.compile(checkpointer=InMemorySaver())


if __name__ == "__main__":
    # La consola de Windows no siempre usa UTF-8; esto evita acentos rotos.
    sys.stdout.reconfigure(encoding="utf-8")

    app = crear_grafo()

    # Turno 1: en un hilo el farmaceutico declara su sucursal.
    print("Turno 1: 'Trabajo en la sucursal Centro.' (hilo declaracion)")
    estado = app.invoke(
        {
            "messages": [HumanMessage(content="Trabajo en la sucursal Centro.")],
            "user_id": "farm-01",
        },
        {"configurable": {"thread_id": "memoria-declaracion"}},
    )
    print("Respuesta:", _texto_respuesta(estado["messages"][-1]))
    for decision in estado["decisiones"]:
        print(f"- {decision}")
    print()

    # Turno 2: hilo NUEVO. La sucursal solo puede venir de la memoria.
    print("Turno 2: '¿Hay stock de Z?' (thread_id nuevo)")
    estado = app.invoke(
        {
            "messages": [HumanMessage(content="¿Hay stock de Z?")],
            "user_id": "farm-01",
        },
        {"configurable": {"thread_id": "memoria-consulta"}},
    )
    print("Memoria recuperada:", estado.get("memoria"))
    print("Respuesta:", _texto_respuesta(estado["messages"][-1]))

    print("\nDecisiones:")
    for decision in estado["decisiones"]:
        print(f"- {decision}")

    print("\nHerramientas usadas:")
    for mensaje in estado["messages"]:
        if isinstance(mensaje, AIMessage):
            for llamada in mensaje.tool_calls:
                print(f"- {llamada['name']} {llamada['args']}")
        elif isinstance(mensaje, ToolMessage):
            print(f"  -> {_texto_respuesta(mensaje)}")
