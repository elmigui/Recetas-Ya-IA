"""Demo guionada de RecetaYa (Fase 7).

Ejecuta 5 casos, imprime la respuesta y las decisiones de cada uno, y deja una
pausa entre casos para respetar el limite del nivel gratuito de Gemini.

La demo es autocontenida: recarga los datos ficticios, reindexa las fichas y
limpia solo la memoria del usuario de demo, para poder repetirla sin arrastrar
estado de ejecuciones anteriores.
"""

import sys
import time
import warnings
from pathlib import Path

# Permite ejecutar `python src/demo.py` desde la raiz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain_core.messages import HumanMessage  # noqa: E402

from src import cargar_datos, config, db, memoria, rag  # noqa: E402
from src.grafo import _texto_respuesta, crear_grafo  # noqa: E402

USUARIO_DEMO = "farm-demo"
PAUSA_SEGUNDOS = 4


def pausa() -> None:
    """Espera entre casos para no superar el limite de peticiones de Gemini."""
    time.sleep(PAUSA_SEGUNDOS)


def ejecutar(app, titulo: str, pregunta: str, thread_id: str, vistas: int = 0):
    """Ejecuta un turno e imprime pregunta, respuesta y decisiones nuevas.

    Devuelve el estado y cuantas decisiones se han visto, para no repetir las
    del turno anterior cuando el caso usa el mismo hilo.
    """
    print("=" * 72)
    print(titulo)
    print(f"Pregunta: {pregunta}")
    estado = app.invoke(
        {
            "messages": [HumanMessage(content=pregunta)],
            "user_id": USUARIO_DEMO,
        },
        {"configurable": {"thread_id": thread_id}},
    )
    print(f"Respuesta: {_texto_respuesta(estado['messages'][-1])}")
    print("Decisiones:")
    for decision in estado["decisiones"][vistas:]:
        lineas = decision.splitlines()
        print(f"  - {lineas[0]}")
        for linea in lineas[1:]:
            print(f"    {linea}")
    print()
    return estado, len(estado["decisiones"])


def verificar_dispensacion() -> None:
    """Comprueba que la R-1001 quedo dispensada y el stock se desconto."""
    conn = db.conectar()
    try:
        estado = conn.execute(
            "SELECT estado FROM recetas WHERE id = 'R-1001'"
        ).fetchone()["estado"]
        total = conn.execute(
            "SELECT COUNT(*) FROM dispensaciones WHERE receta_id = 'R-1001'"
        ).fetchone()[0]
        stock = conn.execute(
            "SELECT unidades FROM inventario "
            "WHERE principio_activo = 'Z' AND sucursal = 'Centro'"
        ).fetchone()["unidades"]
    finally:
        conn.close()

    print("=" * 72)
    print("VERIFICACION FINAL")
    print(
        f"R-1001: {estado} | dispensaciones: {total} | "
        f"stock Z Centro: {stock}"
    )


def main() -> None:
    # El modelo configurado usa sampling fijo; se evita el aviso repetido.
    warnings.filterwarnings("ignore", message=".*fixed sampling defaults.*")

    print("Preparando datos ficticios y RAG...")
    cargar_datos.cargar_datos()
    print(f"Fragmentos indexados: {rag.indexar()}")

    # La memoria de largo plazo se conserva entre ejecuciones; para la demo se
    # limpia solo el usuario de demo y se parte de un estado conocido.
    memoria.borrar_usuario(USUARIO_DEMO)
    print()

    app = crear_grafo()

    # Caso 1: consulta simple de stock.
    ejecutar(
        app,
        "CASO 1: consulta simple de stock",
        "¿Hay stock de Z en Centro?",
        "demo-caso-1",
    )
    pausa()

    # Caso 2: interaccion X-Y en la R-1045.
    ejecutar(
        app,
        "CASO 2: interacciones de la R-1045",
        "Revisa la receta R-1045 y dime si hay interacciones.",
        "demo-caso-2",
    )
    pausa()

    # Caso 3: V sin stock en Centro, con alternativa.
    ejecutar(
        app,
        "CASO 3: R-1050 sin stock de V en Centro, con alternativa",
        "La R-1050 no tiene stock de V en Centro. ¿Qué alternativa hay?",
        "demo-caso-3",
    )
    pausa()

    # Caso 4: Inventario caido; debe replanificar y no caerse.
    config.INVENTARIO_CAIDO = True
    try:
        ejecutar(
            app,
            "CASO 4: Inventario caido, con replanificacion",
            "¿Puedo dispensar la R-1050?",
            "demo-caso-4",
        )
    finally:
        config.INVENTARIO_CAIDO = False
    pausa()

    # Caso 5: dispensacion en dos turnos del mismo hilo.
    estado, vistas = ejecutar(
        app,
        "CASO 5a: R-1001, primero pide confirmacion",
        "Quiero dispensar la R-1001. Farmaceutico: farm-01.",
        "demo-caso-5",
    )
    pausa()
    ejecutar(
        app,
        "CASO 5b: R-1001, confirma y registra",
        "Si, confirmo la dispensacion de la R-1001 por farm-01.",
        "demo-caso-5",
        vistas,
    )

    verificar_dispensacion()


if __name__ == "__main__":
    # La consola de Windows no siempre usa UTF-8; esto evita acentos rotos.
    sys.stdout.reconfigure(encoding="utf-8")
    main()
