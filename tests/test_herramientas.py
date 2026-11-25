"""Pruebas de las herramientas (Fase 3), sin llamar al LLM ni a embeddings.

Cada prueba usa una base SQLite temporal cargada con los datos ficticios, para
no modificar data/recetaya.db. La busqueda RAG se sustituye por un doble para
que las pruebas sean rapidas y no dependan de la red.
"""

import pytest

from src import cargar_datos, config, db, herramientas


@pytest.fixture
def base_ficticia(tmp_path, monkeypatch):
    """Carga los datos de demo en una base temporal y la deja lista."""
    monkeypatch.setattr(db, "RUTA_DB", tmp_path / "recetaya_test.db")
    cargar_datos.cargar_datos()
    yield


def _estado_receta(receta_id: str) -> str:
    conn = db.conectar()
    try:
        fila = conn.execute(
            "SELECT estado FROM recetas WHERE id = ?", (receta_id,)
        ).fetchone()
        return fila["estado"]
    finally:
        conn.close()


def test_receta_inexistente(base_ficticia):
    respuesta = herramientas.consultar_receta.invoke({"receta_id": "R-9999"})
    assert "No existe la receta R-9999" in respuesta


def test_interaccion_encontrada(base_ficticia, monkeypatch):
    # Doble de rag.buscar: devuelve una interaccion y una seccion que debe
    # descartarse (Dosis), para verificar el filtro del enunciado.
    def buscar_falso(consulta, k=4, principio_activo=None):
        return [
            {
                "texto": "Interacciones\nNo combinar X con Y: riesgo de sangrado.",
                "fuente": "ficha_x.md",
                "seccion": "Interacciones",
                "similitud": 0.9,
            },
            {
                "texto": "Dosis\n50 mg cada 8 h.",
                "fuente": "ficha_x.md",
                "seccion": "Dosis",
                "similitud": 0.8,
            },
        ]

    monkeypatch.setattr(herramientas.rag, "buscar", buscar_falso)
    respuesta = herramientas.verificar_interacciones.invoke(
        {"principios_activos": ["X", "Y"]}
    )

    assert "sangrado" in respuesta
    assert "ficha_x.md / Interacciones" in respuesta
    assert "50 mg" not in respuesta  # la seccion Dosis fue filtrada
    assert "no concluye" in respuesta.lower()


def test_falta_confirmacion_no_escribe(base_ficticia):
    respuesta = herramientas.registrar_dispensacion.invoke(
        {"receta_id": "R-1001", "farmaceutico": "farm-01"}
    )

    assert "Confirmacion requerida" in respuesta
    assert "No se registro nada" in respuesta
    assert _estado_receta("R-1001") == "reservada"

    conn = db.conectar()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM dispensaciones WHERE receta_id = 'R-1001'"
        ).fetchone()[0]
    finally:
        conn.close()
    assert total == 0


def test_dispensacion_correcta(base_ficticia):
    respuesta = herramientas.registrar_dispensacion.invoke(
        {"receta_id": "R-1001", "farmaceutico": "farm-01", "confirmado": True}
    )

    assert "Dispensacion registrada" in respuesta
    assert _estado_receta("R-1001") == "dispensada"

    conn = db.conectar()
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM dispensaciones WHERE receta_id = 'R-1001'"
        ).fetchone()[0]
        stock = conn.execute(
            "SELECT unidades FROM inventario "
            "WHERE principio_activo = 'Z' AND sucursal = 'Centro'"
        ).fetchone()["unidades"]
    finally:
        conn.close()

    assert total == 1
    assert stock == 14  # 15 iniciales - 1 dispensada


def test_receta_no_reservada(base_ficticia):
    # R-1050 esta emitida: la reserva aun no se completo.
    respuesta = herramientas.registrar_dispensacion.invoke(
        {"receta_id": "R-1050", "farmaceutico": "farm-01", "confirmado": True}
    )

    assert "solo se dispensan recetas reservadas" in respuesta
    assert _estado_receta("R-1050") == "emitida"


def test_inventario_caido(base_ficticia, monkeypatch):
    monkeypatch.setattr(config, "INVENTARIO_CAIDO", True)

    with pytest.raises(RuntimeError, match="Servicio de Inventario no disponible"):
        herramientas.consultar_stock.invoke(
            {"principio_activo": "Z", "sucursal": "Centro"}
        )


def test_consultar_receta_no_expone_datos_personales(base_ficticia):
    respuesta = herramientas.consultar_receta.invoke({"receta_id": "R-1045"})

    assert "P-002" in respuesta
    assert "Bruno" not in respuesta
    assert "22.222.222-2" not in respuesta


def test_alternativa_bioequivalente(base_ficticia):
    # V no tiene stock en Centro, pero W (bioequivalente) si; ademas V
    # tiene 12 unidades en Norte.
    respuesta = herramientas.buscar_alternativa.invoke(
        {"principio_activo": "V", "sucursal": "Centro"}
    )

    assert "W (bioequivalente de V) en Centro: 20 unidades" in respuesta
    assert "V en Norte: 12 unidades" in respuesta
    assert "el farmaceutico" in respuesta.lower()
