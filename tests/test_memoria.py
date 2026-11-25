"""Pruebas de la memoria de largo plazo que no llaman a embeddings.

Solo verifican el id determinista y el filtro de datos de pacientes; el flujo
completo con Chroma se prueba con `python src/grafo.py`.
"""

from src import memoria


def test_id_hecho_es_estable():
    assert memoria._id_hecho("farm-01", "Trabajo en la sucursal Centro") == (
        memoria._id_hecho("farm-01", "Trabajo en la sucursal Centro")
    )


def test_id_hecho_distingue_usuarios():
    assert memoria._id_hecho("farm-01", "Trabajo en Centro") != (
        memoria._id_hecho("farm-02", "Trabajo en Centro")
    )


def test_detecta_datos_de_paciente():
    assert memoria._es_dato_paciente("RUT 11.111.111-1")
    assert memoria._es_dato_paciente("correo ana@example.cl")
    assert memoria._es_dato_paciente("El paciente P-001 retiró su receta")


def test_no_marca_hechos_operativos():
    assert not memoria._es_dato_paciente("Trabajo en la sucursal Centro")
    assert not memoria._es_dato_paciente("Prefiero los turnos de mañana")


def test_rechaza_consultas_puntuales_como_hechos_estables():
    assert not memoria._es_hecho_estable("La receta R-1045 tiene interacción X-Y")
    assert not memoria._es_hecho_estable("Hay 15 unidades de Z en Centro")
    assert memoria._es_hecho_estable("Trabajo en la sucursal Centro")
