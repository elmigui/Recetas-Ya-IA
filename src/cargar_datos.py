"""Borra y recarga los datos ficticios de RecetaYa (Fase 1).

Este script es el unico punto de entrada de la Fase 1: crea data/recetaya.db y
deja las recetas, el inventario y una dispensacion de ejemplo listos para que
las herramientas de la Fase 3 consulten sobre ellos.
"""

import sys
from pathlib import Path

# Al ejecutar `python src/cargar_datos.py`, Python agrega solo src/ a sys.path.
# Agregamos la raiz del repo para poder importar el paquete src.db.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import db  # noqa: E402  (el import va despues del ajuste de sys.path)

# Datos ficticios. Los RUT son inventados y solo se usan para probar el
# enmascaramiento de la Fase 5; nunca corresponden a personas reales.
PACIENTES = [
    ("P-001", "Ana Pérez", "11.111.111-1"),
    ("P-002", "Bruno Soto", "22.222.222-2"),
    ("P-003", "Carla Díaz", "33.333.333-3"),
]

RECETAS = [
    ("R-1001", "P-001", "Centro", "reservada", "Tomar 1 comprimido cada 24 horas."),
    ("R-1045", "P-002", "Centro", "reservada", "Revisar interacción X-Y antes de dispensar."),
    ("R-1050", "P-003", "Centro", "emitida", "Reserva en cola; V sin stock en Centro."),
    ("R-0990", "P-001", "Norte", "dispensada", "Dispensada previamente."),
]

RECETA_ITEMS = [
    ("R-1001", "Z", 20, 24),
    ("R-1045", "X", 50, 8),
    ("R-1045", "Y", 100, 24),
    ("R-1050", "V", 40, 12),
    ("R-0990", "Z", 20, 24),
]

# Inventario exacto indicado en el enunciado. V no tiene stock en Centro y W
# (su bioequivalente) si tiene, lo que permite la demo de alternativa.
INVENTARIO = [
    ("X", "Centro", 10),
    ("Y", "Centro", 30),
    ("Z", "Centro", 15),
    ("Z", "Norte", 8),
    ("V", "Centro", 0),
    ("V", "Norte", 12),
    ("W", "Centro", 20),
]

DISPENSACIONES = [
    ("R-0990", "farm-01", "2026-11-20 10:00:00"),
]


def cargar_datos() -> None:
    """Elimina la base anterior y vuelve a insertar todos los datos ficticios."""
    db.borrar_bd()
    conn = db.conectar()
    try:
        db.crear_esquema(conn)
        conn.executemany(
            "INSERT INTO pacientes (id, nombre, rut) VALUES (?, ?, ?)", PACIENTES
        )
        conn.executemany(
            "INSERT INTO recetas (id, paciente_id, sucursal, estado, indicaciones) "
            "VALUES (?, ?, ?, ?, ?)",
            RECETAS,
        )
        conn.executemany(
            "INSERT INTO receta_items (receta_id, principio_activo, dosis_mg, cada_horas) "
            "VALUES (?, ?, ?, ?)",
            RECETA_ITEMS,
        )
        conn.executemany(
            "INSERT INTO inventario (principio_activo, sucursal, unidades) "
            "VALUES (?, ?, ?)",
            INVENTARIO,
        )
        conn.executemany(
            "INSERT INTO dispensaciones (receta_id, farmaceutico, fecha) "
            "VALUES (?, ?, ?)",
            DISPENSACIONES,
        )
        conn.commit()
        print(f"Base de datos creada en {db.RUTA_DB}")
    except Exception:
        # Si falla una insercion, se revierte todo para no dejar datos a medias.
        conn.rollback()
        raise
    finally:
        conn.close()


def verificar() -> None:
    """Imprime cada receta con sus medicamentos (criterio de aceptacion)."""
    conn = db.conectar()
    try:
        recetas = conn.execute(
            "SELECT id, paciente_id, sucursal, estado FROM recetas ORDER BY id"
        ).fetchall()
        for receta in recetas:
            items = conn.execute(
                "SELECT principio_activo, dosis_mg, cada_horas "
                "FROM receta_items WHERE receta_id = ? ORDER BY principio_activo",
                (receta["id"],),
            ).fetchall()
            medicamentos = ", ".join(
                f"{item['principio_activo']} {item['dosis_mg']} mg cada "
                f"{item['cada_horas']} h"
                for item in items
            )
            print(
                f"{receta['id']} | {receta['paciente_id']} | "
                f"{receta['sucursal']} | {receta['estado']} | {medicamentos}"
            )
    finally:
        conn.close()


if __name__ == "__main__":
    cargar_datos()
    print("\nVerificacion de recetas:")
    verificar()
