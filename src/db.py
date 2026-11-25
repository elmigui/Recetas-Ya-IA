"""Conexion y esquema de la base SQLite que simula los microservicios de RecetaYa.

SQLite se usa porque el enunciado pide simular los microservicios en local sin
levantar servidores. Un solo archivo (data/recetaya.db) contiene las tablas de
Pacientes, Recetas, Inventario y Dispensacion.
"""

import sqlite3
from pathlib import Path

# Ruta absoluta para que la base se cree siempre en data/ sin importar desde
# donde se ejecute el script (raiz del repo, tests, etc.).
RAIZ = Path(__file__).resolve().parent.parent
RUTA_DB = RAIZ / "data" / "recetaya.db"

# El esquema declara las restricciones en la propia base (CHECK y FOREIGN KEY)
# para que los datos ficticios no queden inconsistentes aunque se inserten a mano.
ESQUEMA = """
CREATE TABLE pacientes (
    id TEXT PRIMARY KEY,
    nombre TEXT NOT NULL,
    rut TEXT NOT NULL UNIQUE
);

CREATE TABLE recetas (
    id TEXT PRIMARY KEY,
    paciente_id TEXT NOT NULL,
    sucursal TEXT NOT NULL,
    estado TEXT NOT NULL CHECK (estado IN ('emitida', 'reservada', 'dispensada')),
    indicaciones TEXT,
    FOREIGN KEY (paciente_id) REFERENCES pacientes(id)
);

CREATE TABLE receta_items (
    receta_id TEXT NOT NULL,
    principio_activo TEXT NOT NULL,
    dosis_mg INTEGER NOT NULL,
    cada_horas INTEGER NOT NULL,
    PRIMARY KEY (receta_id, principio_activo),
    FOREIGN KEY (receta_id) REFERENCES recetas(id) ON DELETE CASCADE
);

CREATE TABLE inventario (
    principio_activo TEXT NOT NULL,
    sucursal TEXT NOT NULL,
    unidades INTEGER NOT NULL CHECK (unidades >= 0),
    PRIMARY KEY (principio_activo, sucursal)
);

CREATE TABLE dispensaciones (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    receta_id TEXT NOT NULL,
    farmaceutico TEXT NOT NULL,
    fecha TEXT NOT NULL,
    FOREIGN KEY (receta_id) REFERENCES recetas(id)
);
"""


def conectar(ruta: Path = RUTA_DB) -> sqlite3.Connection:
    """Abre una conexion a la base y activa las claves foraneas.

    sqlite3 no aplica las FOREIGN KEY por defecto; se activan en cada conexion
    para que la integridad se respete en todos los modulos (herramientas, demo).
    """
    ruta.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(ruta)
    # Row permite acceder a las columnas por nombre (fila["estado"]), lo que
    # hace el codigo de las herramientas mucho mas legible que usar indices.
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def crear_esquema(conn: sqlite3.Connection) -> None:
    """Crea todas las tablas si no existen."""
    conn.executescript(ESQUEMA)
    conn.commit()


def borrar_bd(ruta: Path = RUTA_DB) -> None:
    """Elimina el archivo de la base para recargar los datos desde cero.

    Se usa en cargar_datos.py y permite repetir la carga de forma idempotente.
    """
    ruta.unlink(missing_ok=True)
