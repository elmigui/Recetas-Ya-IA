"""Memoria de largo plazo del agente RecetaYa (Fase 6).

Guarda hechos operativos estables del farmaceutico (sucursal, preferencias)
en una coleccion Chroma "memoria", separada del RAG de fichas. Cada hecho se
filtra por user_id y el id se deriva del texto para que guardar dos veces el
mismo hecho no cree duplicados.
"""

import hashlib
import re

import chromadb

from src import config

NOMBRE_COLECCION = "memoria"

# Nunca se guardan datos de pacientes en la memoria de largo plazo.
PATRONES_PACIENTE = (
    re.compile(r"\b\d{1,2}\.\d{3}\.\d{3}-[\dkK]\b"),  # RUT con puntos
    re.compile(r"\b\d{7,8}-[\dkK]\b"),  # RUT sin puntos
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),  # correo
    re.compile(r"\bpaciente\b", re.IGNORECASE),  # referencias a pacientes
    re.compile(r"\bP-\d{3}\b"),  # ids de paciente como P-001
)

# Consultas puntuales que no deben convertirse en memoria estable.
PATRONES_NO_ESTABLES = (
    re.compile(r"\bR-\d{4}\b"),  # receta concreta
    re.compile(r"\bstock\b", re.IGNORECASE),
    re.compile(r"\bunidades\b", re.IGNORECASE),
    re.compile(r"\breceta\b", re.IGNORECASE),
)


def _coleccion():
    """Abre (o crea) la coleccion de memoria con similitud coseno."""
    config.DIR_CHROMA.mkdir(parents=True, exist_ok=True)
    cliente = chromadb.PersistentClient(path=str(config.DIR_CHROMA))
    return cliente.get_or_create_collection(
        name=NOMBRE_COLECCION,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def _id_hecho(user_id: str, hecho: str) -> str:
    """Id determinista: el mismo usuario y texto siempre producen el mismo id."""
    return hashlib.sha256(f"{user_id}|{hecho}".encode("utf-8")).hexdigest()


def _es_dato_paciente(hecho: str) -> bool:
    """Evita guardar RUT, correos, ids o referencias a pacientes."""
    return any(patron.search(hecho) for patron in PATRONES_PACIENTE)


def _es_hecho_estable(hecho: str) -> bool:
    """Rechaza consultas puntuales (recetas, stock, unidades)."""
    return not any(patron.search(hecho) for patron in PATRONES_NO_ESTABLES)


def recuperar(user_id: str, consulta: str, k: int = 3) -> list[str]:
    """Devuelve los k hechos mas parecidos a la consulta para ese usuario."""
    if not user_id or not consulta.strip():
        return []

    coleccion = _coleccion()
    if coleccion.count() == 0:
        return []

    vector = config.crear_embeddings().embed_query(consulta)
    resultado = coleccion.query(
        query_embeddings=[vector],
        n_results=k,
        where={"user_id": user_id},
        include=["documents"],
    )
    documentos = resultado.get("documents") or [[]]
    if not documentos or not documentos[0]:
        return []
    return [texto for texto in documentos[0] if texto]


def guardar(user_id: str, hechos: list[str]) -> int:
    """Guarda hechos nuevos y devuelve cuantos se procesaron.

    Se usa upsert con id derivado del texto: repetir el mismo hecho actualiza
    el documento en vez de duplicarlo.
    """
    if not user_id:
        return 0

    limpios = []
    vistos = set()
    for hecho in hechos:
        texto = hecho.strip()
        if not texto or _es_dato_paciente(texto) or not _es_hecho_estable(texto):
            continue
        identificador = _id_hecho(user_id, texto)
        if identificador in vistos:
            continue
        vistos.add(identificador)
        limpios.append((identificador, texto))

    if not limpios:
        return 0

    embeddings = config.crear_embeddings().embed_documents(
        [texto for _, texto in limpios]
    )
    coleccion = _coleccion()
    coleccion.upsert(
        ids=[identificador for identificador, _ in limpios],
        documents=[texto for _, texto in limpios],
        metadatas=[{"user_id": user_id} for _ in limpios],
        embeddings=embeddings,
    )
    return len(limpios)


def borrar_usuario(user_id: str) -> None:
    """Borra los hechos de un usuario (util para una demo repetible)."""
    if not user_id:
        return
    _coleccion().delete(where={"user_id": user_id})
