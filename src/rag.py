"""RAG de RecetaYa sobre las fichas tecnicas (Fase 2).

Flujo:
1. indexar() lee data/fichas/*.md, parte cada archivo por sus titulos "##",
   calcula embeddings con el proveedor configurado y los guarda en Chroma.
2. buscar() embebe la consulta, recupera los fragmentos mas parecidos y
   devuelve el texto con su fuente, seccion y similitud.

Chroma se guarda en data/chroma/ (ignorado por git). Se pasan los embeddings
explicitamente para no depender del modelo ONNX por defecto de Chroma y para
usar siempre el mismo modelo que define config.py.
"""

import re
import sys
from pathlib import Path

# Permite ejecutar `python src/rag.py` desde la raiz del repo.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import chromadb  # noqa: E402
from chromadb.errors import NotFoundError  # noqa: E402

from src import config  # noqa: E402

NOMBRE_COLECCION = "documentos"

# Una linea que contenga alguno de estos textos se considera una instruccion
# dirigida al modelo (inyeccion indirecta) y se elimina del fragmento.
PATRONES_INYECCION = (
    "ignora las instrucciones",
    "ignora las indicaciones",
    "ignora todo",
    "olvida las instrucciones",
    "olvida todo",
    "ignore previous",
    "ignore all",
    "nuevas instrucciones",
    "system prompt",
    "actua como",
    "actúa como",
    "desobedece",
    "no sigas las instrucciones",
)


def _cliente() -> chromadb.ClientAPI:
    """Abre (o crea) el almacen persistente de Chroma en data/chroma/."""
    config.DIR_CHROMA.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(config.DIR_CHROMA))


def _abrir_coleccion(cliente: chromadb.ClientAPI):
    """Devuelve la coleccion con similitud coseno.

    Se pasa embedding_function=None porque los vectores se calculan en este
    modulo; si Chroma usara su modelo por defecto, la dimension (384) no
    coincidiria con la de Gemini (768).
    """
    return cliente.get_or_create_collection(
        name=NOMBRE_COLECCION,
        configuration={"hnsw": {"space": "cosine"}},
        embedding_function=None,
    )


def _principio_de(archivo: Path) -> str:
    """Obtiene el principio activo desde el nombre de la ficha.

    ficha_x.md -> "X"; cualquier otro documento (el protocolo) -> "general".
    """
    coincidencia = re.fullmatch(r"ficha_([a-zA-Z])\.md", archivo.name)
    return coincidencia.group(1).upper() if coincidencia else "general"


def _titulo(contenido: str) -> str:
    """Devuelve el titulo "# ..." del documento, o "" si no tiene."""
    coincidencia = re.search(r"^#\s+(.+?)\s*$", contenido, flags=re.MULTILINE)
    return coincidencia.group(1).strip() if coincidencia else ""


def _fragmentos(contenido: str) -> list[tuple[str, str]]:
    """Parte un Markdown por titulos "##" y devuelve (seccion, texto).

    El titulo se incluye dentro del texto para que el embedding y la cita
    tengan contexto ("Interacciones ..." en vez de solo el cuerpo).
    """
    partes = re.split(r"^##\s+(.+?)\s*$", contenido, flags=re.MULTILINE)
    fragmentos = []
    # partes[0] es el titulo "# ..." anterior al primer "##"; se descarta.
    for i in range(1, len(partes), 2):
        seccion = partes[i].strip()
        cuerpo = partes[i + 1].strip()
        if seccion and cuerpo:
            fragmentos.append((seccion, f"{seccion}\n{cuerpo}"))
    return fragmentos


def _limpiar_inyeccion(texto: str) -> str:
    """Quita lineas con instrucciones dirigidas al modelo.

    Se filtra linea por linea para conservar el contenido clinico del resto
    del fragmento; no se intenta reescribir el texto.
    """
    lineas_limpias = []
    for linea in texto.splitlines():
        if any(patron in linea.lower() for patron in PATRONES_INYECCION):
            continue
        lineas_limpias.append(linea)
    return "\n".join(lineas_limpias).strip()


def indexar() -> int:
    """Reindexa las fichas desde cero y devuelve cuantos fragmentos guardo.

    Se elimina la coleccion anterior para que el indice refleje siempre el
    contenido actual de data/fichas/ (no quedan fragmentos viejos).
    """
    archivos = sorted(config.DIR_FICHAS.glob("*.md"))
    if not archivos:
        raise FileNotFoundError(f"No se encontraron fichas en {config.DIR_FICHAS}")

    cliente = _cliente()
    try:
        cliente.delete_collection(NOMBRE_COLECCION)
    except NotFoundError:
        # La primera vez la coleccion no existe; no es un error.
        pass

    coleccion = _abrir_coleccion(cliente)
    ids: list[str] = []
    textos: list[str] = []
    metadatos: list[dict] = []

    for archivo in archivos:
        contenido = archivo.read_text(encoding="utf-8")
        principio = _principio_de(archivo)
        # El titulo "# ..." se agrega a cada fragmento: sin el, secciones
        # iguales de fichas distintas (todas tienen "Interacciones") quedan
        # casi empatadas y el medicamento consultado no se distingue.
        titulo = _titulo(contenido)
        for seccion, texto in _fragmentos(contenido):
            ids.append(f"{archivo.stem}::{seccion}")
            textos.append(f"{titulo}\n\n{texto}" if titulo else texto)
            metadatos.append(
                {
                    "fuente": archivo.name,
                    "seccion": seccion,
                    "principio_activo": principio,
                }
            )

    if not textos:
        raise ValueError("Las fichas no tienen secciones '##' para indexar")

    # Los embeddings se calculan en lote y se guardan junto a los documentos.
    embeddings = config.crear_embeddings().embed_documents(textos)
    coleccion.add(
        ids=ids,
        documents=textos,
        metadatas=metadatos,
        embeddings=embeddings,
    )
    return len(textos)


def buscar(
    consulta: str,
    k: int = 4,
    principio_activo: str | None = None,
) -> list[dict]:
    """Busca fragmentos similares y devuelve texto, fuente, seccion y similitud.

    principio_activo filtra por medicamento (ej. "X"); con None busca en todas
    las fichas. La similitud se calcula como 1 - distancia coseno.
    """
    if not consulta or not consulta.strip():
        raise ValueError("La consulta no puede estar vacia")
    if k <= 0:
        raise ValueError("k debe ser mayor que 0")

    cliente = _cliente()
    coleccion = _abrir_coleccion(cliente)
    if coleccion.count() == 0:
        return []

    vector = config.crear_embeddings().embed_query(consulta)
    filtro = None
    if principio_activo:
        filtro = {"principio_activo": principio_activo.strip().upper()}

    resultado = coleccion.query(
        query_embeddings=[vector],
        n_results=k,
        where=filtro,
        include=["documents", "metadatas", "distances"],
    )

    ids = resultado["ids"][0]
    if not ids:
        return []

    hallazgos = []
    for texto, metadato, distancia in zip(
        resultado["documents"][0],
        resultado["metadatas"][0],
        resultado["distances"][0],
    ):
        # La limpieza se aplica al recuperar (no al indexar) segun el enunciado.
        texto_limpio = _limpiar_inyeccion(texto or "")
        if not texto_limpio:
            continue
        hallazgos.append(
            {
                "texto": texto_limpio,
                "fuente": metadato["fuente"],
                "seccion": metadato["seccion"],
                "similitud": round(1 - float(distancia), 4),
            }
        )
    return hallazgos


if __name__ == "__main__":
    total = indexar()
    print(f"Fragmentos indexados: {total} en {config.DIR_CHROMA}")
    print("\nBusqueda de prueba: 'interacciones de X'")
    for hallazgo in buscar("interacciones de X", k=4):
        print(
            f"- {hallazgo['fuente']} / {hallazgo['seccion']} "
            f"(similitud {hallazgo['similitud']})"
        )
