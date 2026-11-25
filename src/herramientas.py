"""Herramientas del agente RecetaYa (Fase 3).

Cada funcion decorada con @tool es una herramienta que el LLM puede decidir
usar. El docstring es la descripcion que lee el modelo, por eso explica que
devuelve y que NO hace (el agente informa, no prescribe).

Reglas de diseno:
- Ninguna herramienta devuelve nombre ni RUT del paciente; solo paciente_id.
- Las consultas validan y normalizan sus entradas.
- Las herramientas de lectura no escriben; la unica de escritura es
  registrar_dispensacion y exige confirmacion explicita.
"""

import json
from datetime import datetime

from langchain_core.tools import tool

from src import config, db, rag

# Estados posibles de una receta y su significado para el farmaceutico.
MENSAJES_RESERVA = {
    "emitida": "la reserva de inventario esta en cola",
    "reservada": "el stock esta reservado y la receta puede dispensarse",
    "dispensada": "la receta ya fue dispensada",
}


def _texto_obligatorio(valor: str, nombre: str) -> str:
    """Valida que una entrada de texto no venga vacia."""
    if not isinstance(valor, str) or not valor.strip():
        raise ValueError(f"{nombre} no puede estar vacio")
    return valor.strip()


def _normalizar_receta(receta_id: str) -> str:
    """R-1001, r-1001 y ' R-1001 ' se tratan como la misma receta."""
    return _texto_obligatorio(receta_id, "receta_id").upper()


def _normalizar_principio(principio_activo: str) -> str:
    """Los principios activos se guardan en mayuscula (X, Y, Z, V, W)."""
    return _texto_obligatorio(principio_activo, "principio_activo").upper()


def _normalizar_sucursal(sucursal: str) -> str:
    """Centro/Norte se guardan con inicial mayuscula."""
    return _texto_obligatorio(sucursal, "sucursal").title()


def _items_de_receta(conn, receta_id: str) -> list:
    """Devuelve los medicamentos de una receta ordenados por principio."""
    return conn.execute(
        "SELECT principio_activo, dosis_mg, cada_horas "
        "FROM receta_items WHERE receta_id = ? ORDER BY principio_activo",
        (receta_id,),
    ).fetchall()


def _formatear_items(items: list) -> str:
    """Convierte los medicamentos en lineas legibles para el LLM."""
    if not items:
        return "- (sin medicamentos)"
    return "\n".join(
        f"- {item['principio_activo']}: {item['dosis_mg']} mg cada "
        f"{item['cada_horas']} h"
        for item in items
    )


def _unidades(conn, principio_activo: str, sucursal: str) -> int:
    """Unidades en stock; 0 si no existe la fila de inventario."""
    fila = conn.execute(
        "SELECT unidades FROM inventario "
        "WHERE principio_activo = ? AND sucursal = ?",
        (principio_activo, sucursal),
    ).fetchone()
    return fila["unidades"] if fila else 0


def _api_sanitaria() -> dict:
    """Lee el JSON que simula la API sanitaria externa."""
    return json.loads(config.RUTA_API_SANITARIA.read_text(encoding="utf-8"))


@tool
def consultar_receta(receta_id: str) -> str:
    """Consulta una receta por su ID (por ejemplo R-1001).

    Devuelve estado, sucursal, indicaciones y medicamentos con dosis.
    No entrega nombre ni RUT del paciente, solo su paciente_id.
    """
    receta_id = _normalizar_receta(receta_id)
    conn = db.conectar()
    try:
        receta = conn.execute(
            "SELECT id, paciente_id, sucursal, estado, indicaciones "
            "FROM recetas WHERE id = ?",
            (receta_id,),
        ).fetchone()
        if receta is None:
            return f"No existe la receta {receta_id}."

        items = _items_de_receta(conn, receta_id)
        return (
            f"Receta {receta['id']} | paciente {receta['paciente_id']} | "
            f"sucursal {receta['sucursal']} | estado {receta['estado']}\n"
            f"Indicaciones: {receta['indicaciones'] or 'sin indicaciones'}\n"
            f"Medicamentos:\n{_formatear_items(items)}"
        )
    finally:
        conn.close()


@tool
def consultar_stock(principio_activo: str, sucursal: str) -> str:
    """Consulta las unidades disponibles de un medicamento en una sucursal.

    Si el servicio de Inventario esta simulado como caido (INVENTARIO_CAIDO=1),
    lanza la excepcion "Servicio de Inventario no disponible" para que el
    agente pueda replanificar.
    """
    principio = _normalizar_principio(principio_activo)
    sucursal = _normalizar_sucursal(sucursal)

    if config.INVENTARIO_CAIDO:
        raise RuntimeError("Servicio de Inventario no disponible")

    conn = db.conectar()
    try:
        fila = conn.execute(
            "SELECT unidades FROM inventario "
            "WHERE principio_activo = ? AND sucursal = ?",
            (principio, sucursal),
        ).fetchone()
        if fila is None:
            return f"No hay registro de stock de {principio} en {sucursal}."
        if fila["unidades"] == 0:
            return f"Sin stock: 0 unidades de {principio} en {sucursal}."
        return f"Stock de {principio} en {sucursal}: {fila['unidades']} unidades."
    finally:
        conn.close()


@tool
def estado_reserva(receta_id: str) -> str:
    """Indica el estado de reserva de una receta.

    Distingue entre emitida (reserva en cola), reservada (stock apartado y
    lista para dispensar) y dispensada.
    """
    receta_id = _normalizar_receta(receta_id)
    conn = db.conectar()
    try:
        fila = conn.execute(
            "SELECT estado FROM recetas WHERE id = ?", (receta_id,)
        ).fetchone()
        if fila is None:
            return f"No existe la receta {receta_id}."

        estado = fila["estado"]
        detalle = MENSAJES_RESERVA.get(estado, "estado desconocido")
        return f"Receta {receta_id}: estado {estado} ({detalle})."
    finally:
        conn.close()


@tool
def verificar_interacciones(principios_activos: list[str]) -> str:
    """Busca evidencia de interacciones y contraindicaciones en las fichas.

    Recibe una lista de principios activos (por ejemplo ["X", "Y"]) y devuelve
    los fragmentos encontrados con su fuente. No concluye ni recomienda nada:
    solo entrega evidencia citada para que el farmaceutico decida.
    """
    if not principios_activos:
        raise ValueError("principios_activos no puede estar vacio")

    # Se normalizan y deduplican para no repetir la misma busqueda.
    principios = []
    for principio in principios_activos:
        normalizado = _normalizar_principio(principio)
        if normalizado not in principios:
            principios.append(normalizado)

    bloques = []
    for principio in principios:
        # k=6 porque cada ficha solo tiene 3 secciones; luego se filtra a las
        # secciones clinicas que pide el enunciado.
        resultados = rag.buscar(
            f"interacciones y contraindicaciones de {principio}",
            k=6,
            principio_activo=principio,
        )
        relevantes = [
            resultado
            for resultado in resultados
            if resultado["seccion"] in {"Interacciones", "Contraindicaciones"}
        ]
        if not relevantes:
            bloques.append(
                f"{principio}: no se encontro evidencia en las fichas indexadas."
            )
            continue
        lineas = [f"Evidencia para {principio}:"]
        for resultado in relevantes:
            lineas.append(
                f"- [{resultado['fuente']} / {resultado['seccion']}] "
                f"{resultado['texto']}"
            )
        bloques.append("\n".join(lineas))

    bloques.append("El agente no concluye: el farmaceutico evalua la evidencia.")
    return "\n\n".join(bloques)


@tool
def buscar_alternativa(principio_activo: str, sucursal: str) -> str:
    """Busca bioequivalentes y stock disponible en otras sucursales.

    Usa el JSON de la API sanitaria para los bioequivalentes y SQLite para el
    stock. Solo informa opciones; la sustitucion la decide el farmaceutico.
    """
    principio = _normalizar_principio(principio_activo)
    sucursal = _normalizar_sucursal(sucursal)

    api = _api_sanitaria()
    bioequivalentes = api.get("bioequivalentes", {})

    # El JSON guarda la relacion en un solo sentido (V -> W). Se revisa
    # tambien el sentido inverso para que W encuentre a V.
    alternativas = list(bioequivalentes.get(principio, []))
    for original, lista in bioequivalentes.items():
        if principio in lista and original not in alternativas:
            alternativas.append(original)

    alertas = [
        alerta["mensaje"]
        for alerta in api.get("alertas", [])
        if alerta.get("principio_activo") == principio
    ]

    conn = db.conectar()
    try:
        lineas = []
        # 1) Bioequivalentes con stock en la misma sucursal.
        for alternativa in alternativas:
            unidades = _unidades(conn, alternativa, sucursal)
            if unidades > 0:
                lineas.append(
                    f"- {alternativa} (bioequivalente de {principio}) en "
                    f"{sucursal}: {unidades} unidades."
                )

        # 2) El mismo medicamento disponible en otras sucursales.
        filas = conn.execute(
            "SELECT sucursal, unidades FROM inventario "
            "WHERE principio_activo = ? AND sucursal <> ? AND unidades > 0 "
            "ORDER BY sucursal",
            (principio, sucursal),
        ).fetchall()
        for fila in filas:
            lineas.append(
                f"- {principio} en {fila['sucursal']}: {fila['unidades']} unidades."
            )
    finally:
        conn.close()

    if not lineas:
        respuesta = (
            f"No se encontraron alternativas con stock para {principio} "
            f"en {sucursal}."
        )
    else:
        respuesta = (
            f"Alternativas para {principio} en {sucursal}:\n"
            + "\n".join(lineas)
            + "\nLa sustitucion la decide el farmaceutico."
        )

    if alertas:
        respuesta += "\nAlerta sanitaria: " + " ".join(alertas)
    return respuesta


@tool
def registrar_dispensacion(
    receta_id: str,
    farmaceutico: str,
    confirmado: bool = False,
) -> str:
    """Registra la dispensacion de una receta, solo con confirmacion humana.

    Con confirmado=False no escribe nada: devuelve un resumen para que el
    farmaceutico confirme. Con confirmado=True valida que la receta exista y
    este reservada, descuenta 1 unidad por medicamento, guarda la dispensacion
    y marca la receta como dispensada, todo en una sola transaccion.

    Se descuenta 1 unidad por medicamento porque el modelo de datos no tiene
    un campo de cantidad dispensada; es la regla simple definida para la demo.
    """
    receta_id = _normalizar_receta(receta_id)
    farmaceutico = _texto_obligatorio(farmaceutico, "farmaceutico")

    conn = db.conectar()
    try:
        receta = conn.execute(
            "SELECT id, paciente_id, sucursal, estado FROM recetas WHERE id = ?",
            (receta_id,),
        ).fetchone()
        if receta is None:
            return f"No existe la receta {receta_id}."

        items = _items_de_receta(conn, receta_id)
        resumen = (
            f"Receta {receta_id} | paciente {receta['paciente_id']} | "
            f"sucursal {receta['sucursal']} | estado {receta['estado']}\n"
            f"Medicamentos:\n{_formatear_items(items)}\n"
            f"Farmaceutico: {farmaceutico}"
        )

        # Sin confirmacion no se escribe nada; solo se pide confirmar.
        if not confirmado:
            return (
                "Confirmacion requerida. No se registro nada.\n"
                f"{resumen}\n"
                "Para registrar, vuelve a llamar con confirmado=True."
            )

        if receta["estado"] != "reservada":
            return (
                f"No se puede dispensar {receta_id}: su estado es "
                f"'{receta['estado']}' y solo se dispensan recetas reservadas."
            )

        # Se valida todo el stock antes de escribir para no dejar una
        # dispensacion a medias si falta algun medicamento.
        for item in items:
            if _unidades(conn, item["principio_activo"], receta["sucursal"]) < 1:
                return (
                    f"No hay stock suficiente de {item['principio_activo']} "
                    f"en {receta['sucursal']} para dispensar {receta_id}."
                )

        fecha = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        conn.execute(
            "INSERT INTO dispensaciones (receta_id, farmaceutico, fecha) "
            "VALUES (?, ?, ?)",
            (receta_id, farmaceutico, fecha),
        )
        for item in items:
            conn.execute(
                "UPDATE inventario SET unidades = unidades - 1 "
                "WHERE principio_activo = ? AND sucursal = ?",
                (item["principio_activo"], receta["sucursal"]),
            )
        conn.execute(
            "UPDATE recetas SET estado = 'dispensada' WHERE id = ?", (receta_id,)
        )
        conn.commit()
        return (
            f"Dispensacion registrada: {receta_id} | farmaceutico "
            f"{farmaceutico} | fecha {fecha}\n"
            f"{_formatear_items(items)}"
        )
    except Exception:
        # Ante cualquier error se revierte la transaccion completa.
        conn.rollback()
        raise
    finally:
        conn.close()


# Lista que usara ToolNode en la Fase 4.
HERRAMIENTAS = [
    consultar_receta,
    consultar_stock,
    estado_reserva,
    verificar_interacciones,
    buscar_alternativa,
    registrar_dispensacion,
]
