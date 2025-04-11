# api_fastapi/utils.py
from pymongo import MongoClient, ASCENDING, DESCENDING # Importar ASCENDING/DESCENDING
from dotenv import load_dotenv
import os
from datetime import datetime, date, timedelta, timezone
from typing import Optional, List, Dict, Any, Literal # Importar Literal
from schemas import TimeGroup


load_dotenv()

# Asegúrate que MONGODB_URL esté definida en tu .env o variables de entorno
MONGODB_URL = os.getenv("MONGODB_URL", "mongodb://localhost:27017/")
DB_NAME = os.getenv("MONGODB_DB_NAME", "logs_db") # Hacer el nombre de la DB configurable es bueno

client = MongoClient(MONGODB_URL)
db = client[DB_NAME]
historico_detecciones = db['historico_detecciones']

# --- Función de agregación principal ---

# --- Helper para construir el $match de fechas (sin cambios) ---
def _build_date_match(from_date_str: Optional[str], to_date_str: Optional[str]) -> Dict:
    match_dates = {}
    if from_date_str or to_date_str:
        match_dates['timestamp'] = {}
        if from_date_str:
            try:
                start_dt = datetime.strptime(from_date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                match_dates['timestamp']['$gte'] = start_dt
            except ValueError:
                raise ValueError("Invalid from_date format. Use YYYY-MM-DD.")
        if to_date_str:
            try:
                end_dt = datetime.strptime(to_date_str, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
                match_dates['timestamp']['$lte'] = end_dt
            except ValueError:
                raise ValueError("Invalid to_date format. Use YYYY-MM-DD.")
    return match_dates


# def get_aggregated_stats(
#     group_by: TimeGroup,
#     from_date: Optional[str] = None,
#     to_date: Optional[str] = None,
#     camera_uuid: Optional[str] = None,
#     ambiente: Optional[str] = None,
#     piso: Optional[str] = None
# ) -> List[Dict[str, Any]]:
#     """
#     Calculates aggregated statistics (average persons) from historico_detecciones
#     using MongoDB's aggregation pipeline.

#     Args:
#         group_by: The time unit to group by (hour, weekday, date).
#         from_date: Start date string (YYYY-MM-DD).
#         to_date: End date string (YYYY-MM-DD).
#         camera_uuid: Filter by camera UUID.
#         ambiente: Filter by environment name.
#         piso: Filter by floor level.

#     Returns:
#         A list of dictionaries representing the aggregated results.
#         Example for group_by='hour': [{'hour': 8, 'average_persons': 15.5}, ...]
#         Example for group_by='weekday': [{'weekday': 2, 'average_persons': 12.1}, ...]
#         Example for group_by='date': [{'date': '2025-04-10', 'average_persons': 20.0}, ...]
#     """
#     pipeline = []

#     # --- 1. $match stage (Filtrado) ---
#     match_stage = {}
#     if camera_uuid:
#         match_stage['camera_uuid'] = camera_uuid
#     if ambiente:
#         match_stage['ambiente'] = ambiente
#     if piso:
#         match_stage['piso'] = piso

#     # Filtrado por fecha/hora
#     if from_date or to_date:
#         match_stage['timestamp'] = {}
#         if from_date:
#             try:
#                 # Asume YYYY-MM-DD y convierte a datetime al inicio del día en UTC
#                 start_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
#                 match_stage['timestamp']['$gte'] = start_dt
#             except ValueError:
#                 raise ValueError("Invalid from_date format. Use YYYY-MM-DD.")
#         if to_date:
#             try:
#                 # Asume YYYY-MM-DD y convierte a datetime al final del día en UTC
#                 end_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
#                 # Alternativa: añadir un día y usar $lt
#                 # end_dt = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1)
#                 # match_stage['timestamp']['$lt'] = end_dt
#                 match_stage['timestamp']['$lte'] = end_dt
#             except ValueError:
#                 raise ValueError("Invalid to_date format. Use YYYY-MM-DD.")

#     if match_stage:
#         pipeline.append({"$match": match_stage})

#     # --- 2. $project stage (Calcular total personas por registro/timestamp) ---
#     # Suma las personas detectadas en todas las zonas para cada registro individual.
#     pipeline.append({
#         "$project": {
#             "_id": 0, # Excluir el _id original del documento
#             "timestamp": 1,
#             "total_personas": {"$sum": "$detections.personas_detectadas"}
#         }
#     })

#     # --- 3. $group stage (Agrupar y Calcular Promedio) ---
#     group_stage = {"_id": None, "average_persons": {"$avg": "$total_personas"}}
#     if group_by == TimeGroup.HOUR:
#         # Agrupar por hora del día. Asegurarse que timestamp es interpretado en la zona horaria correcta.
#         # Por defecto, MongoDB usa UTC. Puedes especificar timezone si tus timestamps no lo son.
#         group_stage["_id"] = {"hour": {"$hour": "$timestamp"}} # Podrías añadir timezone: "America/Lima" si es necesario
#     elif group_by == TimeGroup.WEEKDAY:
#         group_stage["_id"] = {"weekday": {"$dayOfWeek": "$timestamp"}} # Podrías añadir timezone: "America/Lima"
#     elif group_by == TimeGroup.DATE:
#         group_stage["_id"] = {"date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}} # Podrías añadir timezone: "America/Lima"
#     else:
#         raise ValueError(f"Unsupported group_by value: {group_by}")

#     pipeline.append({"$group": group_stage})

#     # --- 4. $project stage (Reformatear Salida) ---
#     # Para hacer la salida más limpia y coincidente con los schemas Pydantic
#     project_stage_final = {"_id": 0} # Excluir el _id complejo generado por $group
#     if group_by == TimeGroup.HOUR:
#         project_stage_final["hour"] = "$_id.hour"
#     elif group_by == TimeGroup.WEEKDAY:
#         project_stage_final["weekday"] = "$_id.weekday"
#     elif group_by == TimeGroup.DATE:
#          # Convertir la fecha string a objeto date en el lado de Python si es necesario
#          # O mantener como string si es más fácil para el frontend
#         project_stage_final["date"] = "$_id.date" # Mantenido como string YYYY-MM-DD

#     project_stage_final["average_persons"] = "$average_persons"
#     pipeline.append({"$project": project_stage_final})


#     # --- 5. $sort stage (Ordenar resultados) ---
#     sort_stage = {}
#     if group_by == TimeGroup.HOUR:
#         sort_stage["hour"] = 1
#     elif group_by == TimeGroup.WEEKDAY:
#         sort_stage["weekday"] = 1
#     elif group_by == TimeGroup.DATE:
#         sort_stage["date"] = 1 # Ordenar por fecha

#     pipeline.append({"$sort": sort_stage})

#     # Ejecutar la agregación
#     results = list(historico_detecciones.aggregate(pipeline))

#     # Post-procesamiento para convertir 'date' string a objeto date si es necesario
#     if group_by == TimeGroup.DATE:
#         for item in results:
#             if 'date' in item and isinstance(item['date'], str):
#                 try:
#                     item['date'] = datetime.strptime(item['date'], "%Y-%m-%d").date()
#                 except ValueError:
#                     # Manejar caso si la fecha no es válida (aunque $dateToString debería prevenirlo)
#                     item['date'] = None # O algún valor por defecto

#     return results
# --- Función de agregación temporal (modificada levemente para claridad) ---
def get_aggregated_stats(
    group_by: TimeGroup,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    camera_uuid: Optional[str] = None,
    ambiente: Optional[str] = None,
    piso: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Calcula estadísticas temporales promedio, con filtros opcionales de ubicación."""
    pipeline = []
    match_stage = {}

    # Filtros de ubicación
    if camera_uuid: match_stage['camera_uuid'] = camera_uuid
    if ambiente: match_stage['ambiente'] = ambiente
    if piso: match_stage['piso'] = piso

    # Filtro de fechas
    match_dates = _build_date_match(from_date, to_date)
    if match_dates: match_stage.update(match_dates)

    if match_stage:
        pipeline.append({"$match": match_stage})

    # Proyecto para calcular total_personas (sin cambios)
    pipeline.append({
        "$project": {
            "_id": 0,
            "timestamp": 1,
            "total_personas": {"$sum": "$detections.personas_detectadas"}
        }
    })

    # Group por tiempo (sin cambios)
    group_stage = {"_id": None, "average_persons": {"$avg": "$total_personas"}}
    group_id_definition = {}
    sort_key = {}
    if group_by == TimeGroup.HOUR:
        group_id_definition = {"hour": {"$hour": "$timestamp"}}
        sort_key = {"hour": ASCENDING}
    elif group_by == TimeGroup.WEEKDAY:
        group_id_definition = {"weekday": {"$dayOfWeek": "$timestamp"}}
        sort_key = {"weekday": ASCENDING}
    elif group_by == TimeGroup.DATE:
        group_id_definition = {"date": {"$dateToString": {"format": "%Y-%m-%d", "date": "$timestamp"}}}
        sort_key = {"date": ASCENDING}
    else:
        raise ValueError(f"Unsupported group_by value: {group_by}")
    group_stage["_id"] = group_id_definition
    pipeline.append({"$group": group_stage})

    # Proyecto final para limpiar salida (sin cambios)
    project_stage_final = {"_id": 0, "average_persons": "$average_persons"}
    if group_by == TimeGroup.HOUR: project_stage_final["hour"] = "$_id.hour"
    elif group_by == TimeGroup.WEEKDAY: project_stage_final["weekday"] = "$_id.weekday"
    elif group_by == TimeGroup.DATE: project_stage_final["date"] = "$_id.date"
    pipeline.append({"$project": project_stage_final})

    # Sort (sin cambios)
    if sort_key:
      pipeline.append({"$sort": sort_key})

    results = list(historico_detecciones.aggregate(pipeline))

    # Post-procesamiento de fecha string a date (sin cambios)
    if group_by == TimeGroup.DATE:
        for item in results:
            if 'date' in item and isinstance(item['date'], str):
                try:
                    item['date'] = datetime.strptime(item['date'], "%Y-%m-%d").date()
                except ValueError: item['date'] = None
    return results

# --- Función original (Mantener si es necesaria para otros usos o como referencia) ---
def obtener_detecciones_por_fechas(camera_uuid=None, ambiente=None, piso=None, from_date=None, to_date=None):
    """Deprecated: Prefer get_aggregated_stats for statistics."""
    filtro = {}
    if camera_uuid:
        filtro['camera_uuid'] = camera_uuid
    if ambiente:
        filtro['ambiente'] = ambiente
    if piso:
        filtro['piso'] = piso
    if from_date or to_date:
        filtro['timestamp'] = {}
        if from_date:
            try:
                start_dt = datetime.strptime(from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
                filtro['timestamp']['$gte'] = start_dt
            except ValueError: pass # Manejar error apropiadamente
        if to_date:
             try:
                end_dt = datetime.strptime(to_date, "%Y-%m-%d").replace(hour=23, minute=59, second=59, microsecond=999999, tzinfo=timezone.utc)
                filtro['timestamp']['$lte'] = end_dt
             except ValueError: pass # Manejar error apropiadamente

    # Asegúrate de manejar las excepciones de conexión/consulta en una app real
    return list(historico_detecciones.find(filtro))

# --- NUEVA Función de agregación por ubicación ---
def get_stats_grouped_by_location(
    group_by_field: Literal["ambiente", "piso", "camera_uuid"],
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    ambiente: Optional[str] = None, # Filtro adicional
    piso: Optional[str] = None,     # Filtro adicional
    limit: Optional[int] = None
) -> List[Dict[str, Any]]:
    """Calcula estadísticas promedio agrupadas por un campo de ubicación."""
    pipeline = []
    match_stage = {}

    # Filtros (aplicados antes de agrupar)
    if ambiente: match_stage['ambiente'] = ambiente
    if piso: match_stage['piso'] = piso
    match_dates = _build_date_match(from_date, to_date)
    if match_dates: match_stage.update(match_dates)

    if match_stage:
        pipeline.append({"$match": match_stage})

    # Proyecto inicial para obtener total_personas y campos necesarios para $first
    project_stage_initial = {
        "_id": 0,
        "total_personas": {"$sum": "$detections.personas_detectadas"},
        "group_field": f"${group_by_field}" # Valor del campo por el que agruparemos
    }
    # Si agrupamos por cámara, necesitamos ambiente y piso para $first
    if group_by_field == "camera_uuid":
        project_stage_initial["ambiente"] = "$ambiente"
        project_stage_initial["piso"] = "$piso"
    pipeline.append({"$project": project_stage_initial})

    # Agrupación por el campo de ubicación especificado
    group_stage = {
        "_id": "$group_field", # Agrupar por ambiente, piso o camera_uuid
        "average_persons": {"$avg": "$total_personas"}
    }
    # Si agrupamos por cámara, recuperamos ambiente y piso
    if group_by_field == "camera_uuid":
        group_stage["ambiente"] = {"$first": "$ambiente"}
        group_stage["piso"] = {"$first": "$piso"}
    pipeline.append({"$group": group_stage})

    # Ordenar por promedio descendente si es por cámara (para ranking)
    if group_by_field == "camera_uuid":
         pipeline.append({"$sort": {"average_persons": DESCENDING}})
         if limit and limit > 0:
             pipeline.append({"$limit": limit})
    else:
         # Ordenar por nombre de ambiente/piso para consistencia
         pipeline.append({"$sort": {"_id": ASCENDING}})


    # Proyecto final para dar formato a la salida
    project_stage_final = {"_id": 0, "average_persons": "$average_persons"}
    project_stage_final[group_by_field] = "$_id" # Renombrar _id al nombre del campo
    if group_by_field == "camera_uuid":
        project_stage_final["ambiente"] = "$ambiente"
        project_stage_final["piso"] = "$piso"
    pipeline.append({"$project": project_stage_final})

    results = list(historico_detecciones.aggregate(pipeline))
    return results

# --- NUEVA Función de agregación por zona/mesa ---
def get_stats_grouped_by_zone(
    camera_uuid: str, # Requerido para este nivel de detalle
    from_date: Optional[str] = None,
    to_date: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Calcula estadísticas promedio por zona/label dentro de una cámara específica."""
    pipeline = []
    match_stage = {"camera_uuid": camera_uuid}

    # Filtro de fechas
    match_dates = _build_date_match(from_date, to_date)
    if match_dates: match_stage.update(match_dates)

    pipeline.append({"$match": match_stage})

    # Desenrollar el array de detecciones
    pipeline.append({"$unwind": "$detections"})

    # Agrupar por etiqueta de zona/mesa
    pipeline.append({
        "$group": {
            "_id": "$detections.label", # Agrupar por el label de la zona
            "average_persons": {"$avg": "$detections.personas_detectadas"}
        }
    })

        # --- NUEVA ETAPA $match ---
    # Filtrar resultados donde el label (_id) era null o no existente
    pipeline.append({
        "$match": {
            "_id": {"$ne": None}
        }
    })
    # --- FIN NUEVA ETAPA ---

    # Ordenar por nombre de zona
    pipeline.append({"$sort": {"_id": ASCENDING}})

    # Proyecto final (opcional, Pydantic puede mapear _id con alias)
    # pipeline.append({"$project": {"_id": 0, "zone_label": "$_id", "average_persons": 1}})

    results = list(historico_detecciones.aggregate(pipeline))
    return results

# --- NUEVA Función para obtener ubicaciones distintas ---
def get_distinct_locations() -> Dict[str, List[str]]:
    """Obtiene listas de ambientes y pisos únicos."""
    try:
        ambientes = historico_detecciones.distinct("ambiente")
        pisos = historico_detecciones.distinct("piso")
        return {
            "ambientes": sorted(ambientes),
            "pisos": sorted(pisos)
        }
    except Exception as e:
        print(f"Error fetching distinct locations: {e}")
        return {"ambientes": [], "pisos": []}






#