from pymongo import MongoClient
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()

client = MongoClient(os.getenv("MONGODB_URL"))
db = client["logs_db"]
historico_detecciones = db['historico_detecciones']

def obtener_detecciones_por_fechas(camera_uuid=None, ambiente=None, piso=None, from_date=None, to_date=None):
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
            filtro['timestamp']['$gte'] = datetime.fromisoformat(from_date)
        if to_date:
            filtro['timestamp']['$lte'] = datetime.fromisoformat(to_date)

    return list(historico_detecciones.find(filtro))
