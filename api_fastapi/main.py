from typing import Optional
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from utils import obtener_detecciones_por_fechas
from schemas import StatsResponse
from collections import defaultdict
import uvicorn

app = FastAPI()

# Permitir peticiones desde frontend Django
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # En producción limitar al dominio específico
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/stats/cameras/{camera_uuid}", response_model=StatsResponse)
def stats_por_camara(
    camera_uuid: str,
    from_date: str = Query(None, example="2025-04-01"),
    to_date: str = Query(None, example="2025-04-30"),
    interval: str = Query("daily", example="daily"),
):
    registros = obtener_detecciones_por_fechas(camera_uuid=camera_uuid, from_date=from_date, to_date=to_date)
    datos_agrupados = defaultdict(list)

    for registro in registros:
        fecha = registro['timestamp'].strftime("%Y-%m-%d") if interval == "daily" else registro['timestamp'].strftime("%Y-%m")
        total_personas = sum(d['personas_detectadas'] for d in registro['detections'])
        datos_agrupados[fecha].append(total_personas)

    response_data = [
        {"date": fecha, "average_personas_detectadas": sum(vals) / len(vals)}
        for fecha, vals in datos_agrupados.items()
    ]

    return StatsResponse(
        camera_uuid=camera_uuid,
        interval=interval,
        data=response_data
    )

@app.get("/api/stats/ambiente/{ambiente}", response_model=StatsResponse)
def stats_por_ambiente(
    ambiente: str, 
    from_date: str, 
    to_date: str, 
    piso: Optional[str] = None, 
    interval: str = "daily"
):
    from datetime import datetime, timedelta
    from pymongo import MongoClient

    client = MongoClient("mongodb://localhost:27017/")
    db = client["logs_db"]
    collection = db["historico_detecciones"]

    start_date = datetime.strptime(from_date, "%Y-%m-%d")
    end_date = datetime.strptime(to_date, "%Y-%m-%d") + timedelta(days=1) - timedelta(seconds=1)

    query = {
        "ambiente": ambiente,
        "timestamp": {"$gte": start_date, "$lte": end_date}
    }

    if piso:
        query["piso"] = piso

    cursor = collection.find(query)

    data = []
    for doc in cursor:
        data.append({
            "camera_uuid": doc.get("camera_uuid"),
            "ambiente": doc.get("ambiente"),
            "piso": doc.get("piso"),
            "timestamp": doc.get("timestamp"),
            "detections": doc.get("detections"),
        })

    return StatsResponse(
        camera_uuid=None,
        ambiente=ambiente,
        piso=piso,
        interval=interval,
        data=data
    )




@app.get("/api/stats/summary", response_model=StatsResponse)
def resumen_general(
    date: str = Query(None, example="2025-04-06")
):
    registros = obtener_detecciones_por_fechas(from_date=date, to_date=date)
    total_personas = sum(
        sum(d['personas_detectadas'] for d in registro['detections']) for registro in registros
    )
    total_registros = len(registros)

    data = [{"date": date, "total_personas_detectadas": total_personas, "total_registros": total_registros}]

    return StatsResponse(
        interval="daily",
        data=data
    )

# Inicio servidor
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)
