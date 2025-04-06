from pydantic import BaseModel
from datetime import datetime
from typing import List
from typing import Optional

class Detection(BaseModel):
    id: str
    label: str
    personas_detectadas: int
    max_personas: int
    area: float
    perimeter: float
    points: List[List[float]]
    color: List[int]

class HistoricoDeteccion(BaseModel):
    camera_uuid: str
    ambiente: str
    piso: str
    timestamp: datetime
    detections: List[Detection]

class StatsResponse(BaseModel):
    camera_uuid: Optional[str] = None
    ambiente: Optional[str] = None
    piso: Optional[str] = None  # <— hacer opcional aquí
    interval: str
    data: List[dict]