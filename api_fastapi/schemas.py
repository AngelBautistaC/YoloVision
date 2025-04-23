# api_fastapi/schemas.py
from __future__ import annotations # <-- Añadir esta línea al inicio
from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import List, Optional, Union, Dict, Any
from enum import Enum

# Enum para definir las opciones de agrupación
class TimeGroup(str, Enum):
    HOUR = "hour"
    WEEKDAY = "weekday"
    DATE = "date"

# --- Schemas existentes (pueden mantenerse si se usan en otro lugar) ---
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

# --- Nuevos Schemas para Respuestas Agregadas ---

class HourlyStat(BaseModel):
    hour: int = Field(..., description="Hour of the day (0-23)")
    average_persons: float = Field(..., description="Average number of persons detected during this hour")

class WeekdayStat(BaseModel):
    # MongoDB $dayOfWeek: Sunday=1, Saturday=7. Consider mapping this in frontend/backend if needed.
    weekday: int = Field(..., description="Day of the week (1=Sunday, 7=Saturday)")
    average_persons: float = Field(..., description="Average number of persons detected on this day of the week")

class DailyStat(BaseModel):
    statistic_date: date = Field(..., alias="date", description="Specific date for the statistic")
    average_persons: float = Field(..., description="Average number of persons detected on this date")

# Schema de respuesta unificado para estadísticas agregadas
class AggregatedStatsResponse(BaseModel):
    group_by: TimeGroup = Field(..., description="The time interval used for grouping the data")
    filters: Dict[str, Any] = Field(..., description="Filters applied to the query (dates, location, etc.)")
    data: List[Union[HourlyStat, WeekdayStat, DailyStat]] = Field(..., description="List of aggregated data points")

# Schema de respuesta para el endpoint original (si decides mantenerlo o como referencia)
# Puedes marcarlo como deprecated si lo reemplazas completamente.
class StatsResponse(BaseModel):
    """
    Deprecated: Use AggregatedStatsResponse with the /api/stats/average endpoint instead.
    """
    camera_uuid: Optional[str] = None
    ambiente: Optional[str] = None
    piso: Optional[str] = None
    interval: str # Mantenido como str para compatibilidad si es necesario
    data: List[dict] # Mantenido genérico para compatibilidad


class AmbienteStat(BaseModel):
    ambiente: str
    average_persons: float

class CameraStat(BaseModel):
    camera_uuid: str
    nombre_camara: Optional[str]  # <-- Añadir aquí
    ambiente: str
    piso: str
    average_persons: float

class ZoneStat(BaseModel):
    zone_label: str = Field(..., alias="_id") # Mapear _id del $group a zone_label
    average_persons: float

class LocationStatsResponse(BaseModel):
    filters: Dict[str, Any]
    data: List[Union[AmbienteStat, CameraStat]] # Ajustar según el endpoint

class ZoneStatsResponse(BaseModel):
    filters: Dict[str, Any]
    data: List[ZoneStat]

class DistinctLocationsResponse(BaseModel):
    ambientes: List[str]
    pisos: List[str]
