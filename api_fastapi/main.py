# api_fastapi/main.py
from fastapi import FastAPI, Query, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List, Dict, Any, Literal # Importar Literal
from datetime import date
import uvicorn

# Importar funciones y schemas actualizados
from utils import (
    get_aggregated_stats,
    get_stats_grouped_by_location,
    get_stats_grouped_by_zone,
    get_distinct_locations
)
from schemas import (
    AggregatedStatsResponse, # Para /average
    LocationStatsResponse,   # Para /average_by_location
    ZoneStatsResponse,       # Para /average_by_zone
    DistinctLocationsResponse, # Para /distinct_locations
    AmbienteStat, CameraStat, ZoneStat, # Sub-schemas
    TimeGroup,
    StatsResponse # Schema Deprecated
)

app = FastAPI(
    title="YoloVision Statistics API",
    description="API para obtener estadísticas de detección de personas.",
    version="1.0.0"
)

# Configuración CORS (mantener como estaba o ajustar según necesidad)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ¡Restringir en producción!
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Nuevo Endpoint de Estadísticas Agregadas ---

# --- Endpoint existente (Verificar que recibe ambiente/piso) ---
# Definir funciones de dependencia para filtros
def get_date_filters(
    from_date: Optional[date] = Query(None, description="Start date (YYYY-MM-DD)", example="2025-04-01"),
    to_date: Optional[date] = Query(None, description="End date (YYYY-MM-DD)", example="2025-04-10")
) -> Dict:
    return {"from_date": from_date, "to_date": to_date}

def get_location_filters(
    camera_uuid: Optional[str] = Query(None),
    ambiente: Optional[str] = Query(None),
    piso: Optional[str] = Query(None)
) -> Dict:
    return {"camera_uuid": camera_uuid, "ambiente": ambiente, "piso": piso}

@app.get(
    "/api/stats/average",
    response_model=AggregatedStatsResponse,
    summary="Get Aggregated Occupancy Statistics (Temporal)",
    description="Calculates average persons grouped by time (hour, weekday, date), optionally filtered by location."
)
def get_average_stats_endpoint( # Renombrado para claridad interna
    # Dependencias de Filtros Comunes
    date_filters: Dict = Depends(get_date_filters),
    location_filters: Dict = Depends(get_location_filters),
    # Parámetro específico
    group_by: TimeGroup = Query(..., description="Time interval for grouping")
):
    try:
        from_date_str = date_filters['from_date'].isoformat() if date_filters.get('from_date') else None
        to_date_str = date_filters['to_date'].isoformat() if date_filters.get('to_date') else None

        aggregated_data = get_aggregated_stats(
            group_by=group_by,
            from_date=from_date_str,
            to_date=to_date_str,
            camera_uuid=location_filters.get('camera_uuid'),
            ambiente=location_filters.get('ambiente'),
            piso=location_filters.get('piso')
        )

        filters_applied = {
            "group_by": group_by, # Incluir group_by en filtros para claridad
            "from_date": from_date_str, "to_date": to_date_str,
            **{k: v for k, v in location_filters.items() if v is not None} # Añadir filtros de ubicación usados
        }

        response_data = [ {**item, 'average_persons': round(item.get('average_persons', 0.0), 2)} for item in aggregated_data ]

        return AggregatedStatsResponse(
            group_by=group_by, # Mantener aquí también para compatibilidad
            filters=filters_applied,
            data=response_data
        )
    # ... (Manejo de errores sin cambios) ...
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in /api/stats/average: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")




# --- NUEVO Endpoint para Agregación por Ubicación ---
@app.get(
    "/api/stats/average_by_location",
    # response_model=LocationStatsResponse, # El modelo exacto depende del group_by
    summary="Get Aggregated Occupancy Statistics (by Location)",
    description="Calculates average persons grouped by 'ambiente' or 'camera_uuid', optionally filtered."
)
async def get_average_by_location_endpoint(
    # Dependencias de Filtros Comunes
    date_filters: Dict = Depends(get_date_filters),
    location_filters: Dict = Depends(get_location_filters),
    # Parámetro específico
    group_by_location: Literal["ambiente", "camera_uuid"] = Query(..., alias="group_by", description="Location field to group by ('ambiente' or 'camera_uuid')"),
    limit: Optional[int] = Query(10, description="Limit results (used for camera ranking)", ge=1) # Default 10 para cámaras
):
    try:
        from_date_str = date_filters['from_date'].isoformat() if date_filters.get('from_date') else None
        to_date_str = date_filters['to_date'].isoformat() if date_filters.get('to_date') else None

        # Solo aplicar límite si se agrupa por cámara
        effective_limit = limit if group_by_location == "camera_uuid" else None

        aggregated_data = get_stats_grouped_by_location(
            group_by_field=group_by_location,
            from_date=from_date_str,
            to_date=to_date_str,
            ambiente=location_filters.get('ambiente'),
            piso=location_filters.get('piso'),
            limit=effective_limit
        )

        filters_applied = {
            "group_by": group_by_location,
            "from_date": from_date_str, "to_date": to_date_str,
             **{k: v for k, v in location_filters.items() if v is not None}
        }
        if effective_limit: filters_applied["limit"] = effective_limit

        # Formatear y redondear
        response_data = [ {**item, 'average_persons': round(item.get('average_persons', 0.0), 2)} for item in aggregated_data ]

        # Devolver usando un diccionario genérico por ahora, ya que el tipo de 'data' varía
        # O podrías tener lógica para elegir el ResponseModel correcto
        return {"filters": filters_applied, "data": response_data}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in /api/stats/average_by_location: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# --- NUEVO Endpoint para Agregación por Zona/Mesa ---
@app.get(
    "/api/stats/average_by_zone",
    response_model=ZoneStatsResponse,
    summary="Get Aggregated Occupancy Statistics (by Zone/Label)",
    description="Calculates average persons per detection zone/label within a specific camera."
)
async def get_average_by_zone_endpoint(
     # Dependencias de Filtros Comunes
    date_filters: Dict = Depends(get_date_filters),
    # Parámetro específico y requerido
    camera_uuid: str = Query(..., description="UUID of the camera to analyze")
):
    try:
        from_date_str = date_filters['from_date'].isoformat() if date_filters.get('from_date') else None
        to_date_str = date_filters['to_date'].isoformat() if date_filters.get('to_date') else None

        aggregated_data = get_stats_grouped_by_zone(
            camera_uuid=camera_uuid,
            from_date=from_date_str,
            to_date=to_date_str
        )

        filters_applied = {
            "camera_uuid": camera_uuid,
            "from_date": from_date_str, "to_date": to_date_str
        }

        # Pydantic mapeará _id a zone_label y redondeará
        response_data = [ {**item, 'average_persons': round(item.get('average_persons', 0.0), 2)} for item in aggregated_data ]

        return ZoneStatsResponse(filters=filters_applied, data=response_data)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        print(f"Error in /api/stats/average_by_zone: {e}")
        raise HTTPException(status_code=500, detail="Internal server error.")

# --- NUEVO Endpoint para obtener Ubicaciones Distintas ---
@app.get(
    "/api/distinct_locations",
    response_model=DistinctLocationsResponse,
    summary="Get distinct Ambiente and Piso values",
    description="Retrieves lists of unique 'ambiente' and 'piso' values found in the historical data."
)
async def get_distinct_locations_endpoint():
    try:
        locations = get_distinct_locations()
        return locations
    except Exception as e:
        print(f"Error in /api/distinct_locations: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve distinct locations.")




# Inicio servidor (sin cambios)
if __name__ == "__main__":
    # Escuchar en todas las interfaces, útil para Docker o acceso en red local
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=True)