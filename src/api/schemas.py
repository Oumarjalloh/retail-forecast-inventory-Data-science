from pydantic import BaseModel, Field
from typing import List, Optional

class ForecastRequest(BaseModel):
    store_id: int
    item_id: str
    horizon_days: int = Field(14, ge=1, le=60)
    on_hand: float = Field(0, ge=0)
    lead_time_days: int = Field(7, ge=1, le=60)
    service_level: float = Field(0.95, ge=0.5, le=0.999)

class ForecastPoint(BaseModel):
    date: str
    yhat: float
    yhat_lower: float
    yhat_upper: float

class ForecastResponse(BaseModel):
    series: List[ForecastPoint]
    stock: dict
    meta: dict