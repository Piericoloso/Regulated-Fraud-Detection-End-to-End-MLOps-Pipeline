from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    amount: float = Field(..., ge=0)
    hour: int = Field(..., ge=0, le=23)
    country_risk: int = Field(..., ge=1, le=5)
    is_cross_border: int = Field(..., ge=0, le=1)
    customer_tenure_days: int = Field(..., ge=1)
    txns_7d: int = Field(..., ge=0)
    chargebacks_365d: int = Field(..., ge=0)

class PredictResponse(BaseModel):
    probability: float
    decision: int
    model_uri: str
