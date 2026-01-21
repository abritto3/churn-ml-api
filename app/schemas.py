from __future__ import annotations
from pydantic import BaseModel, Field


class ChurnRequest(BaseModel):
    tenure_months: int = Field(..., ge=0, le=120, description="Customer tenure in months")
    monthly_charges: float = Field(..., ge=0, le=500)
    total_charges: float = Field(..., ge=0, le=100000)

    contract_type: str = Field(..., description="month-to-month | one_year | two_year")
    internet_service: str = Field(..., description="dsl | fiber | none")
    payment_method: str = Field(..., description="electronic_check | mailed_check | bank_transfer | credit_card")
    paperless_billing: bool


class ChurnResponse(BaseModel):
    churn_probability: float
    will_churn: bool
    version: str = "v1"

