from __future__ import annotations

from fastapi import FastAPI, HTTPException
from app.schemas import ChurnRequest, ChurnResponse
from app.predict import ChurnModel, ModelNotReady

app = FastAPI(title="Churn ML API", version="1.0.0")

@app.get("/")
def root():
    return {
        "name": "Churn ML API",
        "status": "ok",
        "docs": "/docs",
        "health": "/health"
    }

model = ChurnModel()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=ChurnResponse)
def predict(req: ChurnRequest):
    try:
        proba = model.predict_proba(req.model_dump())
    except ModelNotReady as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    return ChurnResponse(
        churn_probability=proba,
        will_churn=proba >= 0.5,
        version="v1",
    )
