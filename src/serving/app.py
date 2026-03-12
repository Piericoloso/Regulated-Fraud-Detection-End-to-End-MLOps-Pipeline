import time
import yaml
import mlflow
import pandas as pd
from fastapi import FastAPI

from src.common.logging import get_logger
from src.models.registry import get_model_uri
from src.serving.schemas import PredictRequest, PredictResponse

log = get_logger(__name__)
app = FastAPI(title="Regulated MLOps Inference API", version="0.1.0")

MODEL = None
MODEL_URI = None

def load_model():
    global MODEL, MODEL_URI
    cfg = yaml.safe_load(open("configs/serving.yaml", "r", encoding="utf-8"))
    mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
    MODEL_URI = get_model_uri(cfg["mlflow"]["registered_model_name"], cfg["mlflow"]["stage"])
    MODEL = mlflow.pyfunc.load_model(MODEL_URI)
    log.info("model_loaded", extra={"model_uri": MODEL_URI})

@app.on_event("startup")
def _startup():
    load_model()

@app.get("/health")
def health():
    return {"status": "ok", "model_uri": MODEL_URI}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    t0 = time.time()
    df = pd.DataFrame([req.model_dump()])
    prob = float(MODEL.predict(df)[0])  # signature is probability
    decision = int(prob >= 0.5)

    log.info(
        "prediction",
        extra={
            "model_uri": MODEL_URI,
            "probability": prob,
            "decision": decision,
            "latency_ms": int((time.time() - t0) * 1000),
            "features": req.model_dump(),
        },
    )
    return PredictResponse(probability=prob, decision=decision, model_uri=MODEL_URI)
