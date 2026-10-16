"""HTTP inference service for production integration."""

from __future__ import annotations

from contextlib import asynccontextmanager
from os.path import abspath, dirname, join
from typing import Sequence

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pctn.inference import BPInferenceService, BPPrediction
from pctn.utils import load_config

CONFIG_PATH = join(dirname(abspath(__file__)), "configs", "inference.yaml")

_service: BPInferenceService | None = None


def get_service() -> BPInferenceService:
    if _service is None:
        raise RuntimeError("Inference service is not initialized.")
    return _service


class PredictRequest(BaseModel):
    ppg: Sequence[float] = Field(..., description="1D raw PPG waveform.")
    fs: float = Field(..., gt=0, description="Sampling rate in Hz.")


class PredictResponse(BaseModel):
    sbp: float
    dbp: float
    n_windows: int


@asynccontextmanager
async def lifespan(_: FastAPI):
    global _service
    cfg = load_config(CONFIG_PATH)
    _service = BPInferenceService.from_config(CONFIG_PATH)
    host = cfg.get("serve", {}).get("host", "0.0.0.0")
    port = cfg.get("serve", {}).get("port", 8000)
    print(f"Inference service ready at http://{host}:{port}")
    print(f"Device: {_service.device}")
    yield
    _service = None


app = FastAPI(title="PPG Blood Pressure Inference", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        result: BPPrediction = get_service().predict(request.ppg, request.fs)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return PredictResponse(
        sbp=result.sbp,
        dbp=result.dbp,
        n_windows=result.n_windows,
    )


if __name__ == "__main__":
    import uvicorn

    cfg = load_config(CONFIG_PATH)
    serve_cfg = cfg.get("serve", {})
    uvicorn.run(
        "serve:app",
        host=serve_cfg.get("host", "0.0.0.0"),
        port=serve_cfg.get("port", 8000),
        reload=False,
    )
