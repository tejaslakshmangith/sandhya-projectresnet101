"""FastAPI backend for SmartMine AI Safety Detection."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# ---------------------------------------------------------------------------
# Add the ai-model directory to sys.path so its modules are importable.
# The folder name "ai-model" contains a hyphen, so it cannot be imported
# as a regular package; we add it directly to sys.path instead.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "ai-model"))

from inference import load_model, predict_from_bytes  # noqa: E402

# ---------------------------------------------------------------------------
# Configuration (override via environment variables)
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_PATH = str(_REPO_ROOT / "ai-model" / "models" / "resnet101_smartmine.pth")
_DEFAULT_CLASS_NAMES = "safe,unsafe,helmet,hazard"

MODEL_PATH: str = os.environ.get("MODEL_PATH", _DEFAULT_MODEL_PATH)
CLASS_NAMES: list[str] = os.environ.get("CLASS_NAMES", _DEFAULT_CLASS_NAMES).split(",")

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="SmartMine AI Safety Detection API",
    description="REST API for mine-safety image classification using ResNet-101.",
    version="1.0.0",
)

# NOTE: allow_origins="*" is intentional for local development.
# In production, set the ALLOWED_ORIGINS environment variable to a
# comma-separated list of trusted frontend origins, e.g.:
#   ALLOWED_ORIGINS=https://my-app.example.com
_raw_origins = os.environ.get("ALLOWED_ORIGINS", "*")
_allowed_origins: list[str] = (
    ["*"] if _raw_origins == "*" else [o.strip() for o in _raw_origins.split(",")]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=_raw_origins != "*",
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model state
_model: Any = None
_class_names: list[str] = CLASS_NAMES
_model_loaded: bool = False


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@app.on_event("startup")
async def startup_event() -> None:
    """Load the ResNet-101 model once when the server starts."""
    global _model, _class_names, _model_loaded
    try:
        _model, _class_names = load_model(
            model_path=MODEL_PATH,
            num_classes=len(CLASS_NAMES),
            device="cpu",
        )
        _model_loaded = True
        print(f"[startup] Model loaded from {MODEL_PATH}")
        print(f"[startup] Classes: {_class_names}")
    except FileNotFoundError:
        # Server starts in a degraded state; /predict will return 503
        _model_loaded = False
        print(f"[startup] WARNING: model file not found at {MODEL_PATH}. "
              "Train the model first and set MODEL_PATH accordingly.")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict:
    """Health check endpoint."""
    return {"status": "ok", "model_loaded": _model_loaded}


@app.get("/classes")
async def classes() -> dict:
    """Return the list of class names the model was trained on."""
    return {"classes": _class_names}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> dict:
    """Classify a mine-safety image.

    Accepts a multipart/form-data upload with field name ``file``.

    Returns:
        JSON with ``prediction``, ``confidence``, ``all_probabilities``,
        and ``status``.
    """
    if not _model_loaded or _model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please train the model first.",
        )

    # Validate content type
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=422,
            detail=f"Expected an image file, got content-type '{file.content_type}'.",
        )

    try:
        image_bytes = await file.read()
        result = predict_from_bytes(image_bytes, _model, _class_names, device="cpu")
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(status_code=422, detail="Could not decode the uploaded image.") from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail="An unexpected error occurred during prediction.") from exc

    return {
        "prediction": result["class_name"],
        "confidence": result["probability"],
        "all_probabilities": result["all_probabilities"],
        "status": "success",
    }


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
