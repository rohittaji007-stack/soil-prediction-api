from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pickle
import os

app = FastAPI()

# CORS: allow the browser-based device dashboard to call this API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PACK = 'soil_model_pack_rf.pkl'

# Canonical target name -> short key used by the frontend/report.
TARGET_MAP = {
    'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K',
    'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC',
    'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S',
}
CLASS_LABELS = {0: 'Low', 1: 'Medium', 2: 'High'}

_PACK = None


def _load_pack():
    """Load the model pack once and cache it."""
    global _PACK
    if _PACK is None:
        with open(MODEL_PACK, 'rb') as f:
            _PACK = pickle.load(f)
    return _PACK


def _l1_normalise(X):
    """Per-scan normalisation -- MUST match train.py.l1_normalise exactly."""
    X = np.asarray(X, dtype=float)
    return X / (X.sum(axis=1, keepdims=True) + 1e-9)


@app.get("/health")
def health():
    if not os.path.exists(MODEL_PACK):
        return {"status": "error", "message": "Model file (.pkl) not found."}
    pkg = _load_pack()
    return {
        "status": "ok",
        "sklearn_version": pkg.get("sklearn_version"),
        "tiers": pkg.get("tiers"),
        "metrics": pkg.get("metrics"),
    }


@app.post("/predict_batch/{device_id}")
async def predict_batch(device_id: str, all_scans: list = Body(...)):
    """
    Returns ONLY what the training data can actually support:
      - 'measured'  parameters -> averaged numeric value (+ its CV R^2)
      - 'screening' parameters -> Low/Med/High class     (+ its CV accuracy)
      - 'unavailable' parameters are NOT predicted; they are listed so the
        UI can show "not available" instead of a fabricated number.
    """
    try:
        if not os.path.exists(MODEL_PACK):
            return {"status": "error", "message": "Model file (.pkl) not found."}

        pkg = _load_pack()
        tiers = pkg["tiers"]
        metrics = pkg["metrics"]
        units = pkg.get("units", {})

        # Validate + preprocess every scan (each must be 18 channels).
        rows = []
        for scan in all_scans:
            if not isinstance(scan, (list, tuple)) or len(scan) != 18:
                return {"status": "error",
                        "message": "Each scan must be an array of 18 values."}
            rows.append(scan)
        if not rows:
            return {"status": "error", "message": "No scans provided."}

        X = pkg["scaler"].transform(_l1_normalise(rows))  # (n_scans, 18)

        final_report = {}
        unavailable = []

        for raw, short in TARGET_MAP.items():
            tier = tiers.get(raw, "unavailable")

            if tier == "measured":
                # Average the numeric prediction across all scans.
                preds = pkg["models"][raw].predict(X)
                value = round(max(0.0, float(np.mean(preds))), 2)
                final_report[short] = {
                    "type": "value",
                    "value": value,
                    "unit": units.get(raw, ""),
                    "reliability": "measured",
                    "cv_r2": metrics[raw]["r2"],
                }

            elif tier == "screening":
                # Average class probabilities across scans, then pick the mode.
                proba = pkg["classifiers"][raw].predict_proba(X).mean(axis=0)
                idx = int(np.argmax(proba))
                final_report[short] = {
                    "type": "class",
                    "class": CLASS_LABELS.get(idx, str(idx)),
                    "confidence": round(float(proba[idx]), 2),
                    "reliability": "screening",
                    "cv_accuracy": metrics[raw]["acc"],
                }

            else:
                unavailable.append(short)

        return {
            "status": "success",
            "device": device_id,
            "samples_processed": len(rows),
            "final_report": final_report,
            "unavailable": unavailable,
            "disclaimer": (
                "Prototype screening only. 'measured' values and 'screening' "
                "classes are limited by an 18-channel (410-940nm) sensor and a "
                "small training set; they are not a substitute for a soil lab. "
                "Parameters in 'unavailable' had no reliable signal in training "
                "and are intentionally not reported."
            ),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
