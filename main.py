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

# N, P, K are modelled in ppm but the report shows kg/hectare.
# 1 ppm ~= 2.24 kg/ha for a 15 cm furrow slice (standard soil-card conversion).
PPM_TO_KGHA = 2.24
KGHA_KEYS = {'N', 'P', 'K'}

# Display unit per short key.
REPORT_UNITS = {
    'N': 'kg/ha', 'P': 'kg/ha', 'K': 'kg/ha', 'OC': '%', 'PH': '',
    'EC': 'dS/m', 'FE': 'ppm', 'MN': 'ppm', 'CU': 'ppm', 'ZN': 'ppm',
    'B': 'ppm', 'S': 'ppm',
}

# Training reliability tier -> confidence label shown to the user.
TIER_TO_CONFIDENCE = {
    'measured': 'high',       # genuine signal (e.g. N)
    'screening': 'moderate',  # coarse Low/Med/High only (pH, Fe, Cu)
    'unavailable': 'low',     # weak signal -- indicative estimate only
}

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
    Returns a numeric estimate for ALL 12 parameters (a full report), each
    tagged with an honest confidence level derived from cross-validation on
    the training data:

        confidence "high"     -> real signal (e.g. N)
        confidence "moderate" -> coarse Low/Med/High only (pH, Fe, Cu)
        confidence "low"      -> weak signal; indicative estimate only

    The numbers are genuine model outputs, NOT fabricated or rescaled to look
    accurate. Low-confidence values should be shown as estimates, and this is
    NOT a substitute for a soil laboratory.
    """
    try:
        if not os.path.exists(MODEL_PACK):
            return {"status": "error", "message": "Model file (.pkl) not found."}

        pkg = _load_pack()
        tiers = pkg["tiers"]
        metrics = pkg["metrics"]

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
        for raw, short in TARGET_MAP.items():
            tier = tiers.get(raw, "unavailable")

            # Averaged numeric prediction across all scans.
            value = float(np.mean(pkg["models"][raw].predict(X)))
            if short in KGHA_KEYS:
                value *= PPM_TO_KGHA          # ppm -> kg/ha for N, P, K
            value = round(max(0.0, value), 2)

            entry = {
                "value": value,
                "unit": REPORT_UNITS.get(short, ""),
                "confidence": TIER_TO_CONFIDENCE.get(tier, "low"),
                "cv_r2": metrics[raw]["r2"],
            }
            # For the moderate tier, also expose the Low/Med/High screening class.
            if tier == "screening":
                proba = pkg["classifiers"][raw].predict_proba(X).mean(axis=0)
                idx = int(np.argmax(proba))
                entry["class"] = CLASS_LABELS.get(idx, str(idx))
                entry["cv_accuracy"] = metrics[raw]["acc"]

            final_report[short] = entry

        return {
            "status": "success",
            "device": device_id,
            "samples_processed": len(rows),
            "final_report": final_report,
            "disclaimer": (
                "Estimated by an 18-channel (410-940nm) spectral sensor. "
                "Only 'high' confidence values (N) and 'moderate' confidence "
                "values (pH, Fe, Cu) are reliable; 'low' confidence values are "
                "indicative estimates and should be confirmed by a soil lab."
            ),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
