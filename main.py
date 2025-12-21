from fastapi import FastAPI, Body, Request
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI()

# --- THE POST ENDPOINT ---
@app.post("/predict_batch/{device_id}")
async def predict_batch(device_id: str, all_scans: list = Body(...)):
    """
    PASTE YOUR DATA HERE:
    1. Click 'Try it out' (Top Right).
    2. A white 'Request body' box WILL appear below.
    3. Paste your [[18 values], ... x10] list into that box.
    """
    try:
        # Load the brain
        MODEL_PACK = 'soil_model_pack_rf.pkl'
        TARGET_MAP = {
            'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K', 
            'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC', 
            'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S'
        }

        if not os.path.exists(MODEL_PACK):
            return {"status": "error", "message": "Model file not found"}

        with open(MODEL_PACK, 'rb') as f:
            pkg = pickle.load(f)

        # Process each scan individually
        batch_results = []
        for scan in all_scans:
            input_data = np.array(scan).reshape(1, -1)
            scaled_data = pkg['scaler'].transform(input_data)
            
            single_pred = {}
            for raw, clean in TARGET_MAP.items():
                val = float(pkg['models'][raw].predict(scaled_data)[0])
                single_pred[clean] = round(max(0.0, val), 3)
            batch_results.append(single_pred)

        return {"status": "success", "device": device_id, "results": batch_results}

    except Exception as e:
        return {"status": "error", "message": str(e)}