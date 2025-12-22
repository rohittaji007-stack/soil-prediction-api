from fastapi import FastAPI, Body, Request
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI()

# --- CONFIGURATION ---
MODEL_PACK = 'soil_model_pack_rf.pkl'
TARGET_MAP = {
    'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K', 
    'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC', 
    'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S'
}

@app.post("/predict_batch/{device_id}")
async def predict_batch(device_id: str, all_scans: list = Body(...)):
    try:
        # 1. Check if the uploaded brain exists
        if not os.path.exists(MODEL_PACK):
            return {"status": "error", "message": "Model file (.pkl) not found. Please upload it to GitHub."}

        # 2. Load the pre-trained brain
        with open(MODEL_PACK, 'rb') as f:
            pkg = pickle.load(f)

        # 3. Run predictions on the 10-scan 'Single Throw'
        batch_results = []
        for scan in all_scans:
            # Prepare data
            input_data = np.array(scan).reshape(1, -1)
            scaled_data = pkg['scaler'].transform(input_data)
            
            # Predict all 12 parameters
            single_pred = {}
            for raw, clean in TARGET_MAP.items():
                val = float(pkg['models'][raw].predict(scaled_data)[0])
                single_pred[clean] = round(max(0.0, val), 3)
            
            batch_results.append(single_pred)

        return {
            "status": "success",
            "device": device_id,
            "results": batch_results
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}