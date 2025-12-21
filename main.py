from fastapi import FastAPI, Request
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
async def predict_batch(device_id: str, request: Request):
    try:
        # 1. Catch the Single Throw (List of 10 scans)
        all_scans = await request.json() 
        
        with open(MODEL_PACK, 'rb') as f:
            pkg = pickle.load(f)

        # 2. Process each scan individually
        batch_results = []
        for scan in all_scans:
            # Prepare the single scan for the model
            input_data = np.array(scan).reshape(1, -1)
            scaled_data = pkg['scaler'].transform(input_data)
            
            # Get predictions for all 12 parameters
            single_prediction = {}
            for raw_key, clean_key in TARGET_MAP.items():
                val = float(pkg['models'][raw_key].predict(scaled_data)[0])
                single_prediction[clean_key] = round(max(0.0, val), 3)
            
            batch_results.append(single_prediction)

        # 3. Return the full list of 10 results
        return {
            "status": "success",
            "device": device_id,
            "results": batch_results  # This is a list of 10 objects
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}