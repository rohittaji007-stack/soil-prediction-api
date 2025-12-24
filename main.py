from fastapi import FastAPI, Body, Request
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI()

# ==========================================
# ADD CORS MIDDLEWARE - This fixes the 405 error!
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (you can restrict this later)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods including OPTIONS
    allow_headers=["*"],  # Allows all headers including Content-Type
)

MODEL_PACK = 'soil_model_pack_rf.pkl'
TARGET_MAP = {
    'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K', 
    'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC', 
    'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S'
}

@app.post("/predict_batch/{device_id}")
async def predict_batch(device_id: str, all_scans: list = Body(...)):
    try:
        if not os.path.exists(MODEL_PACK):
            return {"status": "error", "message": "Model file (.pkl) not found."}

        with open(MODEL_PACK, 'rb') as f:
            pkg = pickle.load(f)

        # 1. Run all 10 predictions
        all_predictions = []
        for scan in all_scans:
            input_data = np.array(scan).reshape(1, -1)
            scaled_data = pkg['scaler'].transform(input_data)
            
            single_pred = {}
            for raw, clean in TARGET_MAP.items():
                val = float(pkg['models'][raw].predict(scaled_data)[0])
                single_pred[clean] = val
            all_predictions.append(single_pred)

        # 2. Calculate the AVERAGE for each of the 12 parameters
        final_averaged_report = {}
        for clean_key in TARGET_MAP.values():
            # Sum up the value from all 10 predictions and divide by 10
            avg_val = sum(p[clean_key] for p in all_predictions) / len(all_predictions)
            final_averaged_report[clean_key] = round(max(0.0, avg_val), 3)

        # 3. Return only the SINGLE averaged result
        return {
            "status": "success",
            "device": device_id,
            "samples_processed": len(all_scans),
            "final_report": final_averaged_report
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
