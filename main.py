from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, HTMLResponse
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List

app = FastAPI()

# --- 1. SETTINGS ---
CSV_FILE = 'final_training_base.csv'
MODEL_PACK = 'soil_model_pack_rf.pkl'
TARGET_MAP = {'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K', 'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC', 'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S'}

# Memory storage for 10+ devices
device_data: Dict[str, List[List[float]]] = {}

# --- 2. HOME PAGE (Fixes the "Not Found" on the main link) ---
@app.get("/", response_class=HTMLResponse)
async def root():
    return "<html><body style='font-family:Arial;text-align:center;margin-top:100px;'><h1>🌱 Soil API is Online</h1><p>Use <b>/predict/DeviceID?data=...</b> to send scans.</p></body></html>"

# --- 3. THE PREDICT ENDPOINT (Stable Query Pattern) ---
@app.get("/predict/{device_id}")
async def predict(device_id: str, data: str = None):
    # Load model
    if not os.path.exists(MODEL_PACK):
        return {"status": "error", "message": "Brain file missing. Ensure CSV is uploaded."}
    
    with open(MODEL_PACK, 'rb') as f:
        pkg = pickle.load(f)

    # If no data is sent, show the current status of that device
    if not data:
        count = len(device_data.get(device_id, []))
        return {"status": "pending", "device": device_id, "scans": f"{count}/10"}

    try:
        # Convert data string to list
        vals = [float(x) for x in data.split(',')]
        if len(vals) != 18:
            return {"status": "error", "message": "Need exactly 18 values separated by commas."}

        # Store data in device-specific room
        if device_id not in device_data:
            device_data[device_id] = []
        device_data[device_id].append(vals)

        # At 10 scans, calculate Random Forest
        if len(device_data[device_id]) >= 10:
            avg = np.mean(device_data[device_id], axis=0).reshape(1, -1)
            scaled = pkg['scaler'].transform(avg)
            
            # Clean results (N, P, K...)
            results = {clean: round(max(0.0, float(pkg['models'][raw].predict(scaled)[0])), 3) 
                       for raw, clean in TARGET_MAP.items()}
            
            device_data[device_id] = [] # Garbage Collection
            return {"status": "success", "device": device_id, "prediction": results}

        return {"status": "stored", "device": device_id, "current_count": len(device_data[device_id])}

    except Exception as e:
        return {"status": "error", "message": "Invalid format. Use ?data=1,2,3..."}

# --- 4. JSON ERROR HANDLER (Replaces "Not Found" with JSON) ---
@app.exception_handler(404)
async def custom_404(request: Request, __):
    return JSONResponse(status_code=404, content={"status": "error", "message": "Invalid Endpoint"})