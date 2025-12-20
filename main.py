from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List

app = FastAPI(title="Soil Intelligence Pro API")

# --- CONFIGURATION ---
CSV_FILE = 'final_training_base.csv'
MODEL_PACK = 'soil_model_pack_rf.pkl'

# Memory Storage: { "device_id": [[scans]] }
device_data: Dict[str, List[List[float]]] = {}

# --- 4. CLEAN PARAMETERS (Standard Format) ---
TARGET_MAP = {
    'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K', 
    'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC', 
    'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S'
}

# Load/Build Brain
if not os.path.exists(MODEL_PACK):
    df = pd.read_csv(CSV_FILE, encoding='latin1')
    df.columns = df.columns.str.strip()
    x_features = [f'X_{i}' for i in range(1, 19)]
    targets = list(TARGET_MAP.keys())
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(df[x_features].values)
    models = {t: RandomForestRegressor(n_estimators=100).fit(scaler.transform(df[x_features].values), df[t].values) for t in targets}
    with open(MODEL_PACK, 'wb') as f:
        pickle.dump({'scaler': scaler, 'models': models, 'targets': targets}, f)

with open(MODEL_PACK, 'rb') as f:
    pkg = pickle.load(f)

# --- 3. JSON ERROR HANDLING (No more HTML 404s) ---
@app.exception_handler(404)
async def custom_404_handler(request: Request, __):
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Endpoint not found. Use /predict/{id}/{data}"}
    )

# --- 1, 2 & 5. THE CONSOLIDATED GET PREDICT ---
@app.get("/predict/{device_id}/{data_string}")
async def predict(device_id: str, data_string: str):
    """
    Example: /predict/DEV01/1.2,3.4,5.6... (18 values)
    This single call Pushes, Manages, and Returns predictions.
    """
    try:
        # 1. Parse Input
        vals = [float(x) for x in data_string.split(',')]
        if len(vals) != 18:
            return {"status": "error", "message": "Provide exactly 18 comma-separated values"}

        # 2. Add to Device Memory (Isolated by ID)
        if device_id not in device_data:
            device_data[device_id] = []
        device_data[device_id].append(vals)

        # 3. Process Logic
        if len(device_data[device_id]) >= 10:
            avg = np.mean(device_data[device_id], axis=0).reshape(1, -1)
            scaled = pkg['scaler'].transform(avg)
            
            # Create Clean Key-Value Output (N: 10, P: 5, etc.)
            output = {}
            for raw_key, clean_key in TARGET_MAP.items():
                val = float(pkg['models'][raw_key].predict(scaled)[0])
                output[clean_key] = round(max(0.0, val), 3)
            
            # Garbage Collection: Clear memory for this device after prediction
            device_data[device_id] = []
            
            return {"status": "success", "device": device_id, "prediction": output}

        # Return status if 10 scans not reached
        return {
            "status": "stored", 
            "device": device_id, 
            "current_count": len(device_data[device_id]),
            "needed": 10
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}