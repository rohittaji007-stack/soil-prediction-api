from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, List

app = FastAPI(title="Soil Intelligence Pro API")

# --- 1. CONFIGURATION & MAPPING ---
CSV_FILE = 'final_training_base.csv'
MODEL_PACK = 'soil_model_pack_rf.pkl'

# Clean keys as requested
TARGET_MAP = {
    'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K', 
    'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC', 
    'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S'
}

# Isolated memory for 10+ devices
device_data: Dict[str, List[List[float]]] = {}

# --- 2. AUTO-TRAIN LOGIC (Self-Healing) ---
def build_model_if_missing():
    if not os.path.exists(MODEL_PACK):
        if not os.path.exists(CSV_FILE): return
        df = pd.read_csv(CSV_FILE, encoding='latin1')
        df.columns = df.columns.str.strip()
        x_features = [f'X_{i}' for i in range(1, 19)]
        scaler = StandardScaler().fit(df[x_features].values)
        X_scaled = scaler.transform(df[x_features].values)
        models = {t: RandomForestRegressor(n_estimators=100).fit(X_scaled, df[t].values) for t in TARGET_MAP.keys()}
        with open(MODEL_PACK, 'wb') as f:
            pickle.dump({'scaler': scaler, 'models': models, 'targets': list(TARGET_MAP.keys())}, f)

build_model_if_missing()

def get_brain():
    if os.path.exists(MODEL_PACK):
        with open(MODEL_PACK, 'rb') as f: return pickle.load(f)
    return None

# --- 3. JSON ERROR HANDLER (Fixes HTML 404s) ---
@app.exception_handler(404)
async def custom_404_handler(request: Request, __):
    return JSONResponse(status_code=404, content={"status": "error", "message": "Invalid URL. Use /predict/ID?data=1,2,3..."})

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<body style='font-family:Arial;text-align:center;margin-top:100px;'><h1>🌱 Soil API Live</h1><p><a href='/docs'>Visit /docs to test</a></p></body>"

# --- 4. STABLE UNIFIED PREDICT (The Fix) ---
@app.get("/predict/{device_id}")
async def predict(device_id: str, data: str = None):
    pkg = get_brain()
    if not pkg: return {"status": "error", "message": "Brain missing. Upload CSV."}

    # If no data is sent, show current buffer status
    if not data:
        count = len(device_data.get(device_id, []))
        return {"status": "pending", "device": device_id, "scans": f"{count}/10"}

    try:
        # Convert comma string to list
        vals = [float(x) for x in data.split(',')]
        if len(vals) != 18: return {"status": "error", "message": "Need 18 values"}

        if device_id not in device_data: device_data[device_id] = []
        device_data[device_id].append(vals)

        if len(device_data[device_id]) >= 10:
            avg = np.mean(device_data[device_id], axis=0).reshape(1, -1)
            scaled = pkg['scaler'].transform(avg)
            prediction = {clean: round(max(0.0, float(pkg['models'][raw].predict(scaled)[0])), 3) for raw, clean in TARGET_MAP.items()}
            device_data[device_id] = [] # Garbage Collection
            return {"status": "success", "device": device_id, "prediction": prediction}

        return {"status": "stored", "device": device_id, "count": len(device_data[device_id])}
    except:
        return {"status": "error", "message": "Use numbers separated by commas"}