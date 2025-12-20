from fastapi import FastAPI, HTTPException, Query
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler
import pickle
import os
import asyncio
import websockets
import json
from typing import Dict, List

app = FastAPI(title="Soil Intelligence Pro API")

# --- CONFIGURATION ---
CSV_FILE = 'final_training_base.csv' 
MODEL_PACK = 'soil_model_pack_rf.pkl'
WS_URL = "wss://nikolaindustry-realtime.onrender.com/?id=905cd742-d1c3-4fdb-b940-6cac16faa792"
SCAN_COUNT = 10 

# Storage for multiple sensors
device_data: Dict[str, List[List[float]]] = {}
device_results: Dict[str, Dict[str, float]] = {}

# --- AUTO-TRAIN LOGIC ---
def build_model_if_missing():
    """Automatically builds the Random Forest brain on the Render server."""
    if not os.path.exists(MODEL_PACK):
        print(f"🧠 Brain missing. Training now from {CSV_FILE}...")
        if not os.path.exists(CSV_FILE):
            raise FileNotFoundError(f"❌ Critical Error: {CSV_FILE} is missing from GitHub!")
            
        df = pd.read_csv(CSV_FILE, encoding='latin1')
        df.columns = df.columns.str.strip()
        x_features = [f'X_{i}' for i in range(1, 19)]
        targets = ['N(ppm)', 'P(ppm)', 'K(ppm)', 'OC_percent', 'pH', 'EC', 'Fe', 'Mn', 'Cu', 'Zn', 'B', 'S']
        
        scaler = StandardScaler().fit(df[x_features].values)
        X_train_scaled = scaler.transform(df[x_features].values)
        
        trained_models = {}
        for t in targets:
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train_scaled, df[t].values)
            trained_models[t] = model
            
        with open(MODEL_PACK, 'wb') as f:
            pickle.dump({'scaler': scaler, 'models': trained_models, 'targets': targets}, f)
        print("✅ Brain built successfully on Render!")

# Build and then Load
build_model_if_missing()
with open(MODEL_PACK, 'rb') as f:
    pkg = pickle.load(f)

# --- ENDPOINTS ---
@app.delete("/scans/{device_id}")
async def delete_scan(device_id: str, index: int = None, clear_all: bool = False):
    if device_id not in device_data:
        raise HTTPException(status_code=404, detail="Device not found")
    if clear_all:
        device_data[device_id] = []
        return {"status": "success", "message": "Cleared all scans"}
    if index is not None:
        device_data[device_id].pop(index - 1)
        return {"status": "success", "message": f"Deleted scan {index}"}

@app.post("/push/{device_id}")
async def push_data(device_id: str, sensor_values: List[float]):
    if device_id not in device_data:
        device_data[device_id] = []
    device_data[device_id].append(sensor_values)
    if len(device_data[device_id]) >= SCAN_COUNT:
        avg_vals = np.mean(device_data[device_id], axis=0).reshape(1, -1)
        scaled = pkg['scaler'].transform(avg_vals)
        preds = {t: round(max(0.0, float(pkg['models'][t].predict(scaled)[0])), 3) 
                 for t in pkg['targets']}
        device_results[device_id] = preds
        device_data[device_id] = []
        return {"status": "calculated", "data": preds}
    return {"status": "stored", "count": len(device_data[device_id])}

@app.get("/predict/{device_id}")
async def get_prediction(device_id: str):
    if device_id not in device_results:
        return {"status": "pending", "message": f"Collected {len(device_data.get(device_id, []))}/10 scans"}
    return {"status": "success", "data": device_results[device_id]}