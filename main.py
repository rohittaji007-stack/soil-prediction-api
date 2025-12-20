from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import numpy as np
import pandas as pd
import pickle
import os
from typing import Dict, List

app = FastAPI()

# --- SETTINGS ---
CSV_FILE = 'final_training_base.csv'
MODEL_PACK = 'soil_model_pack_rf.pkl'
TARGET_MAP = {
    'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K', 
    'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC', 
    'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S'
}

# Multi-tenant memory for 10+ devices
device_data: Dict[str, List[List[float]]] = {}

# --- AUTO-TRAIN LOGIC ---
if not os.path.exists(MODEL_PACK) and os.path.exists(CSV_FILE):
    df = pd.read_csv(CSV_FILE, encoding='latin1')
    df.columns = df.columns.str.strip()
    x_features = [f'X_{i}' for i in range(1, 19)]
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler().fit(df[x_features].values)
    models = {t: RandomForestRegressor(n_estimators=100).fit(scaler.transform(df[x_features].values), df[t].values) for t in TARGET_MAP.keys()}
    with open(MODEL_PACK, 'wb') as f:
        pickle.dump({'scaler': scaler, 'models': models, 'targets': list(TARGET_MAP.keys())}, f)

# --- ROUTES ---

@app.get("/", response_class=HTMLResponse)
async def home():
    return "<html><body style='text-align:center;padding-top:50px;'><h1>🌱 Soil API: Online</h1><p>Status: Connected</p></body></html>"

@app.get("/predict/{device_id}")
async def predict(device_id: str, data: str = None):
    # 1. Load the brain
    if not os.path.exists(MODEL_PACK):
        return {"status": "error", "message": "Model not built. Upload final_training_base.csv"}
    
    with open(MODEL_PACK, 'rb') as f:
        pkg = pickle.load(f)

    # 2. Check if data is being sent
    if not data:
        count = len(device_data.get(device_id, []))
        return {"status": "connected", "device": device_id, "scans": f"{count}/10"}

    # 3. Process the incoming scan
    try:
        vals = [float(x) for x in data.split(',')]
        if len(vals) != 18:
            return {"status": "error", "message": "Need exactly 18 values"}

        if device_id not in device_data:
            device_data[device_id] = []
        
        device_data[device_id].append(vals)

        # 4. If 10 scans reached, predict and clear memory (Garbage Collection)
        if len(device_data[device_id]) >= 10:
            avg = np.mean(device_data[device_id], axis=0).reshape(1, -1)
            scaled = pkg['scaler'].transform(avg)
            
            prediction = {clean: round(max(0.0, float(pkg['models'][raw].predict(scaled)[0])), 3) 
                          for raw, clean in TARGET_MAP.items()}
            
            device_data[device_id] = [] # Memory wipe
            return {"status": "success", "device": device_id, "prediction": prediction}

        return {"status": "stored", "device": device_id, "count": len(device_data[device_id])}

    except Exception as e:
        return {"status": "error", "message": str(e)}