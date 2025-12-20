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

# Clean keys for output as requested
TARGET_MAP = {
    'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K', 
    'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC', 
    'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S'
}

# --- 2. MULTI-TENANT STORAGE ---
# Isolated memory for 10+ devices
device_data: Dict[str, List[List[float]]] = {}

# --- 3. AUTO-TRAIN BRAIN (Self-Healing) ---
# This fixes the FileNotFoundError seen in your logs
def build_model_if_missing():
    if not os.path.exists(MODEL_PACK):
        print(f"🧠 Training brain from {CSV_FILE}...")
        if not os.path.exists(CSV_FILE):
            return # Wait for user to upload CSV
        
        df = pd.read_csv(CSV_FILE, encoding='latin1')
        df.columns = df.columns.str.strip()
        x_features = [f'X_{i}' for i in range(1, 19)]
        targets = list(TARGET_MAP.keys())
        
        scaler = StandardScaler().fit(df[x_features].values)
        X_scaled = scaler.transform(df[x_features].values)
        
        models = {}
        for t in targets:
            m = RandomForestRegressor(n_estimators=100, random_state=42)
            m.fit(X_scaled, df[t].values)
            models[t] = m
            
        with open(MODEL_PACK, 'wb') as f:
            pickle.dump({'scaler': scaler, 'models': models, 'targets': targets}, f)

build_model_if_missing()

# Helper to load the model safely
def get_brain():
    if os.path.exists(MODEL_PACK):
        with open(MODEL_PACK, 'rb') as f:
            return pickle.load(f)
    return None

# --- 4. JSON ERROR HANDLER ---
# Returns JSON instead of HTML for 404 errors
@app.exception_handler(404)
async def custom_404_handler(request: Request, __):
    return JSONResponse(
        status_code=404,
        content={"status": "error", "message": "Invalid URL. Use /predict/Device_ID?data=values"}
    )

# --- 5. HOME PAGE ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <body style='font-family: Arial; text-align: center; margin-top: 100px;'>
        <h1 style='color: #2E7D32;'>🌱 Soil API is Live</h1>
        <p>Device Status: Operational</p>
        <p>Docs: <a href='/docs'>/docs</a></p>
    </body>
    """

# --- 6. UNIFIED PREDICT (Stable Query Pattern) ---
# This single endpoint handles Push, Store, and Predict
@app.get("/predict/{device_id}")
async def predict(device_id: str, data: str = None):
    pkg = get_brain()
    if not pkg:
        return {"status": "error", "message": "Model brain not built. Check CSV file."}

    # If no data, just show count (Garbage collection check)
    if not data:
        count = len(device_data.get(device_id, []))
        return {"status": "pending", "device": device_id, "scans_collected": f"{count}/10"}

    try:
        # Convert data string to list
        vals = [float(x) for x in data.split(',')]
        if len(vals) != 18:
            return {"status": "error", "message": "Need exactly 18 values"}

        if device_id not in device_data:
            device_data[device_id] = []
        
        device_data[device_id].append(vals)

        # Process at 10 scans
        if len(device_data[device_id]) >= 10:
            avg = np.mean(device_data[device_id], axis=0).reshape(1, -1)
            scaled = pkg['scaler'].transform(avg)
            
            # Clean Key-Value Output
            prediction = {}
            for raw_key, clean_key in TARGET_MAP.items():
                val = float(pkg['models'][raw_key].predict(scaled)[0])
                prediction[clean_key] = round(max(0.0, val), 3)
            
            # Garbage Collection: Reset memory for this ID
            device_data[device_id] = []
            return {"status": "success", "device": device_id, "prediction": prediction}

        return {"status": "stored", "device": device_id, "current_count": len(device_data[device_id])}

    except Exception as e:
        return {"status": "error", "message": "Data must be 18 numbers separated by commas"}