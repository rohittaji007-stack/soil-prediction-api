from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler
import pickle
import os
from typing import Dict, List

app = FastAPI(title="Soil Intelligence Pro API")

# --- CONFIGURATION ---
CSV_FILE = 'final_training_base.csv' 
MODEL_PACK = 'soil_model_pack_rf.pkl'

# --- 4. PRODUCTION STORAGE ---
device_data: Dict[str, List[List[float]]] = {}
device_results: Dict[str, Dict[str, float]] = {}

# --- 2. AUTO-TRAIN BRAIN ---
def build_model():
    if not os.path.exists(MODEL_PACK):
        df = pd.read_csv(CSV_FILE, encoding='latin1')
        df.columns = df.columns.str.strip()
        x_features = [f'X_{i}' for i in range(1, 19)]
        targets = ['N(ppm)', 'P(ppm)', 'K(ppm)', 'OC_percent', 'pH', 'EC', 'Fe', 'Mn', 'Cu', 'Zn', 'B', 'S']
        scaler = StandardScaler().fit(df[x_features].values)
        X_scaled = scaler.transform(df[x_features].values)
        models = {t: RandomForestRegressor(n_estimators=100).fit(X_scaled, df[t].values) for t in targets}
        with open(MODEL_PACK, 'wb') as f:
            pickle.dump({'scaler': scaler, 'models': models, 'targets': targets}, f)

build_model()
with open(MODEL_PACK, 'rb') as f:
    pkg = pickle.load(f)

# --- 5. HOME PAGE (Avoids 404) ---
@app.get("/", response_class=HTMLResponse)
async def home_page():
    return """
    <body style='font-family: Arial; text-align: center; margin-top: 100px;'>
        <h1 style='color: #2E7D32;'>Soil API is Operational</h1>
        <p>To view data: <b>/predict/Device_ID</b></p>
        <p>To test endpoints: <a href='/docs'>Visit /docs</a></p>
    </body>
    """

# --- 1 & 7. MANAGE SCANS (DELETE) ---
@app.delete("/scans/{device_id}")
async def delete_scan(device_id: str, index: int = None):
    if device_id in device_data and index and 0 < index <= len(device_data[device_id]):
        device_data[device_id].pop(index - 1)
        return {"status": "success", "message": f"Scan {index} deleted"}
    raise HTTPException(status_code=404, detail="Scan or device not found")

# --- 3 & 5. PREDICTION ENQUIRY (GET) ---
@app.get("/predict/{device_id}")
async def get_results(device_id: str):
    if device_id not in device_results:
        count = len(device_data.get(device_id, []))
        return {"status": "pending", "scans_collected": f"{count}/10"}
    return {"status": "success", "device": device_id, "data": device_results[device_id]}

# --- PUSH DATA (POST) ---
@app.post("/push/{device_id}")
async def push_data(device_id: str, sensor_values: List[float]):
    if device_id not in device_data: device_data[device_id] = []
    device_data[device_id].append(sensor_values)
    if len(device_data[device_id]) >= 10:
        avg = np.mean(device_data[device_id], axis=0).reshape(1, -1)
        scaled = pkg['scaler'].transform(avg)
        preds = {t: round(max(0.0, float(pkg['models'][t].predict(scaled)[0])), 3) for t in pkg['targets']}
        device_results[device_id] = preds
        device_data[device_id] = [] # Reset after 10
        return {"status": "calculated", "data": preds}
    return {"status": "stored", "count": len(device_data[device_id])}