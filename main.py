import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor 
from sklearn.preprocessing import StandardScaler
import pickle
import os
import asyncio
import websockets
import json
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# --- CONFIGURATION ---
WS_URL = "wss://nikolaindustry-realtime.onrender.com/?id=905cd742-d1c3-4fdb-b940-6cac16faa792"
CSV_FILE = 'final_training_base.csv' 
MODEL_PACK = 'soil_model_pack_rf.pkl'
SCAN_COUNT = 10 

app = FastAPI()

# Enable CORS so any website can enquire your data
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

latest_predictions = {}

def build_model():
    if not os.path.exists(MODEL_PACK):
        df = pd.read_csv(CSV_FILE, encoding='latin1')
        df.columns = df.columns.str.strip()
        x_features = [f'X_{i}' for i in range(1, 19)]
        targets = ['N(ppm)', 'P(ppm)', 'K(ppm)', 'OC_percent', 'pH', 'EC', 'Fe', 'Mn', 'Cu', 'Zn', 'B', 'S']
        scaler = StandardScaler().fit(df[x_features].values)
        X_train_scaled = scaler.transform(df[x_features].values)
        trained_models = {t: RandomForestRegressor(n_estimators=100).fit(X_train_scaled, df[t].values) for t in targets}
        with open(MODEL_PACK, 'wb') as f:
            pickle.dump({'scaler': scaler, 'models': trained_models, 'targets': targets}, f)

# Initial Load
build_model()
with open(MODEL_PACK, 'rb') as f:
    pkg = pickle.load(f)

async def websocket_worker():
    global latest_predictions
    buffer = []
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                while True:
                    msg = await ws.recv()
                    data = json.loads(msg)
                    current_scan = [data.get(f'X_{i}', 0) for i in range(1, 19)]
                    buffer.append(current_scan)
                    
                    if len(buffer) >= SCAN_COUNT:
                        avg_raw = np.mean(buffer, axis=0).reshape(1, -1)
                        scaled = pkg['scaler'].transform(avg_raw)
                        latest_predictions = {
                            t: round(max(0.0, float(pkg['models'][t].predict(scaled)[0])), 3) 
                            for t in pkg['targets']
                        }
                        buffer = []
        except Exception as e:
            print(f"WS Error: {e}. Reconnecting in 5s...")
            await asyncio.sleep(5)

@app.on_event("startup")
async def startup():
    asyncio.create_task(websocket_worker())

@app.get("/predict")
async def get_prediction():
    """Endpoint for external enquiries to get JSON data"""
    if not latest_predictions:
        return {"status": "processing", "message": "Collecting 10 scans..."}
    return {"status": "success", "data": latest_predictions}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)