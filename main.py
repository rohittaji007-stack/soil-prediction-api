from fastapi import FastAPI, HTTPException, Query
import numpy as np
import pandas as pd
import pickle
import os
import asyncio
import websockets
import json
from typing import Dict, List

app = FastAPI(title="Soil Intelligence Pro API")

# --- 4. STRONG ENVIRONMENT: Isolated Storage ---
# Stores raw scans for each device: { "device_id": [[18_vals], [18_vals], ...] }
device_data: Dict[str, List[List[float]]] = {}
# Stores final results for each device: { "device_id": { "pH": 7.0, ... } }
device_results: Dict[str, Dict[str, float]] = {}

MODEL_PACK = 'soil_model_pack_rf.pkl'
with open(MODEL_PACK, 'rb') as f:
    pkg = pickle.load(f)

# --- 1 & 7. DELETE LOGIC (For Hypervisor) ---
@app.delete("/scans/{device_id}")
async def delete_scan(device_id: str, index: int = None, clear_all: bool = False):
    """Allows deleting a specific scan index (e.g., #5) or clearing all data."""
    if device_id not in device_data:
        raise HTTPException(status_code=404, detail="Device not found")
    
    if clear_all:
        device_data[device_id] = []
        return {"status": "success", "message": f"Cleared all scans for {device_id}"}
    
    if index is not None:
        try:
            # index-1 handles the 'Scan 5' logic (0-based indexing)
            device_data[device_id].pop(index - 1)
            return {"status": "success", "message": f"Deleted scan {index}"}
        except IndexError:
            raise HTTPException(status_code=400, detail="Invalid scan index")

# --- 3 & 5. NEW REQUEST PATTERN (Server Host) ---
@app.post("/push/{device_id}")
async def push_data(device_id: str, sensor_values: List[float]):
    """Receives data directly from a device or Hypervisor."""
    if device_id not in device_data:
        device_data[device_id] = []
    
    device_data[device_id].append(sensor_values)
    
    # Process when 10 scans are reached for THIS specific device
    if len(device_data[device_id]) >= 10:
        avg_vals = np.mean(device_data[device_id], axis=0).reshape(1, -1)
        scaled = pkg['scaler'].transform(avg_vals)
        
        # Calculate Random Forest predictions
        preds = {t: round(max(0.0, float(pkg['models'][t].predict(scaled)[0])), 3) 
                 for t in pkg['targets']}
        
        device_results[device_id] = preds # Store result in device's private slot
        device_data[device_id] = [] # Reset buffer for this device only
        return {"status": "calculated", "data": preds}
    
    return {"status": "stored", "count": len(device_data[device_id])}

@app.get("/predict/{device_id}")
async def get_prediction(device_id: str):
    """Endpoint for Hypervisor to call results."""
    if device_id not in device_results:
        return {"status": "pending", "message": "Need more scans for this device"}
    return {"status": "success", "device": device_id, "data": device_results[device_id]}