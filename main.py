from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import numpy as np
import pandas as pd
import pickle
import os

app = FastAPI()

# --- CONFIGURATION ---
CSV_FILE = 'final_training_base.csv'
MODEL_PACK = 'soil_model_pack_rf.pkl'
TARGET_MAP = {
    'N(ppm)': 'N', 'P(ppm)': 'P', 'K(ppm)': 'K', 
    'OC_percent': 'OC', 'pH': 'PH', 'EC': 'EC', 
    'Fe': 'FE', 'Mn': 'MN', 'Cu': 'CU', 'Zn': 'ZN', 'B': 'B', 'S': 'S'
}

# --- AUTO-TRAIN LOGIC (Runs once to create the brain) ---
if not os.path.exists(MODEL_PACK) and os.path.exists(CSV_FILE):
    try:
        df = pd.read_csv(CSV_FILE, encoding='latin1')
        df.columns = df.columns.str.strip()
        x_features = [f'X_{i}' for i in range(1, 19)]
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(df[x_features].values)
        models = {t: RandomForestRegressor(n_estimators=100).fit(scaler.transform(df[x_features].values), df[t].values) for t in TARGET_MAP.keys()}
        with open(MODEL_PACK, 'wb') as f:
            pickle.dump({'scaler': scaler, 'models': models, 'targets': list(TARGET_MAP.keys())}, f)
    except Exception as e:
        print(f"Error building model: {e}")

# --- THE UPDATED POST ENDPOINT ---
@app.post("/predict_batch/{device_id}")
async def predict_batch(device_id: str, request: Request):
    """
    PASTE YOUR DATA HERE: 
    Click 'Try it out' to see the white 'Request body' box below.
    Paste the list of 10 scans [[18 values], ... x10] into that box.
    """
    try:
        # This line catches the 'Package' from the Request Body box
        all_scans = await request.json() 
        
        # Validation
        if not isinstance(all_scans, list) or len(all_scans) != 10:
            return JSONResponse(status_code=400, content={"error": f"Expected 10 scans, got {len(all_scans) if isinstance(all_scans, list) else 'invalid format'}"})

        if not os.path.exists(MODEL_PACK):
            return JSONResponse(status_code=500, content={"error": "Model not found. Ensure CSV is uploaded."})

        with open(MODEL_PACK, 'rb') as f:
            pkg = pickle.load(f)

        # Process each of the 10 scans individually (Your Logic)
        batch_results = []
        for scan in all_scans:
            input_data = np.array(scan).reshape(1, -1)
            scaled_data = pkg['scaler'].transform(input_data)
            
            single_prediction = {}
            for raw_key, clean_key in TARGET_MAP.items():
                val = float(pkg['models'][raw_key].predict(scaled_data)[0])
                single_prediction[clean_key] = round(max(0.0, val), 3)
            
            batch_results.append(single_prediction)

        return {
            "status": "success",
            "device": device_id,
            "results": batch_results
        }

    except Exception as e:
        return JSONResponse(status_code=500, content={"status": "error", "message": str(e)})