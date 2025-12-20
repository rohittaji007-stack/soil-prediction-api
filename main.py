from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
import os

app = FastAPI()

# --- 1. THE FRONT DOOR ---
@app.get("/", response_class=HTMLResponse)
async def home():
    return """
    <html>
        <body style='font-family:sans-serif; text-align:center; padding-top:50px;'>
            <h1 style='color:green;'>✅ Soil API is ONLINE</h1>
            <p>The server is successfully running.</p>
            <p><b>Test Status Link:</b> <a href='/predict/Device_01'>Click here to check Device_01</a></p>
        </body>
    </html>
    """

# --- 2. THE PREDICT LINK ---
@app.get("/predict/{device_id}")
async def predict(device_id: str, data: str = None):
    # This part tells you exactly what the server sees
    if not data:
        return {
            "status": "connected",
            "device": device_id,
            "message": "I am ready. Please send data using ?data=v1,v2...",
            "example": f"/predict/{device_id}?data=1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18"
        }
    
    # Simple check to prove data is arriving
    return {
        "status": "data_received",
        "device": device_id,
        "received_values": data
    }

# --- 3. CATCH-ALL ERROR HANDLER ---
@app.exception_handler(404)
async def custom_404(request: Request, __):
    return JSONResponse(
        status_code=404,
        content={"error": "Not Found", "hint": "Go to the home page to see correct links."}
    )