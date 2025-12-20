from fastapi import FastAPI
from fastapi.responses import HTMLResponse

# Create the app instance
app = FastAPI()

# 1. THE ROOT ROUTE - This MUST show up at your main URL
@app.get("/", response_class=HTMLResponse)
async def root():
    return """
    <html>
        <body style='font-family:sans-serif; text-align:center; padding-top:50px;'>
            <h1 style='color:green;'>✅ Soil API is ONLINE</h1>
            <p>Your server is successfully connected to GitHub.</p>
            <p>Try this link to test: <a href='/predict/test_device'>/predict/test_device</a></p>
        </body>
    </html>
    """

# 2. THE PREDICT ROUTE
@app.get("/predict/{device_id}")
async def predict(device_id: str, data: str = None):
    if not data:
        return {"status": "connected", "device": device_id, "message": "Waiting for data..."}
    
    # Logic for processing 18 values
    return {"status": "received", "device": device_id, "data_received": data}