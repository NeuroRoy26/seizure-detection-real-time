from fastapi import FastAPI
import numpy as np
import random

app = FastAPI()

latest_data = None

@app.post("/ingest")
async def ingest(data: list):
    global latest_data
    latest_data = np.array(data)
    return {"message": "Data ingested successfully"}

@app.get("/latest")
async def get_latest():
    global latest_data
    if latest_data is None:
        return {"message": "No data available"}
    seizure_probability = random.uniform(0.0, 1.0)
    return {"data": latest_data.tolist(), "seizure_probability": seizure_probability}
