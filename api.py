from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import mlflow
import pickle
import torch
import pandas as pd
import numpy as np

app = FastAPI()


class Features(BaseModel):
    features: List[float]  #['footfall', 'tempMode', 'AQ', 'USS', 'CS', 'VOC', 'RP', 'IP','Temperature']

# file:///D:/project/machine_failuire/mlruns/601922764089210881/bf16499b420d4287a7961c49264e994f/artifacts/data/model.pth
model_path = r"d:\project\machine_failuire\mlruns\601922764089210881\bf16499b420d4287a7961c49264e994f/artifacts/model.pth"
model = torch.load(model_path,weights_only=False)
with open("data/scaler.pkl","rb") as f :
    scaler = pickle.load(f)

@app.get('/')
def hello():
    return {"message": "Hello from FastAPI with Machine failuire!"}

@app.post('/predict/')
async def predict(features: Features):

    input_data = np.array([features.features], dtype=np.float32) 
    input_scaled = scaler.transform(input_data) 
    with torch.no_grad():
        input_tensor = torch.tensor(input_scaled, dtype=torch.float32)
        pred = model(input_tensor).numpy()  
   
    return {"prediction": float(pred[0][0])}
 