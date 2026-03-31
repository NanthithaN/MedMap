import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml_model import predictor, DATASET_PATH
from optimizer import optimize_hospitals

app = FastAPI(title="MedMap API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizeRequest(BaseModel):
    selected_areas: list[str]
    predicted_demands: dict[str, int]
    capacity: float

class PredictRequest(BaseModel):
    selected_areas: list[str]

@app.get("/areas")
def get_areas():
    try:
        df = pd.read_csv(DATASET_PATH)
        return {"areas": df['area_name'].tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train-model")
def train_model():
    result = predictor.train()
    if result["status"] == "error":
        raise HTTPException(status_code=500, detail=result["message"])
    return result

@app.post("/predict-demand")
def predict_demand(req: PredictRequest):
    result = predictor.predict(req.selected_areas)
    if result["status"] == "error":
        raise HTTPException(status_code=400, detail=result["message"])
    return result

@app.post("/optimize")
def optimize(req: OptimizeRequest):
    if len(req.selected_areas) == 0:
        raise HTTPException(status_code=400, detail="No selected areas")
        
    try:
        df = pd.read_csv(DATASET_PATH)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load dataset: {e}")
        
    filtered_df = df[df['area_name'].isin(req.selected_areas)].copy()
    if filtered_df.empty:
        raise HTTPException(status_code=404, detail="Selected areas not found in dataset")
        
    # Ensure standard ordering
    filtered_df = filtered_df.set_index('area_name').loc[req.selected_areas].reset_index()
    
    distance_cols = [f'distance_hospital_{k}' for k in range(1, 11)]
    
    selected_areas_info = []
    for _, row in filtered_df.iterrows():
        area_name = row['area_name']
        demand = req.predicted_demands.get(area_name, 0)
        
        # Extract distances safely
        distances = []
        for col in distance_cols:
            if col in row:
                distances.append(float(row[col]))
            else:
                distances.append(0.0)
                
        selected_areas_info.append({
            "name": area_name,
            "demand": demand,
            "distances": distances
        })
        
    result = optimize_hospitals(selected_areas_info, req.capacity)
    
    if result["status"] == "infeasible":
        raise HTTPException(status_code=400, detail=result["message"])
        
    return result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
