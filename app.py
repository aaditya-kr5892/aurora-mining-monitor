import uvicorn
from fastapi import FastAPI, BackgroundTasks, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Any, Dict, List, Optional
import os

# Import your pipeline logic
# Ensure pipeline.py is in the same folder as app.py
from pipeline import run_pipeline, PIPELINE_PROGRESS

app = FastAPI(title="Mining Detection API")

# 1. Allow Frontend to Connect (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Serve the generated images
os.makedirs("outputs", exist_ok=True)
app.mount("/static", StaticFiles(directory="outputs"), name="static")

@app.get("/")
def health_check():
    return {"status": "ok", "message": "Mining Detection API is running"}

# --- Data Models ---
class PipelineRequest(BaseModel):
    mine_geojson: Dict[str, Any]
    no_go_geojson_list: Optional[List[Dict[str, Any]]] = None

# --- Global Result Store ---
ANALYSIS_RESULT = {}

def background_wrapper(mine_data, nogo_data):
    """Wrapper to capture the return value of your pipeline"""
    global ANALYSIS_RESULT
    try:
        result = run_pipeline(mine_data, nogo_data)
        ANALYSIS_RESULT = result
    except Exception as e:
        print(f"Background Task Error: {e}")
        # Error handling is managed inside pipeline.py's global PIPELINE_PROGRESS variable

@app.post("/run", status_code=status.HTTP_202_ACCEPTED)
def run(req: PipelineRequest, background_tasks: BackgroundTasks):
    global ANALYSIS_RESULT

    # Prevent overlapping analysis runs
    if PIPELINE_PROGRESS.get("status") == "running":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An analysis is currently in progress. Please wait for it to complete."
        )

    ANALYSIS_RESULT = {} # Clear previous results
    
    # Extract dicts directly without Pydantic deprecation warnings
    mine_dict = req.mine_geojson
    nogo_list = req.no_go_geojson_list if req.no_go_geojson_list else None
    
    # Run in background so we don't block the progress bar
    background_tasks.add_task(background_wrapper, mine_dict, nogo_list)
    
    return {"status": "started", "message": "Analysis successfully accepted and started"}

@app.get("/progress")
def get_progress():
    return PIPELINE_PROGRESS

@app.get("/results")
def get_results():
    return ANALYSIS_RESULT

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)