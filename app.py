import os
import sys
import time
import re
import pickle
from pathlib import Path
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, Response
from fastapi.templating import Jinja2Templates
from prometheus_client import Counter, Histogram, CollectorRegistry, generate_latest, CONTENT_TYPE_LATEST

from src.data_processing.data_preprocessing import TextPreprocessor 

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

ml_objects: dict = {}

SENTIMENT_MAP = { 0: "Negative", 1: "Positive" }

# Safety Overrides
NEGATIVE_TRIGGERS = [
    "waste of time", "worst", "awful", "terrible", "boring", "useless", 
    "don't watch", "skip it", "bad movie", "garbage", "poorly", "disaster"
]

POSITIVE_TRIGGERS = [
    "must watch", "loved it", "great movie", "masterpiece", "excellent",
    "highly recommend", "amazing", "best movie", "watch again", "watch it again"
]

TEMPLATES_DIR = BASE_DIR / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

# ---------------------------------------------------------------------
# Logic Helpers
# ---------------------------------------------------------------------

def check_smart_negation(text: str) -> bool:
    """Detects 'not ... recommend' patterns."""
    text = text.lower()
    pattern = r"(not|no|never|n't)\s+(.{0,35}\s+)?(recommend|like|good|worth|watch)"
    return bool(re.search(pattern, text))

def load_local_artifacts() -> None:
    try:
        # Check if model files exist
        vectorizer_path = MODEL_DIR / "vectorizer.pkl"
        model_path = MODEL_DIR / "model.pkl"

        if not vectorizer_path.exists() or not model_path.exists():
            print(f"❌ Artifacts missing in {MODEL_DIR}")
            return

        ml_objects["preprocessor"] = TextPreprocessor()
        
        with open(vectorizer_path, "rb") as file:
            ml_objects["vectorizer"] = pickle.load(file)
            
        with open(model_path, "rb") as file:
            ml_objects["model"] = pickle.load(file)
            
        print("✅ Artifacts loaded successfully")
    except Exception as e:
        print(f"❌ Error loading artifacts: {e}")

# ---------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------

# CRITICAL FIX: Add root_path parameter for NGINX proxy handling
app = FastAPI(
    title="Sentiment Analysis Pro",
    root_path="/Sentiment-Analysis"
)

registry = CollectorRegistry()
REQUEST_COUNT = Counter("app_request_count", "Total requests", ["method", "endpoint"], registry=registry)

@app.on_event("startup")
def startup_event():
    load_local_artifacts()

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, text: str = Form(...)):
    start_time = time.time()
    try:
        if "preprocessor" not in ml_objects:
            load_local_artifacts()
            if "preprocessor" not in ml_objects:
                raise Exception("Model artifacts not loaded.")

        # 1. Preprocess
        preprocessor = ml_objects["preprocessor"]
        clean_text = preprocessor._preprocess_text(text)

        # 2. Hybrid Logic (Rule Override)
        text_lower = text.lower()
        rule_prediction = None
        
        # Negative Check
        if check_smart_negation(text_lower):
            rule_prediction = 0
        else:
            for trig in NEGATIVE_TRIGGERS:
                if trig in text_lower:
                    rule_prediction = 0
                    break
        
        # Positive Check (Only if not already Negative)
        if rule_prediction is None:
            for trig in POSITIVE_TRIGGERS:
                if trig in text_lower:
                    rule_prediction = 1
                    break

        # 3. ML Prediction
        vectorizer = ml_objects["vectorizer"]
        model = ml_objects["model"]
        
        features = vectorizer.transform([clean_text]) 
        ml_pred = int(model.predict(features)[0])
        
        confidence = 0.0
        if hasattr(model, "predict_proba"):
            confidence = max(model.predict_proba(features)[0])
        elif hasattr(model, "decision_function"):
             # Fallback confidence for SGD/SVM
             confidence = 1.0 # Placeholder for non-probabilistic models

        # 4. Consensus
        if rule_prediction is not None and rule_prediction != ml_pred:
            final_pred = rule_prediction
            conf_display = "99.00% (Rule Override)"
        else:
            final_pred = ml_pred
            conf_display = f"{confidence * 100:.2f}%"

        result_label = SENTIMENT_MAP.get(final_pred, "Unknown")

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result_label,
            "confidence": conf_display,
            "original_text": text
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request, "result": "Error", "error_message": str(e)
        })

if __name__ == "__main__":
    import uvicorn
    from dotenv import load_dotenv
    load_dotenv()
    # Note: When running locally without NGINX, access via http://localhost:5000/Sentiment-Analysis/
    # or simply remove the root_path from the URL in browser if testing directly.
    uvicorn.run("app:app", host="0.0.0.0", port=5000, reload=True, reload_excludes=["logs/*"])