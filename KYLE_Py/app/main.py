from pathlib import Path
from typing import Union
import python_multipart
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from ultralytics import YOLO

from app.Utils.ModelCache import ModelCache
from app.ModelClasses.Load_Classify import Load_Classify
from app.ModelClasses.Roi_Predict import Roi_Predict

from app.ModelClasses.KYLE import KYLE
import logging

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Create a logger instance
logger = logging.getLogger(__name__)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://unnatural-intelligence.onrender.com"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

@app.on_event("startup")
def load_models():
    global detect_one

    session_classify = ModelCache.get_classify_session()
    session_roi = ModelCache.get_roi_session()

    classify_model = Load_Classify(session_classify)
    roi_predict = Roi_Predict(session_roi)
    # roi_predict = Roi_Predict(BASE_DIR /"Models/crane_models/LoadRegressionAdam250.pth")


    detect_one = KYLE(
        objd_model=YOLO, 
        ob_weights=BASE_DIR / "Models/crane_models/yolov8_basic.pt", 
        roi_model=roi_predict, 
        classify_model=classify_model)

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"Hello": "World"}

@app.get("/test_detect")
def detect_test():
    logger.info("Running test detection model")
    image = BASE_DIR / "images" / "crane.jpg"
    return detect_one.detect(image)

@app.post("/detect")
async def detect_one_image(file: UploadFile = File(...)):
    logger.info("Detect endpoint accessed")
    
    content_type = file.content_type
    logger.info(f"File uploaded: {content_type}")
    if (content_type == "image/jpeg" or content_type == "image/jpg"): # Would png work?
        
        logger.info(f"Awaiting file read.")
        image_bytes = await file.read()
        
        logger.info(f"Converting image format.")
        image = np.frombuffer(image_bytes, dtype=np.uint8) # Why int not float32?
        image = cv2.imdecode(image, cv2.IMREAD_COLOR) # What is this doing?
        
        logger.info(f"Running inference...")
        result = detect_one.detect(image)
        
        logger.info("Inferrence Complete")
        return result
    else:
        logger.info(f"Incorrect file type. Use JPG or JPEG only.")
        return {"Response":"Incorrect file type. Use JPG or JPEG only."}