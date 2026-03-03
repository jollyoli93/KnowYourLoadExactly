from pathlib import Path
from typing import Union
import python_multipart
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np
from ultralytics import YOLO

from app.Models.Load_Classify import Load_Classify
from app.Models.Roi_Predict import Roi_Predict

from app.KYLE import KYLE
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

@app.get("/")
def read_root():
    logger.info("Root endpoint accessed")
    return {"Hello": "World"}

@app.get("/test_detect")
def detect_test():
    logger.info("Running test detection model")
    BASE_DIR = Path(__file__).resolve().parent
    image = BASE_DIR / "images" / "crane.jpg"
    # objd_model_ss = YOLO("./Models/crane_models/yolov8_basic.pt")
    roi_predict = Roi_Predict("./Models/crane_models/LoadRegressionStandardv2.pth")
    classify_model = Load_Classify("./Models/crane_models/crane_classifier.pth")
    kyle1 = KYLE(objd_model=YOLO, ob_weights="./Models/crane_models/yolov8_basic.pt", roi_model=roi_predict, classify_model=classify_model)
    return kyle1.detect(image)

@app.post("/detect/")
async def detect_one_image(file: UploadFile = File(...)):
    logger.info("Detect endpoint accessed")
    
    content_type = file.content_type
    logger.info(f"File uploaded: {content_type}")
    if (content_type == "image/jpeg" or content_type == "image/jpg"):
        detect_one = KYLE()
        
        logger.info(f"Awaiting file read.")
        image_bytes = await file.read()
        
        logger.info(f"Converting image format.")
        image = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        logger.info(f"Running inference...")
        result = detect_one.detect_image(image)
        
        logger.info("Inferrence Complete")
        return result
    else:
        logger.info(f"Incorrect file type. Ending.")
        return {"Response":"Incorrect file type"}