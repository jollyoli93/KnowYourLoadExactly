from typing import Union
import python_multipart
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import cv2
import numpy as np

from . import crane_load_detector as cdt
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
    image = 'app/images/2503061230060000_jpg.rf.25f5fb4afcbba2631a14691fc0869d85.jpg'
    kyle1 = cdt.KYLE()
    return kyle1.detect_image(image)

@app.post("/detect/")
async def detect_one_image(file: UploadFile = File(...)):
    logger.info("Detect endpoint accessed")
    
    content_type = file.content_type
    logger.info(f"File uploaded: {content_type}")
    if (content_type == "image/jpeg" or content_type == "image/jpg"):
        detect_one = cdt.KYLE()
        
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
        loger.info(f"Incorrect file type. Ending.")
        return {"Response":"Incorrect file type"}