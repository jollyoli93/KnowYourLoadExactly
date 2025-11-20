from typing import Union
import python_multipart
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import cv2

from . import crane_load_detector as cdt

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/test_detect")
def detect_test():
    image = 'app/images/2503061230060000_jpg.rf.25f5fb4afcbba2631a14691fc0869d85.jpg'
    kyle1 = cdt.KYLE()
    return kyle1.detect_image(image)

@app.post("/detect/")
async def detect_one_image(file: UploadFile = File(...)):
    content_type = file.content_type
    
    print(content_type)
    
    if (content_type == "image/jpeg" or content_type == "image/jpg"):
        detect_one = cdt.KYLE()
        result = detect_one.detect_image(file)
        
        return {"Response":f"{resul[0]}"}
    else:
        return {"Response":"Incorrect file type"}