from typing import Union
import python_multipart

from fastapi import FastAPI
import cv2
from . import crane_load_detector as cdt

app = FastAPI()


@app.get("/")
def read_root():
    image = 'app/images/2503061230060000_jpg.rf.25f5fb4afcbba2631a14691fc0869d85.jpg'
    best_weights = "app/Models/crane_models/yolov8_basic.pt"

    kyle1 = cdt.KYLE(best_weights)
    return kyle1.detect_image(image)
    # return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}