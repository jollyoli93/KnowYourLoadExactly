from typing import Union

from fastapi import FastAPI
import crane_load_detector

app = FastAPI()


@app.get("/")
def read_root():
    image = 'images/2507190940060000_jpg.rf.25413f'
    img = cv2.imread(image)
    
    kyle1 = crane_load_detector.KYLE()
    kyle1.detect_image(img)
    # return {"Hello": "World"}


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    return {"item_id": item_id, "q": q}