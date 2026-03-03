from pydantic import BaseModel

class Box_Result(BaseModel):
    x_c: int
    y_c: int
    width: int
    height: int
    conf: float
    class_type: str

class Image_Result(BaseModel):
    image_name: str
    image_path: str
    image_size: tuple
    image_results: list[Box_Result]