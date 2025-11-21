from pydantic import BaseModel

class Box_Result(BaseModel):
    class_type: str
    conf: float
    x_top: int
    y_top: int
    width: int
    height: int
    
class Image_Result(BaseModel):
    image_name: str
    image_size: tuple
    image_results: list[Box_Result]