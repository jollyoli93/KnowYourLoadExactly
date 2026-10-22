from app.ModelClasses import Load_Classify

from .. import KYLE
from ultralytics import YOLO
from app.ModelClasses import Roi_Predict

class YOLO_KYLE(KYLE):
    def __init__(self):
        super().__init__(self)
        self.objd_model = YOLO("app/Models/crane_models/yolov8_basic.pt")
        self.objd_model_ss = YOLO("app/Models/crane_models/yolov8_basic.pt")
        self.roi_predict = Roi_Predict("app/Models/crane_models/LoadRegressionStandardv2.pth")
        self.classify_model = Load_Classify()