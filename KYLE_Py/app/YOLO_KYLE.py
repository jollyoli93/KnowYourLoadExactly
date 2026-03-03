from . import KYLE
from ultralytics import YOLO
from app.Models import Roi_Predict, Load_Classify

class YOLO_KYLE(KYLE):
    def __init__(self):
        super().__init__(self)
        self.objd_model = YOLO("app/Models/crane_models/yolov8_basic.pt")
        self.objd_model_ss = YOLO("app/Models/crane_models/yolov8_basic.pt")
        self.roi_predict = Roi_Predict("app/Models/crane_models/LoadRegressionStandardv2.pth")
        self.classify_model = Load_Classify("app/Models/crane_models/crane_classifier.pth")