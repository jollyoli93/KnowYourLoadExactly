import cv2
import crane_load_detector as cdt

image = '../images/2503061230060000_jpg.rf.25f5fb4afcbba2631a14691fc0869d85.jpg'
best_weights = "./Models/crane_models/yolov8_basic.pt"

kyle1 = cdt.KYLE(best_weights)
kyle1.detect_image(image)