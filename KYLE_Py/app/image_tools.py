import torch

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()
    
