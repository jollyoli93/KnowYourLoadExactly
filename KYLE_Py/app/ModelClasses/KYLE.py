from ultralytics import YOLO
import cv2
import torch
import numpy as np
from pathlib import Path

from app.Results import Box_Result, Image_Result
from app.image_tools import extract_coords, get_pairs, box_gating, corner_to_center, center_to_corners

def cv_show(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(5000)
    cv2.destroyAllWindows()


from ultralytics import YOLO
import cv2
import numpy as np

# Assuming box_gating, center_to_corners, corner_to_center, Image_Result, Box_Result are imported

class KYLE:
    def __init__(self, objd_model, ob_weights, roi_model, classify_model, second_detect=True):
        self.objd_model = objd_model(ob_weights)
        self.roi_predict = roi_model
        self.classify_model = classify_model
        self.predictions = []
        self.second_detect = second_detect

    def _crop_load(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]

    def _second_stage_detector(self, crop, roi_size):
        print("Predicting Second Crop")
        ss_results = self.objd_model.predict(crop, imgsz=roi_size, stream=False)

        for result in ss_results:
            if not result.boxes:
                continue

            for box in result.boxes:
                class_id = int(box.cls[0])
                if class_id == 1: # Load Class
                    conf = float(box.conf[0])
                    lx1, ly1, lx2, ly2 = map(int, box.xyxy[0].tolist())
                    load_only_crop = crop[ly1:ly2, lx1:lx2]
                    return load_only_crop, lx1, ly1, lx2, ly2, conf
        
        return None, 0, 0, 0, 0, 0

    def _predict_from_hook(self, image, box_size, hook_dims, second_detect=True):
        print("Predict from hook box size and dims:", box_size, hook_dims)
        roi = self.roi_predict.predict_load_roi(box_size, hook_dims)
        xl, yt, crop_w, crop_h = roi

        ROI = image[yt:yt+crop_h, xl:xl+crop_w]
        if ROI.size == 0: 
            return None

        load_crop, lx1, ly1, lx2, ly2, yolo_conf = self._second_stage_detector(ROI, max(crop_w, crop_h))

        if load_crop is not None and second_detect:
            print("Second ROI detected")
            gx1, gy1 = xl + lx1, yt + ly1
            gx2, gy2 = xl + lx2, yt + ly2
            return gx1, gy1, gx2 - gx1, gy2 - gy1
            
        if not second_detect:
            print("MLP proposed a region, but YOLO stage 2 found no load there.")
        else:
            print("Returning ROI")
            
        return roi

    def _predict_one_img(self, result, _img, idx, predict_from_hook=True):
        if not result.boxes:
            print("No results")
            image_name = str(getattr(result, "path", f"frame_{idx}")).rsplit("/", 1)[-1]
            return Image_Result(
                image_name=image_name, 
                image_path=getattr(result, "path", None),
                image_results=[], 
                image_size=result.orig_shape
            )

        box_preds = []

        for idx, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            obj_conf = float(box.conf[0])

            if class_id == 1:
                # Direct to list conversion
                xc, yc, w, h = map(int, box.xywh[0].tolist())
                xl, yt = xc - (w//2), yc - (h//2)
                box_preds.append([1, xc, yc, w, h, obj_conf, "obj", None])

            # Explicit check for class_id 0 (Hook)
            elif class_id == 0 and predict_from_hook:
                hook_dims_n = box.xywhn[0].tolist()
                hook_dims = box.xywh[0].tolist()

                print("Hook found - Crop ROI")
                roi_dims = self._predict_from_hook(_img, box.orig_shape, hook_dims_n, self.second_detect)

                if roi_dims is None:
                    print("ROI is empty")
                    continue

                center = corner_to_center(*roi_dims)
                if isinstance(center, np.ndarray):
                    center = center.tolist()
                    
                center = [int(c) for c in center]

                box_preds.append([0, *hook_dims, None, "hook", idx])
                box_preds.append([1, *center, None, "roi", idx])

        gated_boxes = box_gating(box_preds)
        image_results = []

        for boxes in gated_boxes:
            xc, yc, w, h = boxes[1:5]
            xl, _, yt, _, w, h = center_to_corners(xc, yc, w, h)

            xl = int(max(0, min(xl, _img.shape[1] - 1)))
            yt = int(max(0, min(yt, _img.shape[0] - 1)))
            w  = int(max(1, min(w, _img.shape[1] - xl)))
            h  = int(max(1, min(h, _img.shape[0] - yt)))

            crop = self._crop_load(_img, xl, yt, w, h)
            
            # Prevent classification if crop is invalid
            if crop.size == 0:
                continue
                
            pred_class, confidence = self.classify_model.predict_one_image(crop)

            image_results.append(Box_Result(
                class_type=pred_class,
                x_c=xc, y_c=yc, width=w, height=h, conf=confidence,
            ))

        image_name = result.path.rsplit("/", 1)[-1]
        return Image_Result(
            image_name=image_name, 
            image_path=result.path,
            image_results=image_results, 
            image_size=result.orig_shape
        )

    # Replaces detect and detect_yolo_only
    def detect(self, input, predict_from_hook=True):
        results = self.objd_model.predict(input, stream=False)
        if results is None:
            print("No predictions")
            return []

        outputs = []
        for r in results:
            img = r.orig_img
            out = self._predict_one_img(r, img, idx=1, predict_from_hook=predict_from_hook)
            outputs.append(out)
            self.predictions.append(out)
            
        return outputs

    # Replaces detect_stream and detect_stream_yolo
    def detect_stream(self, stream, predict_from_hook=True):
        results = self.objd_model.predict(stream, stream=True)

        for i, r in enumerate(results):
            img = r.orig_img
            out = self._predict_one_img(r, img, idx=i, predict_from_hook=predict_from_hook)
            self.predictions.append(out)
            yield out, r