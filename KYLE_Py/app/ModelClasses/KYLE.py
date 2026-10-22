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


class KYLE:
    def __init__(self, objd_model, ob_weights, roi_model, classify_model):
        self.objd_model = objd_model(ob_weights)
        self.objd_model_ss = objd_model(ob_weights)
        self.roi_predict = roi_model
        self.classify_model = classify_model
        self.predictions = []

    def _crop_load(self, img, x, y, w, h):
        cropped_img = img[y:y+h, x:x+w]
        print(cropped_img.shape)
        cv2.imwrite(str(f"crop.png"), cropped_img)
        # cv_show("crop",cropped_img)
        return cropped_img

    def _second_stage_detector(self, crop, roi_size):
      """
        From ROI crop predict new object using YOLO
        If object found returns cropped image, local x1 x2, local y1 y2, confidence
        Else - None, 0,0,0,0,0

      """
      print("Predicting Second Crop")
      ss_results = self.objd_model_ss.predict(crop, imgsz=roi_size, stream=False)

      for result in ss_results:
        if not result.boxes:
          continue

        for box in result.boxes.cpu():
            class_id = int(box.cls[0])
            class_name = self.objd_model_ss.names[class_id]  # Get class name using the class ID
            conf = float(box.conf[0])
            label = f"{class_name}: {conf:.2f}"

            if class_id == 1: #Load Class
            #  LOCAL coordinates (relative to the ROI) top left, bottom right
              lx1, ly1, lx2, ly2 = box.xyxy[0].cpu().numpy().astype(int)

              # actual cropped image of the load
              load_only_crop = crop[ly1:ly2, lx1:lx2]
              return load_only_crop, lx1, ly1, lx2, ly2, conf
      return None, 0,0,0,0, 0

    def _predict_from_hook(self, image, box_size, hook_dims):
      """ Input original image, predict the ROI from first pass
          Pass through second stage object detector.
          Classify results
          returns x left, y top, w, h
       """
      print("predict from hook box size and dims",box_size, hook_dims)
      roi = self.roi_predict.predict_load_roi(box_size, hook_dims)
      xl, yt, crop_w, crop_h = roi

      # extract the ROI from the main image
      ROI = image[yt:yt+crop_h, xl:xl+crop_w]
      if ROI.size == 0: return

      # Find the load inside that ROI
      load_crop, lx1, ly1, lx2, ly2, yolo_conf = self._second_stage_detector(ROI, max(crop_w, crop_h))

      if load_crop is not None:
          print("Second ROI detected \n")
          # GLOBAL COORDINATES: ROI start + Local YOLO start
          gx1, gy1 = xl + lx1, yt + ly1
          gx2, gy2 = xl + lx2, yt + ly2

          w, h = gx2 - gx1, gy2 - gy1

          # Draw the final accurate box
          # predicted_class, confidence = self.classify_model.predict_one_image(load_crop)
          # xl, yt, w, h
          print("Second predict", gx1,gy1,w,h)
          return gx1,gy1,w,h
          # return gx1, gy1
      else:
          print("MLP proposed a region, but YOLO stage 2 found no load there. \n")
          # xl, yt, w, h
          return roi

    # Object detection
    def _predict_one_img(self, result, _img, idx):
      load_results = None

      if not result.boxes:
        print("No results")
        image_name = str(getattr(result, "path", f"frame_{idx}")).rsplit("/", 1)[-1]

        return Image_Result(image_name=image_name, image_path=getattr(result, "path", None),
                    image_results=[], image_size=result.orig_shape)


      box_preds = []

      for idx, box in enumerate(result.boxes.cpu()):
        # r = box.xyxy[0].numpy().astype(int) # Get corner points as int (Actual Co-ords)
        class_id = int(box.cls[0])  # Get class ID
        class_name = self.objd_model.names[class_id]  # Get class name using the class ID
        obj_conf = float(box.conf[0])
        label = f"{class_name}: {obj_conf:.2f}"
        # -----------------------------------
        # LOAD (class 1)
        # -----------------------------------
        if class_id == 1:
            xc, yc, w, h = box.xywh[0].cpu().numpy().tolist()
            xc, yc, w, h = int(xc), int(yc), int(w), int(h)
            xl = xc - (w//2)
            yt = yc - (h//2)

            xl = int(xl)
            yt = int(yt)
            w  = int(w)
            h  = int(h)

            # use OD conf for gating
            box_preds.append([1, xc, yc, w, h, obj_conf, "obj", None])

        # -----------------------------------
        # HOOK (class 0)
        # -----------------------------------
        else:
          hook_dims_n = box.xywhn[0]
          hook_dims = box.xywh[0].cpu().tolist()
          print("Hook found - Crop ROI")
          roi_dims = self._predict_from_hook(_img, box.orig_shape, hook_dims_n)
          if roi_dims is None:
            print("ROI is empty")
            continue

          roi_crop = self._crop_load(_img, *roi_dims)

          center = corner_to_center(*roi_dims)
          if isinstance(center, np.ndarray):
              center = center.tolist()
          center = [int(center[0]), int(center[1]), int(center[2]), int(center[3])]
          box_preds.append([0, *hook_dims, None, "hook", idx])
          box_preds.append([1, *center, None, "roi", idx])

      gated_boxes = box_gating(box_preds)
      image_results = []

      for boxes in gated_boxes:
        xc, yc, w, h = boxes[1:5]
        xl,_, yt,_, w, h = center_to_corners(xc, yc, w, h)

        # # clamp
        xl = int(max(0, min(xl, _img.shape[1] - 1)))
        yt = int(max(0, min(yt, _img.shape[0] - 1)))
        w  = int(max(1, min(w, _img.shape[1] - xl)))
        h  = int(max(1, min(h, _img.shape[0] - yt)))

        crop = self._crop_load(_img, xl, yt, w, h)
        print(crop.shape)
        pred_class, confidence = self.classify_model.predict_one_image(crop)

        box_result = Box_Result(
              class_type=pred_class,
              x_c=xc,
              y_c=yc,
              width=w,
              height=h,
              conf=confidence,
              )
        image_results.append(box_result)

      image_name = result.path.rsplit("/", 1)[-1]
      load_results = Image_Result(image_name=image_name, image_path = result.path,image_results = image_results, image_size=result.orig_shape)
      return load_results

    #Run inference
    def detect(self, input):
        results = self.objd_model.predict(input, stream=False)
        if results is None:
          print("No predictions")
          return

        crop_results = None

        if isinstance(self.objd_model, YOLO):
            outputs = []
            for r in results:
                img = r.orig_img
                out = self._predict_one_img(r, img, idx=1)
                outputs.append(out)
                self.predictions.append(out)
            return outputs

    # #Run inference on bulk images - upload array of images.
    def detect_stream(self, stream):
        results = self.objd_model.predict(stream, stream=True)

        for i, r in enumerate(results):
            img = r.orig_img
            out = self._predict_one_img(r, img, idx=i)
            self.predictions.append(out)
            yield out
