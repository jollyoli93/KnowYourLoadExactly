import torch
from sklearn.preprocessing import StandardScaler
import numpy as np
from pathlib import Path
from math import sqrt

def extract_coords(coords):
    coord_split = [
        [float(x) for x in line.strip().split()]
        for line in coords
        if line.strip()
    ]
    return coord_split

#Gets statistics from training data for MLP ROI Predictor
def get_pairs(coords, from_file=True):
  coord_split = []

  if from_file:
    coord_split = extract_coords(coords)
  else:
    coord_split = coords

  hooks = [d for d in coord_split if d[0] == 0]  # hook
  loads = [d for d in coord_split if d[0] == 1]  # load

  pairs = {'hook': [], 'load': []}
  used_hooks = set()

  for load in loads:
      xc_load, yc_load = load[1], load[2]
      h_load = load[-1]
      closest = None
      closest_idx = -1
      min_dist = float("inf")

      for i, hook in enumerate(hooks):
          if i in used_hooks:
              continue

          xc_hook, yc_hook = hook[1], hook[2]

          yt_load = yc_load - (h_load /2)
          yb_hook = yc_hook + hook[-1]/2 #Location at bottom of hook

          # Hook must be above load
          if yb_hook > yt_load:
              continue

          # Horizontal closeness
          dist = abs(xc_load - xc_hook)
          if dist < min_dist:
              min_dist = dist
              closest = hook
              closest_idx = i

      if closest is not None:
          xc, yc, w, h = closest[1:5]
          # aspect = w / h if h != 0 else 0.0
          # area = w*h
          yb = yc + (h / 2)
          #dist = sqrt((xc - xc_load)**2 + (yc - yc_load)**2). feading load dims to features might create shortcut/memorisations

          # # get relative offsets normalised
          dx = (xc_load - xc) /w
          dy = (yc_load - yb) /h

          pairs['hook'].append(
              [xc, yc, w, h] + [yb]
          )
          pairs['load'].append([dx, dy])

          used_hooks.add(closest_idx)

  return pairs

def crop_load(img, x, y, w, h):
  """
    Takes as input, top left corner, width and height.
  """
  crop_img = img.copy()
  cropped_img = crop_img[y:y+h,x:x+w]

  return cropped_img

def box_gating(boxes):
  """
    Boxes contain a list of [class_idx, xc, yc, w, h, conf, src, hook_id] in pixel space.
    class 0 = hook, class 1 = load
    src in {"hook","obj","roi"}
    hook_id is an int for hook/roi loads, None for obj loads
    Returns loads only
  """
  if not boxes:
    print("Empty boxes")
    return []

  hooks = [d for d in boxes if d[6] == "hook"]
  obj_loads = [d for d in boxes if d[6] == "obj"]
  roi_loads  = [d for d in boxes if d[6] == "roi"]

  used_obj = set()
  matched_obj = set()
  kept_loads = []

  # For each hook find its highest matching load
  for hook in hooks:
    hook_x, hook_y, hook_w = hook[1], hook[2], hook[3]
    hook_id = hook[7]
    hook_boundary = hook_w * 3.0

    best_obj_idx = None
    best_obj_conf = float("-inf")

    for i, load in enumerate(obj_loads):
      load_x, load_y = load[1], load[2]

      # Gating - if it doesnt match then assume its on another hook
      if load_y < hook_y:
        continue
      if load_x < (hook_x - hook_boundary) or load_x > (hook_x + hook_boundary):
        continue

      matched_obj.add(i)

      if i in used_obj:
        continue

      conf = load[5]
      if conf > best_obj_conf:
        best_obj_conf = conf
        best_obj_idx = i

    if best_obj_idx is not None:
      kept_loads.append(obj_loads[best_obj_idx])
      used_obj.add(best_obj_idx)
    else:
      roi_candidate = [d for d in roi_loads if d[7] == hook_id]
      if roi_candidate:
        kept_loads.append(roi_candidate[0])

  for i, load in enumerate(obj_loads):
    if i not in matched_obj:
      kept_loads.append(load)

  return kept_loads

def corner_to_center(xt, yt, w,h):
  xc = xt+(w/2)
  yc = yt+(h/2)
  return xc, yc, w, h

def center_to_corners(xc, yc, w,h):
  xl = xc - (w/2)
  xr = xc + (w/2)
  yt = yc - (h/2)
  yb = yc + (h/2)
  return xl, xr, yt, yb, w, h