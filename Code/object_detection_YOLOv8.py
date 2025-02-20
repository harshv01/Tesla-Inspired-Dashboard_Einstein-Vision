import torch
from PIL import Image
import os
import pickle
from ultralytics import YOLO
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

scene = 'scene3'

def calculate_center(bbox):
    x_min, y_min, x_max, y_max = bbox
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return int(y_center), int(x_center)

model = YOLO('yolov8n.pt').to(device)

frame_names = os.listdir(f"P3Data/Frames/{scene}")
frame_names.sort(key=lambda x: int(x.strip('.jpg')[5:]))

results_bbs = []
results_labels= []

for i in range(len(frame_names)):

    image = Image.open(f'P3Data/Frames/{scene}/{frame_names[i]}').convert("RGB")
    pred = model.predict(source=image, conf=0.5)[0]

    bbs_for_detected_objs = []
    classes_for_detected_objs = []

    boxes = pred.boxes
    labels = pred.names
    
    for box in boxes:
        bbs_for_detected_objs.append(box.xyxy.cpu().tolist()[0])
        classes_for_detected_objs.append(labels[int(box.cls)])
    
    results_bbs.append(bbs_for_detected_objs)
    results_labels.append(classes_for_detected_objs)
  

pickle_data = {"bb": results_bbs, "labels": results_labels}
pickle_file_path = f'results/yolo_{scene}_all.pkl'

with open(pickle_file_path, "wb") as pickle_file:
    pickle.dump(pickle_data, pickle_file)
    
