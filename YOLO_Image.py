import torch
import cv2
import os

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
               'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

def detect_image(image_path):
    img = cv2.imread(image_path)
    results = model(img)
    alert_flag = False
    for *xyxy, conf, cls in results.xyxy[0]:
        x1, y1, x2, y2 = map(int, xyxy)
        label = f"{class_names[int(cls)]} {conf:.2f}"
        if class_names[int(cls)].startswith("NO-"):
            alert_flag = True
        color = (0,255,0) if "NO-" not in class_names[int(cls)] else (0,0,255)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    result_path = os.path.join("static/files", "result.jpg")
    cv2.imwrite(result_path, img)
    return result_path, alert_flag
