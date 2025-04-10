import torch
import cv2

model = torch.hub.load('yolov5', 'custom', path='best.pt', source='local')
class_names = ['Hardhat', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 'Person',
               'Safety Cone', 'Safety Vest', 'machinery', 'vehicle']

def video_detection(video_path):
    cap = cv2.VideoCapture(video_path)
    alert_triggered = False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{class_names[int(cls)]} {conf:.2f}"
            if class_names[int(cls)].startswith("NO-"):
                alert_triggered = True
            color = (0,255,0) if "NO-" not in class_names[int(cls)] else (0,0,255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        yield frame, alert_triggered
    cap.release()

def webcam_detection():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        alert_triggered = False
        results = model(frame)
        for *xyxy, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = map(int, xyxy)
            label = f"{class_names[int(cls)]} {conf:.2f}"
            if class_names[int(cls)].startswith("NO-"):
                alert_triggered = True
            color = (0,255,0) if "NO-" not in class_names[int(cls)] else (0,0,255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        yield frame, alert_triggered