import torch
from ultralytics import YOLO
import subprocess
import time
import threading
import os
from PIL import Image
import cv2
from ultralytics import YOLO

def train_model(device: str, source: str):
    model = YOLO('yolov5su.pt')
    model.train(
        data=source,
        epochs=30,
        imgsz=416,
        batch=8,
        device=device,
        amp=True,
        workers=8,
        lr0=0.001
    )

def show_inference_result(annotated_frame):
    """Helper function to display the annotated frame."""
    cv2.imshow('YOLO Live', annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Stopped by user.")
        return False
    return True

def process_video(model, source, device, res):
    cap = cv2.VideoCapture(source)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, res)
        results = model.predict(source=frame, device=device)
        annotated = results[0].plot() 
        if not show_inference_result(annotated):
            break
    cap.release()
    cv2.destroyAllWindows()

def process_image(model, source, device):
    img = cv2.imread(source)
    results = model.predict(source=img, device=device)
    annotated = results[0].plot() 
    show_inference_result(annotated)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def process_image_folder(model, source, device):
    abs_path = os.path.abspath(source)
    results = model.predict(source=abs_path, device=device, save=True)
    
    for r in results:
        num_boxes = len(r.boxes)
        avg_conf = r.boxes.conf.mean().item() if num_boxes > 0 else 0
        print(f"{r.path}: {num_boxes} boxes, avg conf {avg_conf:.2f}")

def validate_model(device: str, modelSource: str, source: str, res = None):
    model = YOLO(modelSource)

    if source.endswith(('.mp4', '.avi')):
        process_video(model, source, device, res)
    elif os.path.isfile(source):
        process_image(model, source, device)
    else:
        process_image_folder(model, source, device)


def evaluate_model(device: str, modelSource: str, data_yaml: str):
    print("\n--- Evaluating Model ---")

    model = YOLO(modelSource)
    metrics = model.val(data=data_yaml, device=device)

    precision = metrics.box.mp
    recall = metrics.box.mr
    map50 = metrics.box.map50
    map = metrics.box.map

    print("\n--- Evaluation Results ---")
    print(f"Precision (mean):     {precision:.4f}")
    print(f"Recall (mean):        {recall:.4f}")
    print(f"mAP@0.5:              {map50:.4f}")
    print(f"mAP@0.5:0.95:         {map:.4f}")

if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nRunning on: {device}")

    import yaml


    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dataset4', 'data.yaml'))

    #train_model(device, yaml_path)

    #validate_model(device, 'runs/detect/train/weights/best.pt', 'video.mp4', (1280, 720))

    evaluate_model(device, 'runs/detect/train_not_best/weights/best.pt', yaml_path)

