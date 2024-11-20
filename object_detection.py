import cv2
import numpy as np
from ultralytics import YOLO

class ObjectDetection:
    def __init__(self, model_path="yolov8s.pt", target_classes=None, confidence_threshold=0.5):
        self.model = YOLO(model_path)
        self.target_classes = target_classes if target_classes else [67, 65]
        self.confidence_threshold = confidence_threshold

    def initializate_cameras():
        cam = cv2.VideoCapture(0)
        if not cam.isOpened():
            print("Erro ao iniciar as câmaras.")
            exit()
        return cam

    def detect_object_positions(self, frame):
        frame = cv2.flip(frame, 1)
        # frame2 = cv2.flip(frame2, 1)

        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.model(frame_RGB, verbose=False)

        object_positions = []
        for obj in results[0].boxes.data:
            x1, y1, x2, y2, conf, class_id = obj.cpu().numpy()
            if int(class_id) in self.target_classes and conf > self.confidence_threshold:
                object_center_x = (x1 + x2) / 2
                object_positions.append(object_center_x)

        return object_positions

    def annotate_frame(self, frame, frame_rate):
        annotated_frame = frame.copy()
        results = self.model(frame, verbose=False)

        for obj in results[0].boxes.data:
            x1, y1, x2, y2, conf, class_id = obj.cpu().numpy()
            if int(class_id) in self.target_classes and conf > self.confidence_threshold:
                cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                label = f"{self.model.names[int(class_id)]}: {conf:.2f}"
                cv2.putText(annotated_frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.putText(annotated_frame, f"FPS: {int(frame_rate)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        return annotated_frame