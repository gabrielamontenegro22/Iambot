from ultralytics import YOLO
import config


class Detector:

    def __init__(self):
        self.model = YOLO(config.MODEL_PATH)

    def detect(self, frame):
        results = self.model(frame, conf=0.25, imgsz=416)

        detections = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])

                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                detections.append({
                    "x": cx,
                    "y": cy,
                    "conf": conf,
                    "class": cls
                })

        return detections