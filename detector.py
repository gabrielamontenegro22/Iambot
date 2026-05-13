import os
from ultralytics import YOLO
import config


def _detect_format(path):
    if os.path.isdir(path):
        files = os.listdir(path)
        if any(f.endswith('.param') for f in files):
            return 'ncnn'
    if path.endswith('.pt'):
        return 'pt'
    if path.endswith('.onnx'):
        return 'onnx'
    return None


class Detector:
    """
    Detector con MODO DIAGNÓSTICO activado.

    Imprime al arrancar:
      - Qué formato detectó del modelo
      - Cuántas clases tiene el modelo (un modelo entrenado solo para patos
        debería tener 1 clase, no 80 como COCO)

    Y cada 30 detecciones imprime:
      - Cuántas detecciones por frame
      - Rango de coordenadas (debería estar dentro de RESIZE_WIDTH)
      - Rango de confianza
      - Qué clases salieron
    """

    def __init__(self):
        fmt = _detect_format(config.MODEL_PATH)

        print(f"[Detector] MODEL_PATH = {config.MODEL_PATH}")
        print(f"[Detector] Formato detectado: {fmt}")
        print(f"[Detector] CONFIDENCE_THRESHOLD = {config.CONFIDENCE_THRESHOLD}")
        print(f"[Detector] RESIZE_WIDTH = {config.RESIZE_WIDTH}")

        if fmt == 'ncnn':
            self.model = YOLO(config.MODEL_PATH, task='detect')
        elif fmt == 'pt':
            self.model = YOLO(config.MODEL_PATH)
            try:
                self.model.fuse()
            except Exception:
                pass
        else:
            self.model = YOLO(config.MODEL_PATH)

        # ─── INFO DEL MODELO ────────────────────────────────────────
        try:
            names = self.model.names if hasattr(self.model, 'names') else None
            print(f"[Detector] Clases del modelo: {names}")
            if names and len(names) > 5:
                print(f"[Detector] ⚠️  MODELO TIENE {len(names)} CLASES")
                print(f"[Detector] ⚠️  Parece COCO genérico, NO un modelo")
                print(f"[Detector] ⚠️  entrenado solo para patos.")
                print(f"[Detector] ⚠️  Esto explica clicks en cosas que no son el objetivo.")
        except Exception as e:
            print(f"[Detector] No se pudieron leer clases: {e}")

        # Warmup
        import numpy as np
        dummy = np.zeros((config.RESIZE_HEIGHT, config.RESIZE_WIDTH, 3),
                         dtype=np.uint8)
        print("[Detector] Warmup...")
        for _ in range(2):
            self.model(dummy, conf=config.CONFIDENCE_THRESHOLD,
                       imgsz=config.RESIZE_WIDTH, verbose=False)
        print("[Detector] Listo.\n")

        # Contadores para log periódico
        self._call_count = 0
        self._log_every  = 30

    def detect(self, frame):
        h_frame, w_frame = frame.shape[:2]
        self._call_count += 1

        results = self.model(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            imgsz=config.RESIZE_WIDTH,
            verbose=False,
        )

        detections = []
        classes_seen = []

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                continue

            xyxy  = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()

            cls_arr = None
            try:
                if r.boxes.cls is not None:
                    cls_arr = r.boxes.cls.cpu().numpy().astype(int)
            except Exception:
                pass

            for i, ((x1, y1, x2, y2), conf) in enumerate(zip(xyxy, confs)):
                x1c = max(0, min(int(x1), w_frame - 1))
                y1c = max(0, min(int(y1), h_frame - 1))
                x2c = max(0, min(int(x2), w_frame - 1))
                y2c = max(0, min(int(y2), h_frame - 1))

                if x2c <= x1c or y2c <= y1c:
                    continue

                cx = (x1c + x2c) // 2
                cy = (y1c + y2c) // 2
                w  = x2c - x1c
                h  = y2c - y1c

                cls_id = int(cls_arr[i]) if cls_arr is not None else -1
                classes_seen.append(cls_id)

                detections.append({
                    "x":     cx,
                    "y":     cy,
                    "w":     w,
                    "h":     h,
                    "conf":  float(conf),
                    "cls":   cls_id,
                })

        # ─── LOG PERIÓDICO ──────────────────────────────────────────
        if self._call_count % self._log_every == 0:
            n = len(detections)
            if n > 0:
                xs = [d["x"] for d in detections]
                ys = [d["y"] for d in detections]
                confs_list = [d["conf"] for d in detections]
                print(f"[DET #{self._call_count}] {n} dets | "
                      f"x:[{min(xs)}-{max(xs)}] y:[{min(ys)}-{max(ys)}] | "
                      f"conf:[{min(confs_list):.2f}-{max(confs_list):.2f}] | "
                      f"clases: {set(classes_seen)}")
            else:
                print(f"[DET #{self._call_count}] 0 dets")

        return detections