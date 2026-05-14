import os
import cv2
import numpy as np
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


# Backend YOLO (red neuronal)
class _YoloBackend:
    def __init__(self):
        from ultralytics import YOLO

        fmt = _detect_format(config.MODEL_PATH)

        print(f"[Detector/YOLO] MODEL_PATH = {config.MODEL_PATH}")
        print(f"[Detector/YOLO] Formato detectado: {fmt}")
        print(f"[Detector/YOLO] CONFIDENCE_THRESHOLD = {config.CONFIDENCE_THRESHOLD}")

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

        try:
            names = self.model.names if hasattr(self.model, 'names') else None
            print(f"[Detector/YOLO] Clases del modelo: {names}")
            if names and len(names) > 5:
                print(f"[Detector/YOLO] El modelo tiene {len(names)} clases, "
                      f"parece COCO generico. Para Duck Hunt usar 'color'.")
        except Exception as e:
            print(f"[Detector/YOLO] No se pudieron leer clases: {e}")

        # Warmup (las primeras inferencias son mas lentas)
        dummy = np.zeros((config.RESIZE_HEIGHT, config.RESIZE_WIDTH, 3),
                         dtype=np.uint8)
        print("[Detector/YOLO] Warmup...")
        for _ in range(2):
            self.model(dummy, conf=config.CONFIDENCE_THRESHOLD,
                       imgsz=config.RESIZE_WIDTH, verbose=False)
        print("[Detector/YOLO] Listo.\n")

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

        target_classes = getattr(config, "TARGET_CLASSES", None)
        detections     = []
        classes_seen   = []

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
                cls_id = int(cls_arr[i]) if cls_arr is not None else -1

                if target_classes is not None and cls_id not in target_classes:
                    continue

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

                classes_seen.append(cls_id)

                detections.append({
                    "x":    cx,
                    "y":    cy,
                    "w":    w,
                    "h":    h,
                    "conf": float(conf),
                    "cls":  cls_id,
                })

        if self._call_count % self._log_every == 0:
            n = len(detections)
            if n > 0:
                xs = [d["x"] for d in detections]
                ys = [d["y"] for d in detections]
                cl = [d["conf"] for d in detections]
                print(f"[YOLO #{self._call_count}] {n} dets | "
                      f"x:[{min(xs)}-{max(xs)}] y:[{min(ys)}-{max(ys)}] | "
                      f"conf:[{min(cl):.2f}-{max(cl):.2f}] | "
                      f"clases: {set(classes_seen)}")
            else:
                print(f"[YOLO #{self._call_count}] 0 dets")

        return detections


# Backend por color (HSV + filtro de tamano/forma)
# Para sprites del juego anda mucho mejor que YOLO generico
class _ColorBackend:

    def __init__(self):
        self.min_area = getattr(config, "COLOR_MIN_AREA", 80)
        self.max_area = getattr(config, "COLOR_MAX_AREA", 6000)
        self.max_dets = getattr(config, "COLOR_MAX_DETS_PER_FRAME", 4)
        self.ignore_bottom_frac = getattr(config, "COLOR_IGNORE_BOTTOM_FRAC", 0.0)
        self.use_motion = getattr(config, "COLOR_USE_MOTION", False)

        if self.use_motion:
            # Frame differencing: lo que cambia frame a frame = se esta moviendo.
            # Sin esto el tronco del arbol entraba al rango de color y disparaba.
            self._prev_gray = None
            self._motion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (15, 15)
            )
            self._motion_threshold = 3
            self._motion_dilate_iters = 8

        self.red_low1  = np.array(config.COLOR_RED_LOW1,  dtype=np.uint8)
        self.red_high1 = np.array(config.COLOR_RED_HIGH1, dtype=np.uint8)
        self.red_low2  = np.array(config.COLOR_RED_LOW2,  dtype=np.uint8)
        self.red_high2 = np.array(config.COLOR_RED_HIGH2, dtype=np.uint8)
        self.dark_low  = np.array(config.COLOR_DARK_LOW,  dtype=np.uint8)
        self.dark_high = np.array(config.COLOR_DARK_HIGH, dtype=np.uint8)

        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

        self._call_count = 0
        self._log_every  = 30

        print(f"[Detector/COLOR] Modo HSV color matching")
        print(f"[Detector/COLOR] Area valida: [{self.min_area}, {self.max_area}] px2")
        print(f"[Detector/COLOR] Max dets/frame: {self.max_dets}")

    def detect(self, frame):
        self._call_count += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Pato rojo/marron, uso dos rangos por el wrap del Hue
        mask_red = cv2.inRange(hsv, self.red_low1, self.red_high1) | \
                   cv2.inRange(hsv, self.red_low2, self.red_high2)

        # Pato oscuro/gris (no verdoso, sino entraria el arbol)
        mask_dark = cv2.inRange(hsv, self.dark_low, self.dark_high)

        mask = mask_red | mask_dark

        # Combino con mascara de movimiento (AND).
        # Si el objeto no se mueve, no me importa que tenga el color.
        if self.use_motion:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if self._prev_gray is not None:
                diff = cv2.absdiff(gray, self._prev_gray)
                _, motion = cv2.threshold(
                    diff, self._motion_threshold, 255, cv2.THRESH_BINARY
                )
                motion = cv2.dilate(motion, self._motion_kernel,
                                    iterations=self._motion_dilate_iters)
                mask = cv2.bitwise_and(mask, motion)
            self._prev_gray = gray

        # Ignoro la parte de abajo (perro/pasto)
        if self.ignore_bottom_frac > 0:
            h_mask = mask.shape[0]
            y_cut  = int(h_mask * (1.0 - self.ignore_bottom_frac))
            mask[y_cut:, :] = 0

        # Limpieza morfologica
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  self._kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area or area > self.max_area:
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = w / max(h, 1)
            if aspect < 0.25 or aspect > 5.0:
                continue

            cx = x + w // 2
            cy = y + h // 2

            detections.append({
                "x":    cx,
                "y":    cy,
                "w":    w,
                "h":    h,
                "conf": min(1.0, area / 1000.0),
                "cls":  0,
            })

        # Si hay demasiadas detecciones probablemente es ruido (cambio de pantalla,
        # animacion). Mejor no disparar.
        if len(detections) > self.max_dets:
            if self._call_count % self._log_every == 0:
                print(f"[COLOR #{self._call_count}] {len(detections)} dets "
                      f"> {self.max_dets}, ignoro (ruido)")
            return []

        if self._call_count % self._log_every == 0:
            n = len(detections)
            if n > 0:
                areas = [d["w"] * d["h"] for d in detections]
                print(f"[COLOR #{self._call_count}] {n} dets | "
                      f"areas: [{min(areas)}-{max(areas)}] px2")
            else:
                print(f"[COLOR #{self._call_count}] 0 dets")

        return detections


class Detector:
    # Elige el backend segun config.DETECTOR_BACKEND

    def __init__(self):
        backend = getattr(config, "DETECTOR_BACKEND", "yolo").lower()
        if backend == "color":
            self._impl = _ColorBackend()
        else:
            self._impl = _YoloBackend()

    def detect(self, frame):
        return self._impl.detect(frame)
