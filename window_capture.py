"""
Captura de ventana con thread asíncrono.

El problema en Raspberry Pi: mss/XGetImage hace copia GPU→CPU que en ARM
puede tardar 100-300ms POR CAPTURA. Si el main loop hace todo en serie
(captura → YOLO → click), el lag se acumula.

Solución: thread dedicado que captura en bucle al ritmo más rápido
posible y guarda solo el ÚLTIMO frame. El main loop pide grab() y
recibe el frame más reciente disponible sin bloquear.

Esto convierte la latencia de "captura + YOLO + click" en "max(captura, YOLO+click)".
En Pi típicamente reduce el lag percibido de ~1500ms a ~150ms.
"""

import subprocess
import threading
import time
import mss
import numpy as np
import cv2


def get_window_geometry(window_name):
    """Geometría completa de la ventana del SO (de wmctrl)."""
    try:
        output = subprocess.check_output(
            ["wmctrl", "-lG"], timeout=2
        ).decode()
    except Exception:
        return None

    for line in output.splitlines():
        if window_name.lower() in line.lower():
            parts = line.split()
            x, y, w, h = int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])
            if w > 0 and h > 0:
                return {"left": x, "top": y, "width": w, "height": h}
    return None


def select_game_zone(window_region):
    """ROI selector — sin cambios respecto a versión anterior."""
    with mss.mss() as sct:
        shot = sct.grab(window_region)
        img = np.array(shot, dtype=np.uint8)
        preview = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    max_display = 1200
    scale = min(max_display / preview.shape[1],
                max_display / preview.shape[0],
                1.0)
    if scale < 1.0:
        display = cv2.resize(preview, (0, 0), fx=scale, fy=scale)
    else:
        display = preview
        scale   = 1.0

    cv2.namedWindow("Seleccionar area de juego", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Seleccionar area de juego",
                     display.shape[1], display.shape[0])

    instr = display.copy()
    cv2.putText(instr, "Dibuja el area de juego",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(instr, "ENTER/SPACE=confirmar  C=cancelar",
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    cv2.imshow("Seleccionar area de juego", instr)
    cv2.waitKey(500)

    roi = cv2.selectROI("Seleccionar area de juego", display,
                        showCrosshair=True, fromCenter=False)
    cv2.destroyWindow("Seleccionar area de juego")

    rx, ry, rw, rh = roi
    if rw == 0 or rh == 0:
        return None

    off_x = int(rx / scale)
    off_y = int(ry / scale)
    zone_w = int(rw / scale)
    zone_h = int(rh / scale)

    return (off_x, off_y, zone_w, zone_h)


# ════════════════════════════════════════════════════════════════════
# THREADED CAPTURER
# ════════════════════════════════════════════════════════════════════

class WindowCapture:
    """
    Capturador con thread asíncrono dedicado.

    Modo de uso (idéntico a la versión síncrona):
        cap = WindowCapture(target_w, target_h)
        cap.set_window_region(window_region, game_zone=...)
        frame, region = cap.grab()
        cap.close()

    Internamente, set_window_region() arranca un thread que captura
    constantemente al ritmo más rápido posible. grab() devuelve el
    último frame capturado, sin bloquear.
    """

    def __init__(self, target_w, target_h):
        self.target_w  = target_w
        self.target_h  = target_h

        self._region   = None
        self._scale_x  = 1.0
        self._scale_y  = 1.0

        # Buffer protegido por lock
        self._latest_frame  = None
        self._latest_region = None
        self._frame_lock    = threading.Lock()

        # Thread
        self._stop_event   = threading.Event()
        self._thread       = None
        self._thread_alive = False

        # Estadísticas (para mostrar el lag real)
        self._capture_count    = 0
        self._capture_time_ema = 0.05
        self._stats_lock       = threading.Lock()

    def set_window_region(self, window_region, game_zone=None):
        """Configura región y (re)arranca el thread si hace falta."""
        if game_zone is not None:
            off_x, off_y, gw, gh = game_zone
            new_region = {
                "left":   window_region["left"] + off_x,
                "top":    window_region["top"]  + off_y,
                "width":  gw,
                "height": gh,
            }
        else:
            new_region = dict(window_region)

        # Misma región, thread vivo → nada que hacer
        if (self._region is not None
            and new_region["left"]   == self._region["left"]
            and new_region["top"]    == self._region["top"]
            and new_region["width"]  == self._region["width"]
            and new_region["height"] == self._region["height"]
            and self._thread_alive):
            return

        self._stop_thread()
        self._region  = new_region
        self._scale_x = self._region["width"]  / self.target_w
        self._scale_y = self._region["height"] / self.target_h
        self._start_thread()

    def grab(self):
        """
        Devuelve el último frame disponible (NO bloquea esperando uno nuevo).
        Returns: (frame_bgr, region_dict) o (None, None)
        """
        with self._frame_lock:
            return self._latest_frame, self._latest_region

    def get_capture_stats(self):
        """Tiempo medio de una captura del thread (para diagnóstico)."""
        with self._stats_lock:
            return {
                "avg_capture_ms": self._capture_time_ema * 1000,
                "total_captures": self._capture_count,
            }

    @property
    def scale_x(self):
        return self._scale_x

    @property
    def scale_y(self):
        return self._scale_y

    def close(self):
        self._stop_thread()

    # Alias compatibilidad
    def set_region(self, region):
        self.set_window_region(region, game_zone=None)

    # ════════════════════════════════════════════════════════════════
    # Internos
    # ════════════════════════════════════════════════════════════════

    def _start_thread(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread_alive = True
        self._thread = threading.Thread(
            target=self._capture_loop,
            daemon=True,
            name="WindowCaptureThread",
        )
        self._thread.start()

    def _stop_thread(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=2.0)
        self._thread = None
        self._thread_alive = False

    def _capture_loop(self):
        """
        Loop del thread.
        mss.mss() se crea DENTRO del thread (X11 no es thread-safe).
        """
        sct = mss.mss()
        target_w = self.target_w
        target_h = self.target_h

        try:
            while not self._stop_event.is_set():
                if self._region is None:
                    time.sleep(0.05)
                    continue

                try:
                    t0 = time.time()

                    shot = sct.grab(self._region)
                    img  = np.asarray(shot, dtype=np.uint8)

                    if shot.width != target_w or shot.height != target_h:
                        img = cv2.resize(img, (target_w, target_h),
                                         interpolation=cv2.INTER_LINEAR)
                    bgr = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

                    elapsed = time.time() - t0

                    with self._frame_lock:
                        self._latest_frame  = bgr
                        self._latest_region = self._region

                    with self._stats_lock:
                        self._capture_count    += 1
                        self._capture_time_ema  = (
                            0.9 * self._capture_time_ema + 0.1 * elapsed
                        )

                except Exception as e:
                    print(f"[Capture] Error: {e}")
                    time.sleep(0.1)
        finally:
            try:
                sct.close()
            except Exception:
                pass