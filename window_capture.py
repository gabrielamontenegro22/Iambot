import subprocess
import mss
import numpy as np
import cv2


def get_window_geometry(window_name):
    output = subprocess.check_output(["wmctrl", "-lG"]).decode()

    for line in output.splitlines():
        if window_name.lower() in line.lower():
            parts = line.split()

            x = int(parts[2])
            y = int(parts[3])
            w = int(parts[4])
            h = int(parts[5])

            # Evitar valores inválidos
            if w > 0 and h > 0:
                return {
                    "left": x,
                    "top": y,
                    "width": w,
                    "height": h
                }

    return None


def capture_window(region):
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)