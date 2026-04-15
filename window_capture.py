import subprocess
import mss
import numpy as np
import cv2


def get_window_geometry(window_name):
    output = subprocess.check_output(["wmctrl", "-lG"]).decode()

    for line in output.splitlines():
        if window_name in line:
            parts = line.split()
            return {
                "left": int(parts[2]),
                "top": int(parts[3]),
                "width": int(parts[4]),
                "height": int(parts[5])
            }
    return None


def capture_window(region):
    with mss.mss() as sct:
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)