import cv2


def draw_detections(frame, detections):
    for d in detections:
        x = d["x"]
        y = d["y"]

        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
        cv2.rectangle(frame, (x-10, y-10), (x+10, y+10), (0,255,0), 2)
    return frame