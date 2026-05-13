import cv2


def draw_detections(frame, detections, tracked_targets):
    """
    detections:      list of dicts {x, y, w, h, conf}  — raw YOLO (verde)
    tracked_targets: list of KalmanTracker             — tracks confirmados (rojo + ID)
    """

    # Raw YOLO detections — círculos verdes
    for d in detections:
        cv2.circle(frame, (d["x"], d["y"]), 6, (0, 255, 0), 1)

    # Tracked targets — caja roja + ID + vector de velocidad
    for t in tracked_targets:
        pos = t.position
        x1, y1, x2, y2 = t.bbox

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.circle(frame, pos, 4, (0, 0, 255), -1)

        # ID
        cv2.putText(
            frame, f"ID {t.id}",
            (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            (0, 0, 255), 1, cv2.LINE_AA
        )

        # Vector de velocidad (escalado para visualización).
        # velocity está en px/s; lo escalamos a una fracción para que sea visible.
        try:
            vx, vy = t.velocity
            # Mostrar predicción a 100ms en el futuro
            end = (int(pos[0] + vx * 0.1), int(pos[1] + vy * 0.1))
            cv2.arrowedLine(frame, pos, end, (0, 255, 255), 1, tipLength=0.3)
        except Exception:
            pass

    return frame