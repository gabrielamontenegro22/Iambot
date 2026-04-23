def run_bot():
    import time
    import config
    from window_capture import get_window_geometry, capture_window
    from detector import Detector
    from mouse_controller import move_and_click
    from utils import draw_detections
    import cv2

    detector = Detector()
    last_save = 0

    while True:
        region = get_window_geometry(config.WINDOW_NAME)

        if not region:
            print("Ventana no encontrada")
            time.sleep(1)
            continue

        frame = capture_window(region)
        
          #if time.time() - last_save > 1:
        #    cv2.imwrite(f"dataset/images/all/frame_{time.time()}.jpg", frame)
        #   last_save = time.time()
        
        if time.time() - last_save > 1:
             cv2.imwrite(f"dataset/images/all/frame_{time.time()}.jpg", frame)
             last_save = time.time()
        detections = detector.detect(frame)

        if detections:
            detections.sort(key=lambda d: d["conf"], reverse=True)
            best = detections[0]

            move_and_click(best["x"], best["y"], region)

        if config.DEBUG:
            debug_frame = draw_detections(frame, detections)
            cv2.imshow("Debug", debug_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

        time.sleep(config.CLICK_DELAY)