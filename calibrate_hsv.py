"""
Calibrador HSV interactivo.
Apunta el calibrador a la pantalla donde corre Duck Hunt y ajusta los sliders
hasta que SOLO los patos aparezcan en BLANCO en la mascara.

Uso:
    1. Abri Duck Hunt en el navegador (paus si podes)
    2. Editar GAME_REGION abajo con las coordenadas reales del juego
    3. python calibrate_hsv.py
    4. Mover sliders hasta aislar el pato
    5. Apretar 'q' para salir → imprime los valores a pegar en config.py
"""
import cv2
import numpy as np
import mss

# AJUSTAR a la posicion real del juego en tu pantalla
GAME_REGION = {'top': 200, 'left': 100, 'width': 900, 'height': 600}


def nothing(_x):
    pass


cv2.namedWindow('controls', cv2.WINDOW_NORMAL)
cv2.resizeWindow('controls', 400, 300)
cv2.createTrackbar('H_min', 'controls', 0,   179, nothing)
cv2.createTrackbar('S_min', 'controls', 0,   255, nothing)
cv2.createTrackbar('V_min', 'controls', 0,   255, nothing)
cv2.createTrackbar('H_max', 'controls', 179, 179, nothing)
cv2.createTrackbar('S_max', 'controls', 80,  255, nothing)
cv2.createTrackbar('V_max', 'controls', 90,  255, nothing)

print("Calibrador HSV — apretar 'q' para salir e imprimir valores")
print(f"Capturando region: {GAME_REGION}")

with mss.mss() as sct:
    while True:
        img = np.array(sct.grab(GAME_REGION))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos('H_min', 'controls')
        s_min = cv2.getTrackbarPos('S_min', 'controls')
        v_min = cv2.getTrackbarPos('V_min', 'controls')
        h_max = cv2.getTrackbarPos('H_max', 'controls')
        s_max = cv2.getTrackbarPos('S_max', 'controls')
        v_max = cv2.getTrackbarPos('V_max', 'controls')

        mask = cv2.inRange(
            hsv,
            np.array([h_min, s_min, v_min]),
            np.array([h_max, s_max, v_max]),
        )
        result = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('original', frame)
        cv2.imshow('mask',     mask)
        cv2.imshow('result',   result)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
print()
print("─── Valores calibrados ───")
print(f"H: [{h_min}, {h_max}]")
print(f"S: [{s_min}, {s_max}]")
print(f"V: [{v_min}, {v_max}]")
print()
print("Para pegar en config.py:")
print(f"    COLOR_DARK_LOW  = ({h_min}, {s_min}, {v_min})")
print(f"    COLOR_DARK_HIGH = ({h_max}, {s_max}, {v_max})")
