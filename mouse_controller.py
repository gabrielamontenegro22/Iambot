from pynput.mouse import Controller, Button

# Single controller instance — reused for every click (no re-init overhead)
_mouse = Controller()


def move_and_click(x, y, region):
    """
    Instant move + click using pynput.

    x, y: coordenadas en píxeles RELATIVAS al area de captura (region).
    region: dict con left/top/width/height de la zona de captura.

    El clamp garantiza que el click siempre cae dentro del área del
    juego, incluso si la predicción extrapola fuera.
    """
    # Clamp a los límites del área del juego
    x = max(0, min(x, region["width"]  - 1))
    y = max(0, min(y, region["height"] - 1))

    screen_x = region["left"] + x
    screen_y = region["top"]  + y

    _mouse.position = (screen_x, screen_y)
    _mouse.click(Button.left)