WINDOW_NAME = ""

GAME_ZONE = None
# Modelo. IMPORTANTE en Raspberry Pi:
#   - El formato .pt (PyTorch) funciona pero es LENTO en ARM.
#   - NCNN es ~2-4x más rápido en Raspberry Pi (recomendación oficial Ultralytics).
#   - Para exportar:  yolo export model=models/duck.pt format=ncnn
#     Eso crea  models/duck_ncnn_model/  -> usa esa ruta aquí.
MODEL_PATH = "models/duck_ncnn_model"   # cambia a "models/duck.pt" si aún no exportaste

CONFIDENCE_THRESHOLD = 0.35

SHOTS_PER_TARGET = 1 # Delay (segundos) entre disparos consecutivos al MISMO objetivo (ráfaga).
SHOT_BURST_DELAY = 0.4 # Delay (segundos) entre cambiar de un objetivo a otro distinto.
CLICK_DELAY = 0.2

# Si un mismo target_id sobrevive entre clicks, identificar el "mismo objetivo"
# por ID del tracker (True) o por distancia (False).
USE_TRACKER_ID_FOR_BURST = True

# Distancia (px) bajo la cual dos posiciones se consideran "mismo objetivo"
# cuando USE_TRACKER_ID_FOR_BURST = False.
SAME_TARGET_RADIUS = 60

# Tras dispararle a un target, NO vuelvas a dispararle a ese mismo target
# durante este tiempo (segundos), aunque siga vivo en el tracker.
# Si solo queda un target en pantalla, espera este tiempo y vuelve a tirarle.
TARGET_COOLDOWN = 0.4

DEBUG        = True
DEBUG_WIDTH  = 640
DEBUG_HEIGHT = 480

# Imprime stats por CONSOLA cada N segundos cuando DEBUG=False.
# 0 = desactivado.
CONSOLE_STATS_INTERVAL = 2.0

# 0.0 = detectar lo más rápido posible.
DETECTION_INTERVAL = 0.0

# Render de debug — mantener bajo en Pi.
DEBUG_INTERVAL = 0.1

# Multi-tracker
MAX_DIST            = 200
IOU_THRESHOLD       = 0.25
TRACKING_LOST_LIMIT = 10

# Tamaño de inferencia. En Raspberry esto es el parámetro CRÍTICO:
#   - 256: muy rápido, ~40-60ms/frame. Para CPU bajo presión.
#   - 320: rápido, ~50-80ms/frame en Pi 5 + NCNN. Recomendado.
#   - 416: balance, ~80-130ms/frame.
#   - 640: solo si tus objetivos son muy pequeños y aceptas <8 FPS.
RESIZE_WIDTH  = 320
RESIZE_HEIGHT = 320

# Lead time adicional (ms) sobre el ciclo medido.
# Si los clicks van atrás del objetivo, sube. Si van adelantados, baja.
EXTRA_LEAD_MS = 20