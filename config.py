WINDOW_NAME = ""

GAME_ZONE = None

# Modelo. En Raspberry conviene usar NCNN (mas rapido que PyTorch en ARM).
# Para exportar: yolo export model=models/duck.pt format=ncnn
MODEL_PATH = "yolov8n.pt"

CONFIDENCE_THRESHOLD = 0.15

# Filtro de clases. None = todas. En COCO 14 = pajaro. Si entrenas custom, usar [0].
TARGET_CLASSES = None

# Backend de deteccion:
#   "yolo"  - red neuronal. Lenta sin GPU y mala con sprites pixelados.
#   "color" - HSV + movimiento. Anda mejor para Duck Hunt.
DETECTOR_BACKEND = "color"

# Parametros del backend color
COLOR_MIN_AREA  = 60
COLOR_MAX_AREA  = 6000
COLOR_MAX_DETS_PER_FRAME = 12

# Rangos HSV (OpenCV: H 0-179, S 0-255, V 0-255)
# Pato rojo/marron (el H rojo da la vuelta, por eso dos rangos)
COLOR_RED_LOW1  = (0,   80, 40)
COLOR_RED_HIGH1 = (25, 255, 220)
COLOR_RED_LOW2  = (160, 80, 40)
COLOR_RED_HIGH2 = (179, 255, 220)

# Pato oscuro/gris. Subi V hasta 90 porque con 45 se me escapaban los grises.
# El arbol verde no entra porque tiene saturacion alta.
COLOR_DARK_LOW  = (0,   0, 0)
COLOR_DARK_HIGH = (179, 80, 90)

# Ignora la parte de abajo del frame (perro cazador, pasto, HUD)
COLOR_IGNORE_BOTTOM_FRAC = 0.30

# Si True, ademas del color exige que el objeto se este moviendo.
# Sin esto, el tronco y el perro daban falsos positivos.
COLOR_USE_MOTION = True

# Disparos por pato. CrazyGames parece filtrar clicks consecutivos rapidos
# como autoclicker, asi que con 1 anda mejor que con 3.
SHOTS_PER_TARGET = 1
SHOT_BURST_DELAY = 0.0

# Tiempo entre clicks a distintos patos.
# Probe 0.15 y andaba peor (CrazyGames filtra rafagas). 0.30 dio WIN 7200.
CLICK_DELAY = 0.30

# Si un mismo target sobrevive entre clicks, lo identifico por ID del tracker
USE_TRACKER_ID_FOR_BURST = True

# Si USE_TRACKER_ID_FOR_BURST = False, considera "mismo target" por distancia
SAME_TARGET_RADIUS = 60

# Tras tirarle a un target no le vuelvo a tirar por este tiempo (segundos).
# Subi a 0.5 para forzar al bot a rotar entre los patos antes de re-disparar.
TARGET_COOLDOWN = 0.5

DEBUG        = True
DEBUG_WIDTH  = 640
DEBUG_HEIGHT = 480

# Imprime stats cada N segundos cuando DEBUG=False. 0 desactiva.
CONSOLE_STATS_INTERVAL = 2.0

# 0 = detectar lo mas rapido posible
DETECTION_INTERVAL = 0.0

# Render de debug. En Pi conviene tenerlo bajo.
DEBUG_INTERVAL = 0.1

# Multi-tracker
# MAX_DIST chico para que no confunda dos patos como "el mismo que se movio"
MAX_DIST            = 80
IOU_THRESHOLD       = 0.25
TRACKING_LOST_LIMIT = 10

# Tamano de inferencia. En Raspberry es el parametro mas critico.
# 320 anda bien con NCNN (~50-80ms/frame)
RESIZE_WIDTH  = 320
RESIZE_HEIGHT = 320

# Lead time adicional (ms) sobre el ciclo medido.
# Si los clicks van atras del pato, sube. Si van adelantados, baja.
EXTRA_LEAD_MS = 60
