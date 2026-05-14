WINDOW_NAME = ""

GAME_ZONE = None
# Modelo. IMPORTANTE en Raspberry Pi:
#   - El formato .pt (PyTorch) funciona pero es LENTO en ARM.
#   - NCNN es ~2-4x más rápido en Raspberry Pi (recomendación oficial Ultralytics).
#   - Para exportar:  yolo export model=models/duck.pt format=ncnn
#     Eso crea  models/duck_ncnn_model/  -> usa esa ruta aquí.
MODEL_PATH = "yolov8n.pt"   # COCO genérico para probar en Windows. En Pi usar models/duck_ncnn_model

CONFIDENCE_THRESHOLD = 0.15

# Filtro de clases — solo dispara a estas clases del modelo.
# COCO: 14 = bird (pájaro). None = todas las clases.
# Si entrenás un modelo custom (ej. solo patos), usar [0].
TARGET_CLASSES = None

# Backend de detección:
#   "yolo"  — red neuronal (genérica o entrenada). Lenta sin GPU, no funciona
#             bien con sprites pixelados como Duck Hunt.
#   "color" — visión clásica: detecta movimiento (background subtraction) +
#             filtro de tamaño/forma. Ideal para juegos retro con fondo estático.
DETECTOR_BACKEND = "color"

# Parámetros del backend "color" — ajustar si detecta basura o se pierde patos
COLOR_MIN_AREA  = 60     # bajado de 80 → con motion gating los blobs se cortan un poco, dejamos margen
COLOR_MAX_AREA  = 6000   # px² máximo
COLOR_MAX_DETS_PER_FRAME = 12  # subido de 4 → permite frames con varios patos volando juntos

# Rangos HSV (OpenCV: H ∈ [0,179], S ∈ [0,255], V ∈ [0,255])
# Pato rojo/marrón: H rojo-naranja (0-25 y wrap 160-179), saturado, no muy claro
COLOR_RED_LOW1  = (0,   80, 40)
COLOR_RED_HIGH1 = (25, 255, 220)
COLOR_RED_LOW2  = (160, 80, 40)
COLOR_RED_HIGH2 = (179, 255, 220)

# Pato oscuro/gris: incluye dark hasta gris medio (V hasta 90).
# Antes V=45 (muy negro) → no capturaba patos grises de CrazyGames Duck Hunt.
# Subido para que el sprite gris/dark de los patos entre en la máscara.
# El árbol verde NO entra porque tiene saturación alta (S>80).
COLOR_DARK_LOW  = (0,   0, 0)
COLOR_DARK_HIGH = (179, 80, 90)

# Zona muerta inferior — ignora detecciones en la parte de abajo del frame.
# Útil para no dispararle al perro cazador que aparece sobre el pasto.
# 0.0 = sin zona muerta. 0.30 = ignora el 30% inferior (perro + pasto + HUD).
# Subido de 0.22 → 0.30 porque el perro caminando aparece tambien arriba del
# pasto (y=200-240 en frame 320), y el filtro de velocidad solo lo limita a 35 px/s.
COLOR_IGNORE_BOTTOM_FRAC = 0.30

# Combinar color con movimiento (frame differencing).
# True: un objeto debe tener color de pato Y estar moviéndose para ser detectado.
# False: solo color. El árbol queda filtrado por el FILTRO DE VELOCIDAD del tracker
#        (main.py — _is_moving), no necesitamos motion gating en el detector.
# ACTIVADO (True) — sin esto, el TRONCO del árbol y el PERRO entran al rango HSV
# "rojo/marrón" y disparan falsos positivos. Con motion gating, solo lo que se
# mueve frame a frame cuenta → árbol estático queda excluido del AND.
COLOR_USE_MOTION = True

SHOTS_PER_TARGET = 1     # 1 solo tiro por pato — preciso, sin desperdicio.
                         # Teoria: CrazyGames probablemente filtra clicks consecutivos rapidos
                         # como "autoclicker", aceptando solo 1 cada ~300ms. Si era asi,
                         # rafagas rapidas se descartaban excepto el primero.
SHOT_BURST_DELAY = 0.0   # No aplica (1 tiro por pato).
# ═══════════════════════════════════════════════════════════════════
# CLICK_DELAY — milisegundos entre clicks a diferentes patos.
#
#   0.30 = SEGURO. Probado: WIN con 7200 pts. Algunos escapan en niveles altos.
#   0.15 = AGRESIVO. Mas patos por segundo. Riesgo: si el juego filtra clicks
#          consecutivos rapidos, podriamos volver a 3/6.
#
# Si 0.15 funciona peor, cambia este numero a 0.30 y guarda. Listo.
# ═══════════════════════════════════════════════════════════════════
CLICK_DELAY      = 0.30  # Probando agresivo. Para revertir: cambiar a 0.30

# Si un mismo target_id sobrevive entre clicks, identificar el "mismo objetivo"
# por ID del tracker (True) o por distancia (False).
USE_TRACKER_ID_FOR_BURST = True

# Distancia (px) bajo la cual dos posiciones se consideran "mismo objetivo"
# cuando USE_TRACKER_ID_FOR_BURST = False.
SAME_TARGET_RADIUS = 60

# Tras dispararle a un target, NO vuelvas a dispararle a ese mismo target
# durante este tiempo (segundos), aunque siga vivo en el tracker.
# Si solo queda un target en pantalla, espera este tiempo y vuelve a tirarle.
# Subido de 0.25 → 0.5 para que el bot tenga más tiempo de ir a OTROS patos
# antes de volver al mismo. Combinado con _pick_target_for_burst que excluye
# targets en cooldown, esto fuerza al bot a rotar entre todos los patos.
TARGET_COOLDOWN = 0.5

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
# MAX_DIST bajo evita que el tracker confunda dos patos diferentes como "el mismo
# pato que se movió". Si vuelves a tener IDs saltando entre patos, subilo.
MAX_DIST            = 80
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
# Bajado de 60 → 20: el lead anterior era muy agresivo (24px adelante con vel=200px/s),
# resultaba en clicks fuera de pantalla cuando el pato volaba hacia arriba.
EXTRA_LEAD_MS = 60    # Bajado de 90 → 60 porque SendInput elimino ~28ms de lag.
                      # cycle_time_ema(~10ms) + EXTRA(60) = 70ms total lead.
                      # Lag real del pipeline ahora: capture(16) + python(10) +
                      # SendInput(1) + browser(40) ≈ 67ms. Lead matches lag.
                      # Si los clicks van adelantados (delante del pato), bajar a 40.