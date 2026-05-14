# Duck Hunt Aimbot

Proyecto para la materia de Inteligencia Artificial.

Es un bot que juega solo al Duck Hunt en la versión web (CrazyGames). Captura la pantalla, detecta los patos, predice hacia dónde van a estar en unos milisegundos, y dispara con el mouse en ese punto.

Llegué a ganar la partida (You Win, 7200 puntos), aunque en niveles avanzados se escapan algunos patos por límites físicos del sistema.

## Cómo funciona

El loop principal hace estos pasos en cada iteración:

1. Captura el frame del área del juego con la librería `mss`.
2. Convierte el frame a HSV y aplica una máscara de color para aislar los patos (que son rojos/marrones o gris oscuro). Esto está en `detector.py`.
3. Combina la máscara de color con una máscara de movimiento (diferencia entre frame actual y anterior). Sin esto el tronco del árbol y el perro daban falsos positivos.
4. Encuentra los contornos, filtra por área y descarta los que están en la parte de abajo de la pantalla (donde está el perro cazador).
5. Cada detección se pasa al tracker (`tracker.py`), que mantiene un filtro de Kalman por cada pato para estimar posición y velocidad. La asociación entre detecciones nuevas y trackers existentes se hace con el algoritmo Húngaro (`scipy.optimize.linear_sum_assignment`).
6. Para disparar, no se apunta a la posición actual del pato sino a donde va a estar cuando llegue el click. Esto compensa el lag del pipeline (que mide ~95ms en total). El cálculo está en `main.py` y usa la velocidad del Kalman.
7. El click se hace con `SendInput` de la API de Windows directamente (usando `ctypes`). Esto fue importante porque la librería `pynput` que usaba al principio tardaba ~30ms por click, y con SendInput baja a menos de 2ms. En Linux usa pynput como fallback.

## Estructura

```
app.py                # Arranca el UI y después el bot
main.py               # Loop principal
ui.py                 # Selector de ventana y área (Tkinter)
detector.py           # HSV + motion gating
tracker.py            # Kalman + Hungarian matching
mouse_controller.py   # SendInput (Win) / pynput (Linux)
window_capture.py     # Captura mss + busca la ventana del navegador
learner.py            # Ajuste adaptativo del cooldown entre disparos
config.py             # Todos los parámetros tuneables
calibrate_hsv.py      # Herramienta para ajustar los rangos HSV
diagnose.py           # Mide latencias sin disparar (debug)
```

## Instalación

En Windows:

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

En Linux (probado en Ubuntu, debería funcionar en Raspberry):

```
sudo apt install python3-tk wmctrl xdotool
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

Si estás en Linux con Wayland, la captura y el listado de ventanas no funciona, hay que cambiar a sesión Xorg. Para forzarlo, agregar `WaylandEnable=false` en `/etc/gdm3/custom.conf` debajo de `[daemon]` y reiniciar.

## Cómo se usa

1. Abrir Duck Hunt en el navegador (mejor en pantalla completa con F11).
2. Correr `python app.py`.
3. En la ventana del bot, dar "Actualizar lista" y seleccionar la ventana del navegador.
4. Apretar "Seleccionar área de juego" y dibujar un rectángulo sobre la zona donde vuelan los patos.
5. Iniciar.

## Parámetros importantes

Están en `config.py`. Los que más afectan el resultado:

- `DETECTOR_BACKEND` está en `"color"`. Probé también con YOLO pero el modelo genérico (yolov8n) no funciona bien con sprites pixelados.
- `CLICK_DELAY = 0.30`. Es el tiempo entre clicks a distintos patos. Más bajo no anduvo porque CrazyGames parece filtrar clicks demasiado rápidos (como anti-autoclicker).
- `SHOTS_PER_TARGET = 1`. Mismo motivo: probé con ráfagas de 3 disparos y mataba menos.
- `EXTRA_LEAD_MS = 60`. Cuánto se adelanta el click a la trayectoria del pato.
- `COLOR_USE_MOTION = True`. Combina el filtro de color con el de movimiento.
- `COLOR_IGNORE_BOTTOM_FRAC = 0.30`. Ignora el 30% inferior del frame para no dispararle al perro.
- `MIN_VELOCITY_PX_S = 35`. Velocidad mínima para considerar válido un target.

Si querés recalibrar los colores en otra pantalla con `python calibrate_hsv.py` se abre una ventana con sliders y vas ajustando hasta que solo los patos aparezcan en blanco en la máscara.

## Librerías usadas

- OpenCV (procesamiento de imagen)
- NumPy (Kalman, álgebra)
- SciPy (Hungarian matching)
- mss (captura de pantalla)
- Ultralytics + NCNN (modelo YOLO opcional)
- Tkinter (UI)
- pynput (mouse en Linux)
- ctypes (Win32 SendInput)

## Notas

- El proyecto está pensado para correr eventualmente en una Raspberry Pi, por eso se incluyó la opción del modelo NCNN (que es ~3x más rápido en ARM que PyTorch puro).
- El `learner.py` tiene un sistema que ajusta automáticamente el cooldown entre disparos según la tasa de aciertos, pero está más como prueba de concepto que como mejora real.
- Algunos patos siempre se van a escapar: el pipeline total tarda ~95ms y a velocidades de 200 px/s eso significa que el pato cruzó casi 20 px entre que lo detecté y le disparé. Si el lead predice mal, lo pierdo.
