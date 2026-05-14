# 🦆 Duck Hunt Aimbot

Bot autónomo que detecta y dispara automáticamente a los patos del juego
**Duck Hunt** (versión web — CrazyGames) usando visión por computadora,
predicción de trayectorias y control nativo del mouse.

Proyecto desarrollado para la materia de **Inteligencia Artificial**.

---

## 🎯 Resultado

🏆 **You Win!** con **7200 puntos** en CrazyGames Duck Hunt.

Pipeline completo de captura → detección → tracking → predicción → click
con latencia total de ~95ms.

---

## 🧠 Pipeline técnico

```
┌──────────────┐   ┌─────────────┐   ┌─────────────┐   ┌──────────────┐   ┌──────────────┐
│  Captura     │ → │  Detección  │ → │  Tracking   │ → │  Predicción  │ → │  Click       │
│  (mss)       │   │  (HSV +     │   │  (Kalman +  │   │  (lead time) │   │  (SendInput) │
│  ~16ms       │   │  motion)    │   │  Hungarian) │   │              │   │  <2ms        │
└──────────────┘   └─────────────┘   └─────────────┘   └──────────────┘   └──────────────┘
```

### 1. Captura de pantalla (`window_capture.py`)
- Librería `mss` para captura rápida del área del juego.
- Soporte multiplataforma: `pygetwindow` en Windows, `wmctrl` en Linux.

### 2. Detección de patos (`detector.py`)
Dos backends disponibles, configurables en `config.py`:

- **`color`** (usado): segmentación HSV + motion gating (frame differencing).
  Filtra el árbol, el perro cazador y el HUD por velocidad y zona muerta.
- **`yolo`**: red neuronal YOLOv8 (NCNN para Raspberry Pi). Disponible pero
  no usado en CrazyGames porque los sprites pixelados confunden al modelo
  pre-entrenado.

### 3. Tracking multi-objeto (`tracker.py`)
- Filtro de **Kalman** por cada pato (estimación de posición + velocidad).
- Asociación de detecciones a trackers vía **algoritmo Húngaro**
  (`scipy.optimize.linear_sum_assignment`) con costo híbrido IoU + distancia.
- Filtros anti-explosión de IDs:
  - `MAX_TRACKERS = 40`
  - Filtro de velocidad mínima (descarta blobs estáticos)
  - Cooldown por target tras disparo

### 4. Predicción con lead time (`main.py`)
El click apunta a donde el pato **estará** cuando la bala llegue, no a donde
está ahora. Se calcula:

```
lead_time = cycle_time_ema + EXTRA_LEAD_MS
target_x  = current_x + velocity_x * lead_time
target_y  = current_y + velocity_y * lead_time
```

`EXTRA_LEAD_MS = 60` compensa el lag del browser + el SendInput.

### 5. Click nativo (`mouse_controller.py`)
Inyección de eventos del mouse usando **Win32 API SendInput** vía `ctypes`,
latencia de <2ms (vs ~30ms de `pynput`). Fallback automático a `pynput` en Linux.

---

## 📦 Estructura del proyecto

```
Aimbot/
├── app.py                # Entry point (UI + bot loop)
├── main.py               # Loop principal del bot
├── ui.py                 # Selector de ventana + área de juego (Tkinter)
├── detector.py           # HSV + motion gating + YOLO opcional
├── tracker.py            # Kalman + Hungarian matching multi-objeto
├── mouse_controller.py   # Click nativo Win32 SendInput / pynput fallback
├── window_capture.py     # Captura mss + búsqueda de ventana del navegador
├── learner.py            # Sistema adaptativo (ajusta TARGET_COOLDOWN)
├── utils.py              # Helpers
├── config.py             # ⚙️ TODA la configuración tunable
├── calibrate_hsv.py      # Herramienta interactiva para calibrar colores HSV
├── diagnose.py           # Diagnóstico (mide latencia y FPS sin disparar)
├── models/               # Modelo YOLO custom exportado a NCNN
└── requirements.txt
```

---

## 🚀 Instalación

### Windows

```bash
git clone https://github.com/gabrielamontenegro22/Iambot.git
cd Iambot
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

### Linux (Ubuntu / Raspberry Pi)

```bash
sudo apt install -y python3-pip python3-venv python3-tk wmctrl xdotool
git clone https://github.com/gabrielamontenegro22/Iambot.git
cd Iambot
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 app.py
```

> ⚠️ **Linux con Wayland:** la captura y el listado de ventanas requieren
> sesión **Xorg**. Si estás en Wayland, editá `/etc/gdm3/custom.conf` y
> agregá `WaylandEnable=false` debajo de `[daemon]`, luego reiniciá.

---

## ⚙️ Uso

1. Abrí Duck Hunt en el navegador (recomendado: pantalla completa F11).
2. Ejecutá `python app.py`.
3. En la ventana **Vision Bot**:
   - **Actualizar lista** → seleccioná la ventana del navegador.
   - **Seleccionar área de juego** → dibujá un rectángulo sobre el área donde
     vuelan los patos.
   - **▶ INICIAR**.
4. Para detener, cerrá la ventana o presioná `Ctrl+C` en la terminal.

---

## 🔧 Tuning rápido

Los parámetros más importantes están en `config.py`:

| Parámetro | Valor | Efecto |
|---|---|---|
| `DETECTOR_BACKEND` | `"color"` | `"color"` para sprites pixelados; `"yolo"` para gráficos realistas |
| `CLICK_DELAY` | `0.30` | Tiempo entre clicks. Más bajo = más rápido (pero CrazyGames filtra <0.25) |
| `SHOTS_PER_TARGET` | `1` | Disparos por pato. CrazyGames filtra ráfagas como autoclicker |
| `EXTRA_LEAD_MS` | `60` | Anticipación. Si los clicks caen atrás del pato, subir |
| `COLOR_USE_MOTION` | `True` | Combinar HSV con detección de movimiento (filtra árbol estático) |
| `COLOR_IGNORE_BOTTOM_FRAC` | `0.30` | Ignora 30% inferior (perro cazador + pasto + HUD) |
| `MIN_VELOCITY_PX_S` | `35` | Velocidad mínima para considerar un target (filtra HUD estático) |
| `TARGET_COOLDOWN` | `0.5s` | Tiempo antes de re-tirarle al mismo pato |

Para calibrar los colores HSV en una pantalla nueva:
```bash
python calibrate_hsv.py
```

---

## 📊 Stack tecnológico

- **Python 3.10+**
- **OpenCV** — procesamiento de imágenes, HSV, motion detection
- **NumPy** — álgebra lineal (Kalman)
- **SciPy** — algoritmo Húngaro para asociación de targets
- **mss** — captura de pantalla rápida
- **Ultralytics YOLO** + **NCNN** — modelo opcional para detección por red
- **ctypes** — Win32 API SendInput
- **Tkinter** — UI del selector de ventana/área
- **pynput** — fallback de mouse en Linux

---

## 🧪 Optimizaciones aplicadas

| # | Optimización | Impacto |
|---|---|---|
| 1 | SendInput nativo (vs `pynput`) | -28ms de latencia de click |
| 2 | Motion gating combinado con HSV | Eliminó falsos positivos del árbol y perro |
| 3 | Zona muerta inferior (30%) | Ignora perro cazador y HUD |
| 4 | Filtro de velocidad mínima | Ignora blobs estáticos (HUD, ramas) |
| 5 | Predict-skip si target sale de pantalla | Evita clicks en coordenadas fuera del juego |
| 6 | Lead time adaptativo | Compensa el lag del pipeline (~95ms total) |
| 7 | Cooldown por target ID | Rota entre patos en vez de spamear al mismo |

---

## 📝 Licencia

Proyecto académico — uso educativo.

---

## 👤 Autora

**Gabriela Montenegro** — Proyecto de la materia *Inteligencia Artificial*.
