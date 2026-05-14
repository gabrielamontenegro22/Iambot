import time
import cv2
import config

from detector import Detector
from tracker import MultiTracker

from window_capture import get_window_geometry, WindowCapture
from mouse_controller import move_and_click
from utils import draw_detections


#cd Aimbot
#export TCL_LIBRARY="/c/Users/gabri_qm076gc/AppData/Local/Programs/Python/Python313/tcl/tcl8.6"
#export TK_LIBRARY="/c/Users/gabri_qm076gc/AppData/Local/Programs/Python/Python313/tcl/tk8.6"
#python app.py

# ═══════════════════════════════════════════════════════════════════
# SELECCIÓN DE OBJETIVO
# ═══════════════════════════════════════════════════════════════════

def _nearest_target(targets, last_click_pos):
    """Target más cercano al último click — minimiza recorrido del mouse."""
    if not targets:
        return None
    if last_click_pos is None:
        return targets[0]
    lx, ly = last_click_pos
    return min(
        targets,
        key=lambda t: (t.position[0] - lx)**2 + (t.position[1] - ly)**2,
    )


# ═══════════════════════════════════════════════════════════════════
# BURST STATE — control de disparos múltiples
# ═══════════════════════════════════════════════════════════════════

class BurstState:
    """
    Mantiene a qué objetivo le estamos disparando y cuántos disparos van.
    Cuando se completa la ráfaga, libera el objetivo y permite uno nuevo.
    """

    def __init__(self, shots_per_target, burst_delay,
                 use_id=True, same_radius=60):
        self.shots_required = max(1, int(shots_per_target))
        self.burst_delay    = burst_delay
        self.use_id         = use_id
        self.same_radius2   = same_radius * same_radius

        self.current_id   = None
        self.current_pos  = None
        self.shots_fired  = 0
        self.last_shot_at = 0.0

    def is_same_target(self, target):
        if self.current_id is None and self.current_pos is None:
            return False
        if self.use_id:
            return target.id == self.current_id
        tx, ty = target.position
        cx, cy = self.current_pos
        return (tx - cx)**2 + (ty - cy)**2 <= self.same_radius2

    def is_complete(self):
        return self.shots_fired >= self.shots_required

    def can_fire_now(self, now):
        return (now - self.last_shot_at) >= self.burst_delay

    def register_shot(self, target, now):
        if self.current_id is None and self.current_pos is None:
            self.current_id  = target.id
            self.current_pos = target.position
            self.shots_fired = 1
        else:
            self.shots_fired += 1
            self.current_pos = target.position
        self.last_shot_at = now

    def release(self):
        self.current_id   = None
        self.current_pos  = None
        self.shots_fired  = 0


MIN_VELOCITY_PX_S = 35     # Subido de 10 → 35 para filtrar perro caminando (5-20 px/s)
                           # y jitter de pixeles del HUD. Patos reales vuelan a 50-200 px/s,
                           # pasan sin problema.
HIT_STREAK_GRACE  = 0      # SIN grace. Todo tracker debe demostrar velocidad antes de
                           # ser disparado. Patos reales: Kalman calcula velocidad despues
                           # de 1-2 updates → pasan filtro. Blobs estaticos (iconos HUD,
                           # contador de patos, perro caminando) nunca alcanzan 35 px/s.


def _is_moving(target):
    """
    Filtra trackers estacionarios (probablemente árbol/score/objetos estáticos).
    Trackers nuevos (hit_streak < 4) reciben el beneficio de la duda porque
    la velocidad Kalman aún no convergió.
    """
    if target.hit_streak < HIT_STREAK_GRACE:
        return True
    vx, vy = target.velocity
    return (vx * vx + vy * vy) >= MIN_VELOCITY_PX_S ** 2


def _pick_target_for_burst(active, burst, last_click_pos,
                           recently_shot, now, cooldown):
    """
    Selección de target con tres reglas:

    1. SOLO LO QUE SE MUEVE: targets con velocidad ~0 son ignorados (árbol).
    2. RÁFAGA ACTIVA: si SHOTS_PER_TARGET > 1 y hay una ráfaga en curso,
       seguir disparando al mismo target hasta completar.
    3. EVITAR REPETIR: targets a los que se les disparó hace menos de
       `cooldown` segundos quedan EXCLUIDOS de la selección.
       Solo si TODOS los targets están en cooldown, se permite repetir.
    """
    if not active:
        return None

    # Caso 1: filtrar estacionarios — el árbol nunca se dispara
    active = [t for t in active if _is_moving(t)]
    if not active:
        return None

    # Caso 2: ráfaga activa sin completar
    if burst.current_id is not None or burst.current_pos is not None:
        for t in active:
            if burst.is_same_target(t):
                if not burst.is_complete():
                    return t
                burst.release()
                break
        else:
            burst.release()

    # Caso 3: filtrar targets en cooldown
    fresh = [
        t for t in active
        if (now - recently_shot.get(t.id, float('-inf'))) >= cooldown
    ]

    # Si todos están en cooldown, escoger el menos reciente
    if not fresh:
        return min(active, key=lambda t: recently_shot.get(t.id, float('-inf')))

    return _nearest_target(fresh, last_click_pos)


def _prune_recently_shot(recently_shot, now, cooldown):
    """Elimina entradas viejas para que el dict no crezca sin límite."""
    cutoff = now - cooldown * 3
    for tid in list(recently_shot.keys()):
        if recently_shot[tid] < cutoff:
            del recently_shot[tid]


def _check_display_server():
    """Detecta Wayland y avisa. En Pi, Wayland causa lag severo de captura."""
    import os
    session = os.environ.get("XDG_SESSION_TYPE", "")
    wayland_display = os.environ.get("WAYLAND_DISPLAY", "")

    if session == "wayland" or wayland_display:
        print()
        print("⚠️  WAYLAND DETECTADO — en Raspberry Pi causa lag de captura.")
        print("    Cambia a X11:  sudo raspi-config -> Advanced -> Wayland -> X11")
        print()


# ═══════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════

def run_bot():

    _check_display_server()

    detector      = Detector()
    multi_tracker = MultiTracker(
        iou_threshold = config.IOU_THRESHOLD,
        max_dist      = config.MAX_DIST,
        max_lost      = config.TRACKING_LOST_LIMIT,
    )

    capturer = WindowCapture(config.RESIZE_WIDTH, config.RESIZE_HEIGHT)

    burst = BurstState(
        shots_per_target = config.SHOTS_PER_TARGET,
        burst_delay      = config.SHOT_BURST_DELAY,
        use_id           = config.USE_TRACKER_ID_FOR_BURST,
        same_radius      = config.SAME_TARGET_RADIUS,
    )

    last_detect    = 0.0
    last_debug     = 0.0
    last_click     = 0.0
    last_click_pos = None
    last_geo_check = 0.0
    last_loop_time = time.time()
    last_frame_id  = None

    # Memoria de targets disparados recientemente (anti-repetición)
    recently_shot = {}

    detections        = []
    debug_initialized = False

    # EMA del tiempo de inferencia — para lead time predictivo
    # Valor inicial 0.010 (10ms) realista para HSV+OpenCV en CPU moderna.
    # Antes era 0.06 (60ms) → over-predict en los primeros ~20 frames.
    # Ahora converge sobre un valor cercano a la realidad desde el primer frame.
    cycle_time_ema = 0.010

    print(f"[Bot] Iniciando — SHOTS_PER_TARGET={config.SHOTS_PER_TARGET}, "
          f"target_cooldown={config.TARGET_COOLDOWN*1000:.0f}ms")

    try:
        while True:

            now = time.time()
            dt  = now - last_loop_time
            last_loop_time = now

            # ─────────────────────────────────────────────────────────
            # GEOMETRÍA DE VENTANA — una vez por segundo
            # ─────────────────────────────────────────────────────────
            if now - last_geo_check > 1.0 or capturer._region is None:
                window_region = get_window_geometry(config.WINDOW_NAME)
                if window_region:
                    capturer.set_window_region(
                        window_region,
                        game_zone=getattr(config, "GAME_ZONE", None)
                    )
                last_geo_check = now

            if capturer._region is None:
                time.sleep(0.5)
                continue

            # ─────────────────────────────────────────────────────────
            # KALMAN PREDICT — cada iteración, con dt real
            # ─────────────────────────────────────────────────────────
            if 0 < dt < 1.0:
                multi_tracker.predict_all(dt)

            # ─────────────────────────────────────────────────────────
            # CAPTURA (no bloqueante — devuelve último frame del thread)
            # ─────────────────────────────────────────────────────────
            frame, region = capturer.grab()
            if frame is None:
                time.sleep(0.01)
                continue

            # Saltar YOLO si es el mismo frame que ya procesamos
            frame_is_new = (id(frame) != last_frame_id)
            last_frame_id = id(frame)

            # ─────────────────────────────────────────────────────────
            # YOLO + TRACKER UPDATE — solo si hay frame nuevo
            # ─────────────────────────────────────────────────────────
            if frame_is_new and (now - last_detect >= config.DETECTION_INTERVAL):

                detect_start = time.time()
                raw          = detector.detect(frame)
                detect_dur   = time.time() - detect_start

                cycle_time_ema = 0.8 * cycle_time_ema + 0.2 * detect_dur
                last_detect    = now

                # ─── TRACKER TRABAJA EN ESPACIO DEL FRAME (320x320) ────
                # NO escalar aquí. MAX_DIST=200 está calibrado para frame,
                # no para game zone. Si escalamos, el matching falla y se
                # crean trackers infinitos cada frame.
                # raw ya viene en coords del frame YOLO.
                detections = raw

                multi_tracker.update(detections)

            # ─────────────────────────────────────────────────────────
            # SELECCIÓN DE TARGET + CLICK
            # Los targets están en espacio del FRAME (320x320).
            # Escalamos a game zone SOLO al hacer el click.
            # ─────────────────────────────────────────────────────────
            active = multi_tracker.active_targets()

            _prune_recently_shot(recently_shot, now, config.TARGET_COOLDOWN)

            target = _pick_target_for_burst(
                active, burst, last_click_pos,
                recently_shot, now, config.TARGET_COOLDOWN,
            )

            # ═══ DIAGNOSTICO: imprime estado cada 60 frames si hay activos ═══
            if active and (int(now * 10) % 6 == 0):
                # Mostrar estado de cada tracker activo
                tracker_info = []
                for t in active[:5]:
                    vx, vy = t.velocity
                    speed = (vx*vx + vy*vy) ** 0.5
                    moves = _is_moving(t)
                    in_cd = (now - recently_shot.get(t.id, float('-inf'))) < config.TARGET_COOLDOWN
                    tracker_info.append(
                        f"id={t.id} pos={t.position} hits={t.hit_streak} "
                        f"vel={speed:.0f}px/s move={moves} cd={in_cd}"
                    )
                print(f"[DIAG] active={len(active)} target={'SI id=' + str(target.id) if target else 'NO'}")
                for ti in tracker_info:
                    print(f"       {ti}")

            if target is not None:
                same_target_as_burst = burst.is_same_target(target) and \
                                       (burst.current_id is not None or
                                        burst.current_pos is not None)

                if same_target_as_burst:
                    can_shoot = burst.can_fire_now(now)
                else:
                    can_shoot = (now - last_click) >= config.CLICK_DELAY

                # ═══ DIAGNOSTICO: por que no dispara ═══
                if not can_shoot:
                    time_since_click = (now - last_click) * 1000
                    print(f"[DIAG] NO dispara id={target.id} — can_shoot=False "
                          f"(time_since_last_click={time_since_click:.0f}ms, "
                          f"CLICK_DELAY={config.CLICK_DELAY*1000:.0f}ms)")

                if can_shoot:
                    extra_lead_s = config.EXTRA_LEAD_MS / 1000.0
                    lead_s = cycle_time_ema + extra_lead_s

                    # tx_frame, ty_frame en espacio frame (320x320)
                    tx_frame, ty_frame = target.predict_future(lead_s)

                    # ─── NO disparar si la prediccion cae fuera del frame ───
                    # El pato se escapo por arriba/abajo/lados — clickear ahi
                    # es perder un tiro. Marcamos cooldown para no insistir.
                    frame_w = config.RESIZE_WIDTH
                    frame_h = config.RESIZE_HEIGHT
                    prediction_off_screen = (
                        tx_frame < 0 or ty_frame < 0
                        or tx_frame >= frame_w or ty_frame >= frame_h
                    )

                    if prediction_off_screen:
                        print(f"[SKIP] id={target.id} prediccion fuera de frame "
                              f"({tx_frame},{ty_frame}) — pato escapado, cooldown")
                        recently_shot[target.id] = now  # entra en cooldown
                    else:
                        # ─── ESCALAR A GAME ZONE para el click ────────────
                        sx = capturer.scale_x
                        sy = capturer.scale_y
                        tx = int(tx_frame * sx)
                        ty = int(ty_frame * sy)

                        # Clamp a la región real
                        tx = max(0, min(tx, region["width"]  - 1))
                        ty = max(0, min(ty, region["height"] - 1))

                        # ═══ DIAGNOSTICO: log de cada click ═══
                        print(f"[CLICK] id={target.id} frame=({tx_frame},{ty_frame}) "
                              f"-> game=({tx},{ty}) region={region}")

                        move_and_click(tx, ty, region)

                        burst.register_shot(target, now)
                        recently_shot[target.id] = now
                        last_click     = now
                        last_click_pos = (tx_frame, ty_frame)  # en espacio frame

            # ─────────────────────────────────────────────────────────
            # DEBUG
            # ─────────────────────────────────────────────────────────
            if config.DEBUG:

                if not debug_initialized:
                    cv2.namedWindow("Debug", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Debug", config.DEBUG_WIDTH, config.DEBUG_HEIGHT)
                    debug_initialized = True

                if now - last_debug > config.DEBUG_INTERVAL:
                    # Las detecciones y los targets ya están en espacio del
                    # frame (320×320). No hay que escalar nada para dibujar.
                    debug_frame = draw_detections(
                        frame.copy(),
                        detections,
                        active,
                    )

                    burst_info = f"Burst: {burst.shots_fired}/{burst.shots_required}"
                    if burst.current_id is not None:
                        burst_info += f"  ID={burst.current_id}"

                    cap_stats = capturer.get_capture_stats()
                    fps_text = (f"YOLO: {cycle_time_ema*1000:.1f}ms  "
                                f"Loop: {dt*1000:.1f}ms  "
                                f"Cap: {cap_stats['avg_capture_ms']:.0f}ms  "
                                f"Trk: {len(multi_tracker.trackers)}")

                    cv2.putText(debug_frame, fps_text, (8, 18),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.putText(debug_frame, burst_info, (8, 36),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                (0, 255, 255), 1, cv2.LINE_AA)

                    cv2.imshow("Debug", debug_frame)
                    last_debug = now

                if cv2.waitKey(1) & 0xFF == 27:
                    break
            else:
                time.sleep(0.002)

                # Stats por consola
                console_interval = getattr(config, "CONSOLE_STATS_INTERVAL", 0)
                if console_interval > 0 and (now - last_debug) > console_interval:
                    cap_stats = capturer.get_capture_stats()
                    msg = (f"[stats] cap={cap_stats['avg_capture_ms']:.0f}ms  "
                           f"yolo={cycle_time_ema*1000:.0f}ms  "
                           f"loop={dt*1000:.0f}ms  "
                           f"targets={len(active)}")
                    print(msg)
                    last_debug = now

    finally:
        capturer.close()
        if debug_initialized:
            cv2.destroyAllWindows()