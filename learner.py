"""
Sistema de aprendizaje adaptativo basado en detección visual de muerte.

CÓMO DETECTA UN HIT:
  Tras un disparo, se guarda un "patch" del frame en la posición del target
  (el recorte del pato). Durante HIT_GRACE_PERIOD se compara:

    A) ¿Bajó la confianza de YOLO sobre ese target?      (modelo "duda")
    B) ¿Cambió mucho el color promedio del patch?        (animación de muerte)
    C) ¿Desapareció del tracker?                         (señal clásica)

  Si A, B o C cumplen sus umbrales → HIT.
  Si pasa el grace period sin que ninguna se cumpla → MISS.

QUÉ APRENDE:
  Solo ajusta TARGET_COOLDOWN. Si la tasa de hits es baja,
  posiblemente el cooldown es muy corto (le pega al mismo pato
  antes de que YOLO se entere) o muy largo (deja escapar patos).

PERSISTENCIA:
  Estado en JSON, escritura atómica, recargable entre sesiones.
"""

import json
import os
import time
import random
from collections import deque

import numpy as np
import cv2


# ════════════════════════════════════════════════════════════════════
# Captura del "antes" — patch visual de un target
# ════════════════════════════════════════════════════════════════════

def _extract_patch(frame, bbox, padding=4):
    """
    Recorta el bbox del frame con un padding pequeño.
    bbox en coords del frame (mismo espacio que YOLO procesa).
    Devuelve un ndarray BGR pequeño, o None si bbox inválido.
    """
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = bbox
    x1 = max(0, int(x1) - padding)
    y1 = max(0, int(y1) - padding)
    x2 = min(w, int(x2) + padding)
    y2 = min(h, int(y2) + padding)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2].copy()


def _patch_signature(patch):
    """
    Resume un patch en un vector pequeño para comparar 'antes' vs 'después'.
    Usamos el color promedio en HSV — invariante a cambios pequeños de
    iluminación pero sensible a cambios grandes de color (animación de muerte).
    """
    if patch is None or patch.size == 0:
        return None
    # Reducir antes de pasar a HSV para acelerar
    small = cv2.resize(patch, (16, 16), interpolation=cv2.INTER_AREA)
    hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    # Promedios separados por canal
    return hsv.reshape(-1, 3).mean(axis=0)   # shape (3,)


def _signatures_distance(sig_a, sig_b):
    """
    Distancia entre dos firmas. Tratamos H (matiz) circular en [0, 180]
    porque el espacio HSV de OpenCV usa ese rango.
    """
    if sig_a is None or sig_b is None:
        return 0.0
    h_diff = abs(sig_a[0] - sig_b[0])
    h_diff = min(h_diff, 180 - h_diff)   # circular
    s_diff = abs(sig_a[1] - sig_b[1])
    v_diff = abs(sig_a[2] - sig_b[2])
    # Pondera matiz más fuerte (cambio de color es la señal clave)
    return 1.5 * h_diff + s_diff + v_diff


# ════════════════════════════════════════════════════════════════════
# AdaptiveLearner
# ════════════════════════════════════════════════════════════════════

class AdaptiveLearner:
    """
    Aprende ajustando solo TARGET_COOLDOWN basado en tasa de hits.

    Hits detectados por:
      - Cambio de color del patch del target (animación de muerte)
      - Caída de confidence del modelo
      - Desaparición del tracker
    """

    def __init__(
        self,
        hit_grace_period   = 0.5,
        window_size        = 30,
        explore_prob       = 0.15,
        persist_path       = None,
        initial_cooldown_s = 0.4,
        # Umbrales para detectar hit
        color_change_threshold = 18.0,   # distancia HSV ponderada
        conf_drop_threshold    = 0.15,   # caída absoluta de confidence
    ):
        self.hit_grace_period       = hit_grace_period
        self.window_size            = window_size
        self.explore_prob           = explore_prob
        self.persist_path           = persist_path
        self.color_change_threshold = color_change_threshold
        self.conf_drop_threshold    = conf_drop_threshold

        # Estado adaptativo — solo cooldown
        self.cooldown_s     = float(initial_cooldown_s)
        self.cooldown_min   = 0.1
        self.cooldown_max   = 1.5

        # Pending shots — esperan evaluación
        # cada entrada: {target_id, fired_at, sig_before, conf_before, evaluated}
        self.shots_pending = []

        # Resultados recientes (1=hit, 0=miss)
        self.results = deque(maxlen=window_size)

        # Estadísticas globales
        self.total_shots = 0
        self.total_hits  = 0

        # Hill-climbing: dirección actual del cambio
        self.last_cooldown_delta = 0.05

        self._prev_hit_rate     = 0.5
        self._last_change_time  = 0.0
        self._evaluation_period = 2.0

        self._rng = random.Random()

        # Diagnóstico — contar por tipo de señal cuál disparó cada hit
        self.hits_by_color    = 0
        self.hits_by_conf     = 0
        self.hits_by_vanish   = 0

        if persist_path:
            self._load()

    # ════════════════════════════════════════════════════════════════
    # API
    # ════════════════════════════════════════════════════════════════

    def get_cooldown_seconds(self):
        return self.cooldown_s

    def register_shot(self, target_id, target_bbox, target_conf, frame, now):
        """
        Llamar JUSTO después de cada click.

        target_bbox: (x1,y1,x2,y2) en coords del FRAME (espacio YOLO).
        target_conf: confidence YOLO del target en su última detección.
        frame:       frame BGR actual (mismo que pasaste a YOLO).
        """
        patch = _extract_patch(frame, target_bbox)
        sig   = _patch_signature(patch)

        self.shots_pending.append({
            "target_id":   target_id,
            "fired_at":    now,
            "sig_before":  sig,
            "conf_before": float(target_conf),
        })
        self.total_shots += 1

    def evaluate_pending(self, active_targets_info, frame, now):
        """
        Llamar tras cada update del tracker.

        active_targets_info: dict {target_id: {"bbox": (..), "conf": float}}
                             con la info más reciente de cada target vivo.
        frame:               frame BGR más reciente.

        Evalúa los pending shots cuyo grace_period haya vencido.
        """
        if not self.shots_pending:
            return 0

        still_pending = []
        evaluated     = 0

        for shot in self.shots_pending:
            elapsed = now - shot["fired_at"]
            tid     = shot["target_id"]

            # ─── ¿Detectamos hit ANTES del grace period? ─────────────
            # Si ya hay señal clara, no hay que esperar.
            early_hit = self._check_hit_signals(shot, active_targets_info, frame)

            if early_hit is not None:
                # early_hit es el "tipo" de señal: "color" / "conf" / "vanish"
                self._record_hit(early_hit)
                evaluated += 1
                continue

            # ─── Aún dentro del grace period — sigue pendiente ───────
            if elapsed < self.hit_grace_period:
                still_pending.append(shot)
                continue

            # ─── Grace venció sin señal de hit → MISS ────────────────
            self._record_result(False)
            evaluated += 1

        self.shots_pending = still_pending
        return evaluated

    def update(self, now):
        """Llamar periódicamente para que ajuste el cooldown."""
        if now - self._last_change_time < self._evaluation_period:
            return
        if len(self.results) < 5:
            return

        current_rate = self.hit_rate()
        improved     = current_rate >= self._prev_hit_rate

        if not improved:
            self.last_cooldown_delta = -self.last_cooldown_delta

        if self._rng.random() < self.explore_prob:
            cooldown_change = self._rng.uniform(-0.05, 0.05)
        else:
            cooldown_change = self.last_cooldown_delta

        self.cooldown_s = max(
            self.cooldown_min,
            min(self.cooldown_max, self.cooldown_s + cooldown_change)
        )

        self._prev_hit_rate    = current_rate
        self._last_change_time = now

        if self.persist_path:
            self._save()

    def hit_rate(self):
        if not self.results:
            return 0.0
        return sum(self.results) / len(self.results)

    def stats_string(self):
        return (f"hits {self.total_hits}/{self.total_shots} "
                f"({self.hit_rate()*100:.0f}%) "
                f"cd={self.cooldown_s*1000:.0f}ms "
                f"[c{self.hits_by_color} k{self.hits_by_conf} v{self.hits_by_vanish}]")

    def reset(self):
        self.shots_pending.clear()
        self.results.clear()
        self.total_shots = 0
        self.total_hits  = 0
        self.hits_by_color = 0
        self.hits_by_conf  = 0
        self.hits_by_vanish = 0
        if self.persist_path and os.path.exists(self.persist_path):
            try:
                os.remove(self.persist_path)
            except Exception:
                pass

    # ════════════════════════════════════════════════════════════════
    # Internos — detección de hit
    # ════════════════════════════════════════════════════════════════

    def _check_hit_signals(self, shot, active_targets_info, frame):
        """
        Devuelve "color" / "conf" / "vanish" si hay señal de hit;
        None si aún no hay evidencia.
        """
        tid = shot["target_id"]

        # Señal C: target desapareció del tracker
        if tid not in active_targets_info:
            # Solo lo cuentas como "vanish" si pasó al menos un poquito
            # de tiempo — desaparición instantánea suele ser ruido.
            if (time.time() - shot["fired_at"]) > 0.1:
                return "vanish"
            return None

        target_info = active_targets_info[tid]

        # Señal B: caída de confidence
        conf_now  = target_info.get("conf", shot["conf_before"])
        conf_drop = shot["conf_before"] - conf_now
        if conf_drop >= self.conf_drop_threshold:
            return "conf"

        # Señal A: cambio de color del patch
        patch_now = _extract_patch(frame, target_info["bbox"])
        sig_now   = _patch_signature(patch_now)
        dist      = _signatures_distance(shot["sig_before"], sig_now)
        if dist >= self.color_change_threshold:
            return "color"

        return None

    def _record_hit(self, hit_type):
        self._record_result(True)
        if hit_type == "color":
            self.hits_by_color += 1
        elif hit_type == "conf":
            self.hits_by_conf += 1
        elif hit_type == "vanish":
            self.hits_by_vanish += 1

    def _record_result(self, hit):
        self.results.append(1 if hit else 0)
        if hit:
            self.total_hits += 1

    # ════════════════════════════════════════════════════════════════
    # Internos — persistencia
    # ════════════════════════════════════════════════════════════════

    def _save(self):
        try:
            data = {
                "cooldown_s":          self.cooldown_s,
                "total_shots":         self.total_shots,
                "total_hits":          self.total_hits,
                "hits_by_color":       self.hits_by_color,
                "hits_by_conf":        self.hits_by_conf,
                "hits_by_vanish":      self.hits_by_vanish,
                "last_cooldown_delta": self.last_cooldown_delta,
                "prev_hit_rate":       self._prev_hit_rate,
                "recent_results":      list(self.results),
                "saved_at":            time.time(),
            }
            tmp = self.persist_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(data, f, indent=2)
            os.replace(tmp, self.persist_path)
        except Exception as e:
            print(f"[Learner] No se pudo guardar: {e}")

    def _load(self):
        if not os.path.exists(self.persist_path):
            return
        try:
            with open(self.persist_path, "r") as f:
                data = json.load(f)

            self.cooldown_s          = float(data.get("cooldown_s", self.cooldown_s))
            self.total_shots         = int(data.get("total_shots", 0))
            self.total_hits          = int(data.get("total_hits", 0))
            self.hits_by_color       = int(data.get("hits_by_color", 0))
            self.hits_by_conf        = int(data.get("hits_by_conf", 0))
            self.hits_by_vanish      = int(data.get("hits_by_vanish", 0))
            self.last_cooldown_delta = float(data.get("last_cooldown_delta", 0.05))
            self._prev_hit_rate      = float(data.get("prev_hit_rate", 0.5))

            recent = data.get("recent_results", [])
            for r in recent[-self.window_size:]:
                self.results.append(int(r))

            print(f"[Learner] Cargado: {self.stats_string()}")
        except Exception as e:
            print(f"[Learner] No se pudo cargar (empezando limpio): {e}")