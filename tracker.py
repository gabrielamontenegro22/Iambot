import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_TRACKERS = 40


class KalmanTracker:
    _id_counter = 0

    @classmethod
    def reset_ids(cls):
        cls._id_counter = 0

    def __init__(self, x, y, w, h):
        KalmanTracker._id_counter += 1
        self.id = KalmanTracker._id_counter

        self.lost_frames = 0
        self.hit_streak  = 1
        self.confirmed   = True

        self.w = w
        self.h = h

        self.state = np.array([x, y, 0.0, 0.0], dtype=np.float32)
        self.P     = np.eye(4, dtype=np.float32) * 10.0

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float32)

        self.Q = np.eye(4, dtype=np.float32) * 5.0
        self.R = np.eye(2, dtype=np.float32) * 0.5

    def predict(self, dt):
        """Avanza el estado dt segundos en el futuro."""
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float32)

        self.state = F @ self.state
        self.P = F @ self.P @ F.T + self.Q * max(dt, 1e-3)
        return self.bbox

    def update(self, x, y, w, h):
        z     = np.array([x, y], dtype=np.float32)
        innov = z - self.H @ self.state
        S     = self.H @ self.P @ self.H.T + self.R
        K     = self.P @ self.H.T @ np.linalg.inv(S)

        self.state = self.state + K @ innov
        self.P     = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        self.w     = w
        self.h     = h

        self.lost_frames  = 0
        self.hit_streak  += 1
        self.confirmed    = True

    def predict_future(self, lead_seconds):
        """(x, y) extrapolado sin modificar estado interno."""
        cx, cy, vx, vy = self.state
        return (int(cx + vx * lead_seconds),
                int(cy + vy * lead_seconds))

    @property
    def position(self):
        return (int(self.state[0]), int(self.state[1]))

    @property
    def bbox(self):
        cx, cy = int(self.state[0]), int(self.state[1])
        hw, hh = self.w // 2, self.h // 2
        return (cx - hw, cy - hh, cx + hw, cy + hh)

    @property
    def velocity(self):
        return (float(self.state[2]), float(self.state[3]))


# ═══════════════════════════════════════════════════════════════════
# Cost functions
# ═══════════════════════════════════════════════════════════════════

def _iou(b1, b2):
    ix1 = max(b1[0], b2[0]);  iy1 = max(b1[1], b2[1])
    ix2 = min(b1[2], b2[2]);  iy2 = min(b1[3], b2[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    a1    = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2    = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def _hybrid_cost(t_bbox, t_pos, d, max_dist=250.0, alpha=0.4):
    """Combined cost = alpha*(1-IoU) + (1-alpha)*norm_distance"""
    iou    = _iou(t_bbox, (
        d['x'] - d['w'] // 2, d['y'] - d['h'] // 2,
        d['x'] + d['w'] // 2, d['y'] + d['h'] // 2,
    ))
    dist   = np.sqrt((t_pos[0] - d['x'])**2 + (t_pos[1] - d['y'])**2)
    dist_n = min(dist / max_dist, 1.0)
    return alpha * (1.0 - iou) + (1.0 - alpha) * dist_n


# ═══════════════════════════════════════════════════════════════════
# MultiTracker
# ═══════════════════════════════════════════════════════════════════

class MultiTracker:
    """
    Predict en cada iteración del loop (con dt real), update solo cuando
    llegan detecciones nuevas.

    Salvaguardas anti-explosión de IDs:
      - MAX_TRACKERS: límite duro de trackers simultáneos.
      - Warning si se llega al límite (suele indicar bboxes en escala mala).
    """

    def __init__(self, iou_threshold=0.25, max_dist=250, max_lost=8):
        self.trackers      = []
        self.iou_threshold = iou_threshold
        self.max_dist      = max_dist
        self.max_lost      = max_lost
        self._warned_limit = False

    def predict_all(self, dt):
        for t in self.trackers:
            t.predict(dt)

    def update(self, detections):
        if not detections:
            for t in self.trackers:
                t.lost_frames += 1
                # NO resetear hit_streak — preservamos el conteo de detecciones
                # historicas para que el filtro de velocidad pueda aplicar.
                # Antes: t.hit_streak = 0 → blobs estaticos quedaban en hits=0
                # siempre, bypaseando HIT_STREAK_GRACE.
            self._prune()
            return self.active_targets()

        if not self.trackers:
            self._add_new_trackers(detections)
            return self.active_targets()

        # Hybrid cost matrix
        cost = np.zeros(
            (len(self.trackers), len(detections)), dtype=np.float32
        )
        for i, t in enumerate(self.trackers):
            for j, d in enumerate(detections):
                cost[i, j] = _hybrid_cost(
                    t.bbox, t.position, d, max_dist=self.max_dist
                )

        row_ind, col_ind = linear_sum_assignment(cost)

        matched_t = set()
        matched_d = set()

        for r, c in zip(row_ind, col_ind):
            d = detections[c]
            dist = np.sqrt(
                (self.trackers[r].position[0] - d['x'])**2 +
                (self.trackers[r].position[1] - d['y'])**2
            )
            if dist < self.max_dist:
                self.trackers[r].update(d['x'], d['y'], d['w'], d['h'])
                matched_t.add(r)
                matched_d.add(c)

        for i, t in enumerate(self.trackers):
            if i not in matched_t:
                t.lost_frames += 1
                # NO resetear hit_streak — ver comentario en bloque superior.

        unmatched_detections = [
            detections[j] for j in range(len(detections))
            if j not in matched_d
        ]
        self._add_new_trackers(unmatched_detections)

        self._prune()
        return self.active_targets()

    def _add_new_trackers(self, detections):
        """Crea trackers nuevos respetando MAX_TRACKERS."""
        room = MAX_TRACKERS - len(self.trackers)
        if room <= 0:
            if not self._warned_limit:
                print(f"[Tracker] ⚠️  Alcanzado MAX_TRACKERS={MAX_TRACKERS}. "
                      f"Esto suele indicar bboxes en escala incorrecta — "
                      f"el matching falla y todo se crea como nuevo.")
                self._warned_limit = True
            return

        if room > 5:
            self._warned_limit = False

        for d in detections[:room]:
            self.trackers.append(
                KalmanTracker(d['x'], d['y'], d['w'], d['h'])
            )

    def active_targets(self):
        return [t for t in self.trackers if t.confirmed]

    def _prune(self):
        self.trackers = [
            t for t in self.trackers
            if t.lost_frames <= self.max_lost
        ]