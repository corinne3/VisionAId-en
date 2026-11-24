"""Voice announcements of the guidance: what the tool says, and when.

Only four moments, so as not to saturate listening (the path sound remains the guide):

1. **request understood**: « Je cherche la gourde. Tourne lentement sur toi-même. » (I'm looking
   for the bottle. Turn slowly on the spot.) (then the search sound); unknown request:
   « Je n'ai pas compris… » (I didn't understand…);
2. **object found**: where it is in the room and the overall route, path suspended during the
   sentence: « La gourde est au fond à droite, à environ 4 mètres, derrière le canapé, sur la
   table. Le chemin contourne le canapé par la gauche. » (The bottle is at the far right, about
   4 metres away, behind the sofa, on the table. The path goes around the sofa on the left.);
3. **approach** (carrying furniture close, or object less than 1.6 m away), the path sound continues:
   « Tu arrives vers la table, la gourde sera légèrement sur ta gauche. » (You are getting close
   to the table, the bottle will be slightly on your left.);
4. **correction** if the announced object was elsewhere (probable target then confirmed further away).

Directions are egocentric (relative to the user's torso at the time of the
sentence); « au fond » (at the far end) = far ahead. The furniture comes from `mapping.landmarks`.
The spoken sentences stay in French: the users are French speakers.
"""

from __future__ import annotations

import math

import numpy as np

# detector prompt → (name, gender); gender "m", "f" or "e" (elision: l'étagère)
FR_NOUNS = {
    "table": ("table", "f"),
    "coffee table": ("table basse", "f"),
    "desk": ("bureau", "m"),
    "kitchen counter": ("plan de travail", "m"),
    "sofa": ("canapé", "m"),
    "armchair": ("fauteuil", "m"),
    "chair": ("chaise", "f"),
    "bed": ("lit", "m"),
    "shelf": ("étagère", "e"),
    "cabinet": ("meuble", "m"),
    "door": ("porte", "f"),
    "television": ("télévision", "f"),
}
NOT_SUPPORT = {"door", "television"}  # never « sur la porte » (on the door)
NOT_BETWEEN = {"door"}  # an open door is not an obstacle to go around


def noun(label: str) -> tuple[str, str]:
    return FR_NOUNS.get(label, (label, "m"))


def the(label: str) -> str:
    """« la table », « le canapé », « l'étagère » (French article + noun)."""
    n, g = noun(label)
    return f"l'{n}" if g == "e" else (f"la {n}" if g == "f" else f"le {n}")


def of_the(label: str) -> str:
    """« de la table », « du canapé », « de l'étagère » (French "of the …")."""
    n, g = noun(label)
    return f"de l'{n}" if g == "e" else (f"de la {n}" if g == "f" else f"du {n}")


def to_the(label: str) -> str:
    """« vers la table », « vers le canapé » (French "towards the …")."""
    return f"vers {the(label)}"


def cap(s: str) -> str:
    return s[:1].upper() + s[1:] if s else s


def side(az: float) -> str:
    """az > 0 : gauche."""
    return "gauche" if az > 0 else "droite"


def where_phrase(az: float, dist: float, far_m: float) -> str:
    """Egocentric position: « au fond de la pièce », « devant toi, sur ta droite »…"""
    a = abs(math.degrees(az))
    s = side(az)
    far = dist >= far_m
    if a <= 20:
        return "au fond de la pièce, en face de toi" if far else "devant toi"
    if a <= 60:
        return f"au fond à {s}" if far else f"devant toi, un peu sur ta {s}"
    if a <= 120:
        return f"sur ta {s}"
    if a <= 160:
        return f"derrière toi, sur ta {s}"
    return "derrière toi"


def distance_phrase(dist: float) -> str:
    if not math.isfinite(dist):
        return ""
    if dist < 1.2:
        return "tout près"
    n = int(round(dist))
    return f"à environ {n} mètre{'s' if n > 1 else ''}"


def relative_side_phrase(az: float) -> str:
    """Side where the object will be on arrival: « légèrement sur ta gauche »…"""
    a = abs(math.degrees(az))
    s = side(az)
    if a < 10:
        return "juste devant toi"
    if a < 35:
        return f"légèrement sur ta {s}"
    if a < 80:
        return f"sur ta {s}"
    if a < 135:
        return f"à ta {s}, tourne-toi"
    return "derrière toi, fais demi-tour"


def height_phrase(z: float) -> str:
    if not math.isfinite(z):
        return ""
    if z < 0.3:
        return "par terre"
    if z < 0.6:
        return "à hauteur des genoux"
    if z < 1.2:
        return "à hauteur de taille"
    return "en hauteur"


# ------------------------------------------------------------------ geometry
def _ego(T_head: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    from visionaid.guidance.cues import relative_direction

    az, _, d = relative_direction(T_head, np.asarray(p, float))
    return az, d


def _in_box(xy, box, margin: float) -> bool:
    return (
        box[0] - margin <= xy[0] <= box[2] + margin and box[1] - margin <= xy[1] <= box[3] + margin
    )


def _box_dist(xy, box) -> float:
    dx = max(box[0] - xy[0], 0.0, xy[0] - box[2])
    dy = max(box[1] - xy[1], 0.0, xy[1] - box[3])
    return math.hypot(dx, dy)


def _segment_hits_box(a, b, box, margin: float = 0.05, n: int = 40) -> bool:
    for s in np.linspace(0.0, 1.0, n):
        if _in_box(a + s * (b - a), box, margin):
            return True
    return False


def support_of(obj_xyz, lms, margin: float) -> int | None:
    """Index of the furniture carrying the object (footprint containing it, top just below)."""
    if lms is None or not len(lms.labels) or obj_xyz[2] < 0.25:
        return None
    best, best_area = None, np.inf
    for i, lab in enumerate(lms.labels):
        if lab in NOT_SUPPORT:
            continue
        b = lms.boxes[i]
        top = float(lms.top_z[i])
        if _in_box(obj_xyz[:2], b, margin) and obj_xyz[2] - 0.5 <= top <= obj_xyz[2] + 0.1:
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area < best_area:  # the smallest furniture containing it (table rather than room)
                best, best_area = i, area
    return best


def nearest_landmark(obj_xyz, lms, max_d: float, exclude=()) -> int | None:
    if lms is None or not len(lms.labels):
        return None
    best, best_d = None, max_d
    for i, lab in enumerate(lms.labels):
        if i in exclude or lab in NOT_SUPPORT:
            continue
        d = _box_dist(obj_xyz[:2], lms.boxes[i])
        if d < best_d:
            best, best_d = i, d
    return best


def between_landmark(user_xy, obj_xy, lms, exclude=()) -> int | None:
    """Furniture between the user and the object (the object is "behind" it): the largest one."""
    if lms is None or not len(lms.labels):
        return None
    u, o = np.asarray(user_xy, float), np.asarray(obj_xy, float)
    d_uo = float(np.linalg.norm(o - u))
    best, best_area = None, 0.15  # at least 0.15 m²: not a chair seen side-on
    for i, lab in enumerate(lms.labels):
        if i in exclude or lab in NOT_BETWEEN:
            continue
        b = lms.boxes[i]
        if _in_box(u, b, 0.1) or _box_dist(o, b) < 0.25:
            continue  # the user is on it, or the object is against it (it is its carrier)
        c = lms.centers[i]
        if np.linalg.norm(c - u) >= d_uo:
            continue
        if _segment_hits_box(u, o, b):
            area = (b[2] - b[0]) * (b[3] - b[1])
            if area > best_area:
                best, best_area = i, area
    return best


def pass_side(path_pts: np.ndarray, center: np.ndarray) -> str | None:
    """Side by which the path passes the furniture: "left" if the furniture is on its right."""
    if path_pts is None or len(path_pts) < 3:
        return None
    k = int(np.argmin(np.linalg.norm(path_pts - center, axis=1)))
    k = min(max(k, 1), len(path_pts) - 2)
    d = path_pts[k + 1] - path_pts[k - 1]
    v = center - path_pts[k]
    cross = d[0] * v[1] - d[1] * v[0]
    if abs(cross) < 1e-6:
        return None
    return "droite" if cross > 0 else "gauche"  # furniture on the left (cross > 0): we pass on the right


# ------------------------------------------------------------------ sentence queue
class SpeechQueue:
    """Sentences to speak, one at a time: each stays `hold` seconds on the instructions (message
    losses), with a unique number (the phone only speaks each number once)."""

    def __init__(self, hold: float = 1.0) -> None:
        self.hold = float(hold)
        self.queue: list[tuple[str, bool]] = []
        self.cur: tuple[int, str, bool, float] | None = None
        self.next_id = 1
        self.history: list[tuple[float, str]] = []

    def say(self, text: str, duck: bool = False) -> None:
        if text:
            self.queue.append((text, duck))

    def current(self, t: float) -> tuple[str, int, bool]:
        """Sentence to be carried by the instruction at time t: (text, id, suspends the path)."""
        if self.cur is not None and (t - self.cur[3] >= self.hold or t < self.cur[3]):
            self.cur = None
        if self.cur is None and self.queue:
            text, duck = self.queue.pop(0)
            self.cur = (self.next_id, text, duck, t)
            self.next_id += 1
            self.history.append((t, text))
        if self.cur is None:
            return "", 0, False
        return self.cur[1], self.cur[0], self.cur[2]


# ------------------------------------------------------------------ narrateur
class Narrator:
    """Decides which sentences to speak; `current(t)` gives the sentence currently being played."""

    def __init__(
        self,
        cfg,
        spoken: str = "",
        plural: bool = False,
        announce_start: bool = True,
        speech: SpeechQueue | None = None,
    ):
        n = getattr(cfg, "narration", None)
        g = lambda k, d: float(getattr(n, k, d)) if n is not None else d  # noqa: E731
        self.enabled = bool(getattr(n, "enabled", True)) if n is not None else True
        self.wait_path = g("wait_path_s", 2.5)
        self.far_m = g("far_m", 3.0)
        self.approach_m = g("approach_m", 1.6)
        self.approach_support_m = g("approach_support_m", 1.0)
        self.approach_max = g("approach_max_m", 2.5)
        self.moved_m = g("moved_m", 1.0)
        self.margin = g("support_margin_m", 0.15)
        self.near_m = g("near_m", 0.8)
        tts = getattr(cfg, "tts", None)
        self.hold = float(getattr(tts, "hold_s", 1.0)) if tts is not None else 1.0
        if not spoken:
            t = getattr(cfg, "target", None)
            spoken = str(getattr(t, "spoken", "") or "")
            if not spoken:
                from visionaid.perception.target import spoken_from_label

                lab = str(getattr(t, "label", "") or "")
                spoken = (
                    spoken_from_label(lab) if lab else f"l'objet {getattr(t, 'name', '')}".strip()
                )
        self.spoken = spoken
        self.plural = plural
        self.out = speech if speech is not None else SpeechQueue(self.hold)
        self.t_known: float | None = None
        self.found_pos: np.ndarray | None = None
        self.found_said = False
        self.approach_said = False
        self.t_correction = -math.inf
        self.start_pending = announce_start

    # -------------------------------------------------------------- sentence queue
    def say(self, text: str, duck: bool = False) -> None:
        self.out.say(text, duck)

    def current(self, t: float) -> tuple[str, int, bool]:
        return self.out.current(t)

    @property
    def history(self) -> list[tuple[float, str]]:
        return self.out.history

    # -------------------------------------------------------------- phrases
    def _verb(self) -> str:
        return "sont" if self.plural else "est"

    def start_phrase(self) -> str:
        return f"Je cherche {self.spoken}. Tourne lentement sur toi-même."

    def found_phrase(self, T_head, obj_xyz, path, cm, lms, candidate: bool = False) -> str:
        user = T_head[:2, 3]
        az, dist = _ego(T_head, obj_xyz[:2])
        parts = [where_phrase(az, dist, self.far_m)]
        dp = distance_phrase(dist)
        if dp:
            parts.append(dp)
        sup = support_of(obj_xyz, lms, self.margin)
        excl = () if sup is None else (sup,)
        btw = between_landmark(user, obj_xyz[:2], lms, excl)
        if btw is not None:
            parts.append(f"derrière {the(lms.labels[btw])}")
        if sup is not None:
            parts.append(f"sur {the(lms.labels[sup])}")
        else:
            near = nearest_landmark(
                obj_xyz, lms, self.near_m, excl + ((btw,) if btw is not None else ())
            )
            z = float(obj_xyz[2])
            if z < 0.3:
                parts.append(
                    "par terre" + (f", près {of_the(lms.labels[near])}" if near is not None else "")
                )
            elif near is not None:
                parts.append(f"près {of_the(lms.labels[near])}, {height_phrase(z)}")
            elif _occupied_below(cm, obj_xyz):
                parts.append(f"sur un meuble, {height_phrase(z)}")
            else:
                parts.append(height_phrase(z))
        subj = cap(self.spoken)
        lead = (
            f"Je crois que {self.spoken} {self._verb()}" if candidate else f"{subj} {self._verb()}"
        )
        s1 = f"{lead} {', '.join(p for p in parts if p)}."
        return s1 + " " + self.route_phrase(T_head, obj_xyz, path, lms, btw)

    def route_phrase(self, T_head, obj_xyz, path, lms, btw) -> str:
        pts = None if path is None or not path.valid else np.asarray(path.points, float)
        if pts is None or len(pts) < 2:
            return "Suis le son."
        if btw is not None:
            s = pass_side(pts, lms.centers[btw])
            if s is not None:
                return f"Le chemin contourne {the(lms.labels[btw])} par la {s}, suis le son."
        partial = np.linalg.norm(np.asarray(path.goal) - obj_xyz[:2]) > 0.9
        # starting direction: point of the path at ~1 m
        d = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
        k = int(np.searchsorted(d, min(1.0, d[-1])))
        az, _ = _ego(T_head, pts[min(k, len(pts) - 1)])
        a = abs(math.degrees(az))
        if a < 25:
            first = "Va tout droit"
        elif a < 120:
            first = f"Tourne vers la {side(az)}"
        else:
            first = "Fais demi-tour"
        straight = float(np.linalg.norm(obj_xyz[:2] - T_head[:2, 3]))
        if partial:
            return f"{first}, le chemin se précisera en avançant. Suis le son."
        if path.length > 1.4 * straight + 0.5:
            return f"{first}, le chemin fait un détour. Suis le son."
        return f"{first} et suis le son."

    def approach_phrase(self, T_head, obj_xyz, path, lms) -> str:
        sup = support_of(obj_xyz, lms, self.margin)
        # side of the object when we reach the end of the path, facing the arrival direction
        az = None
        if path is not None and path.valid and len(path.points) >= 2:
            pts = np.asarray(path.points, float)
            goal = pts[-1]
            # point of the path 0.7 m before the end (curvilinear abscissa)
            seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            cum = np.r_[0.0, np.cumsum(seg)]
            sb = max(cum[-1] - 0.7, 0.0)
            back = np.array([np.interp(sb, cum, pts[:, 0]), np.interp(sb, cum, pts[:, 1])])
            fwd = goal - back
            if np.linalg.norm(fwd) < 0.2:
                fwd = goal - T_head[:2, 3]
            v = obj_xyz[:2] - goal
            if np.linalg.norm(fwd) > 0.2 and np.linalg.norm(v) > 0.15:
                az = math.atan2(fwd[0] * v[1] - fwd[1] * v[0], fwd @ v)
        if az is None:
            az, _ = _ego(T_head, obj_xyz[:2])
        where = relative_side_phrase(az)
        if sup is not None:
            fut = "seront" if self.plural else "sera"
            return f"Tu arrives {to_the(lms.labels[sup])}, {self.spoken} {fut} {where}."
        h = height_phrase(float(obj_xyz[2]))
        return f"Tu arrives, {self.spoken} {self._verb()} {where}" + (f", {h}." if h else ".")

    # -------------------------------------------------------------- decision
    def update(self, t, state, T_head, obj, path, cm, lms, dist: float, obst_d: float) -> None:
        if not self.enabled:
            return
        if self.start_pending:
            self.start_pending = False
            self.say(self.start_phrase(), duck=True)
        if T_head is None or state == "TRACKING_LOST":
            return
        known = obj is not None and obj.state not in ("unknown", "stale")
        known = known and np.isfinite(obj.position).all()
        if not known:
            return
        pos = np.asarray(obj.position, float)
        if self.t_known is None:
            self.t_known = t
        path_ok = (
            path is not None
            and path.valid
            and getattr(path, "target_id", 0) == getattr(obj, "target_id", 0)
            and len(path.points) >= 2
        )
        if not self.found_said:
            if path_ok or t - self.t_known >= self.wait_path:
                self.found_said = True
                self.found_pos = pos.copy()
                self.say(
                    self.found_phrase(
                        T_head, pos, path if path_ok else None, cm, lms, obj.state == "candidate"
                    ),
                    duck=True,
                )
            return
        # object announced elsewhere (false probable target): correction, once every 8 s
        if (
            np.linalg.norm(pos[:2] - self.found_pos[:2]) > self.moved_m
            and t - self.t_correction > 8.0
            and not self.approach_said
        ):
            self.t_correction = t
            self.found_pos = pos.copy()
            self.say(
                "Correction. " + self.found_phrase(T_head, pos, path if path_ok else None, cm, lms),
                duck=True,
            )
            return
        if self.approach_said or state == "ARRIVED":
            if state == "ARRIVED":
                self.approach_said = True  # arrived without announcement: the arrival sound is enough
            return
        sup = support_of(pos, lms, self.margin)
        near_support = (
            sup is not None
            and dist <= self.approach_max
            and _box_dist(T_head[:2, 3], lms.boxes[sup]) <= self.approach_support_m
        )
        if dist <= self.approach_m or near_support:
            self.approach_said = True
            self.say(self.approach_phrase(T_head, pos, path if path_ok else None, lms))


def _occupied_below(cm, obj_xyz, r: float = 0.4) -> bool:
    """Occupied cells around the object (unrecognised furniture under it)?"""
    if cm is None:
        return False
    from visionaid.mapping.costmap import OCC

    res = float(cm.resolution)
    j = int((obj_xyz[0] - cm.origin[0]) / res)
    i = int((obj_xyz[1] - cm.origin[1]) / res)
    k = int(r / res)
    h, w = cm.state.shape
    a0, a1, b0, b1 = max(i - k, 0), min(i + k + 1, h), max(j - k, 0), min(j + k + 1, w)
    if a0 >= a1 or b0 >= b1:
        return False
    return bool((cm.state[a0:a1, b0:b1] == OCC).any())
