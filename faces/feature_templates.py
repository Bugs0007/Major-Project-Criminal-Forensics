import cv2
import numpy as np
from typing import Dict, List, Tuple
import os

# ---------------------------------------------------------------------------
# Supersampling factor: all coordinates are multiplied by _K, drawn with
# cv2.LINE_AA, then the final image is down-scaled with INTER_AREA.
# This produces smooth, naturally anti-aliased pencil-sketch lines.
# ---------------------------------------------------------------------------
_K = 4


def _catmull_rom(pts, n_interp=80):
    """Interpolate through *pts* using Catmull-Rom splines."""
    pts = list(pts)
    if len(pts) < 2:
        return pts
    # Pad with virtual start/end control points
    pts = [pts[0]] + pts + [pts[-1]]
    out = []
    for i in range(1, len(pts) - 2):
        p0, p1, p2, p3 = pts[i - 1], pts[i], pts[i + 1], pts[i + 2]
        seg = max(n_interp // (len(pts) - 3), 6)
        for t_i in range(seg):
            t = t_i / seg
            t2, t3 = t * t, t * t * t
            x = 0.5 * ((2 * p1[0]) +
                        (-p0[0] + p2[0]) * t +
                        (2 * p0[0] - 5 * p1[0] + 4 * p2[0] - p3[0]) * t2 +
                        (-p0[0] + 3 * p1[0] - 3 * p2[0] + p3[0]) * t3)
            y = 0.5 * ((2 * p1[1]) +
                        (-p0[1] + p2[1]) * t +
                        (2 * p0[1] - 5 * p1[1] + 4 * p2[1] - p3[1]) * t2 +
                        (-p0[1] + 3 * p1[1] - 3 * p2[1] + p3[1]) * t3)
            out.append((x, y))
    out.append(pts[-2])
    return out


class FeatureTemplateGenerator:
    """Generate realistic pencil-sketch feature templates."""

    # ── scaled drawing primitives ──────────────────────────────────────────

    @staticmethod
    def _p(x, y):
        return (int(round(x * _K)), int(round(y * _K)))

    @staticmethod
    def _r(v):
        return int(round(v * _K))

    @staticmethod
    def _thick(v):
        return max(1, int(round(v * _K * 0.55)))

    @staticmethod
    def _canvas(w, h):
        return np.zeros((h * _K, w * _K, 4), dtype=np.uint8)

    @staticmethod
    def _finalize(img, w, h):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

    @classmethod
    def _spline(cls, img, pts, color=(0, 0, 0, 220), thick=2, closed=False):
        """Draw a smooth Catmull-Rom spline through *pts*."""
        smooth = _catmull_rom(pts, 120)
        a = np.array([cls._p(x, y) for x, y in smooth], np.int32)
        cv2.polylines(img, [a], closed, color, cls._thick(thick), cv2.LINE_AA)

    @classmethod
    def _poly(cls, img, pts, color=(0, 0, 0, 220), thick=2, closed=False):
        a = np.array([cls._p(x, y) for x, y in pts], np.int32)
        cv2.polylines(img, [a], closed, color, cls._thick(thick), cv2.LINE_AA)

    @classmethod
    def _ln(cls, img, x1, y1, x2, y2, color=(0, 0, 0, 220), thick=2):
        cv2.line(img, cls._p(x1, y1), cls._p(x2, y2), color,
                 cls._thick(thick), cv2.LINE_AA)

    @classmethod
    def _ell(cls, img, cx, cy, rx, ry, ang=0, sa=0, ea=360,
             color=(0, 0, 0, 220), thick=2, fill=False):
        t = -1 if fill else cls._thick(thick)
        cv2.ellipse(img, cls._p(cx, cy), (cls._r(rx), cls._r(ry)),
                    ang, sa, ea, color, t, cv2.LINE_AA)

    @classmethod
    def _cir(cls, img, cx, cy, r, color=(0, 0, 0, 220), thick=2, fill=False):
        t = -1 if fill else cls._thick(thick)
        cv2.circle(img, cls._p(cx, cy), cls._r(r), color, t, cv2.LINE_AA)

    @classmethod
    def _fpoly(cls, img, pts, color=(0, 0, 0, 80)):
        a = np.array([cls._p(x, y) for x, y in pts], np.int32)
        cv2.fillPoly(img, [a], color, cv2.LINE_AA)

    @classmethod
    def _fspline(cls, img, pts, color=(0, 0, 0, 80)):
        """Fill a smooth Catmull-Rom region."""
        smooth = _catmull_rom(pts, 120)
        a = np.array([cls._p(x, y) for x, y in smooth], np.int32)
        cv2.fillPoly(img, [a], color, cv2.LINE_AA)

    # ── eye templates ──────────────────────────────────────────────────────

    @classmethod
    def _draw_eye(cls, img, cx, cy, style='almond'):
        """Draw one detailed eye at (cx, cy)."""

        if style == 'almond':
            upper = [
                (cx - 38, cy + 2), (cx - 28, cy - 6), (cx - 16, cy - 13),
                (cx, cy - 16), (cx + 16, cy - 13), (cx + 28, cy - 6),
                (cx + 38, cy + 2),
            ]
            lower = [
                (cx - 38, cy + 2), (cx - 24, cy + 10), (cx - 10, cy + 14),
                (cx, cy + 15), (cx + 10, cy + 14), (cx + 24, cy + 10),
                (cx + 38, cy + 2),
            ]
            crease = [
                (cx - 34, cy - 8), (cx - 20, cy - 19), (cx - 6, cy - 22),
                (cx, cy - 23), (cx + 6, cy - 22), (cx + 20, cy - 18),
                (cx + 34, cy - 10),
            ]
            lid_amp = 16
        elif style == 'round':
            upper = [
                (cx - 34, cy + 2), (cx - 24, cy - 10), (cx - 12, cy - 18),
                (cx, cy - 20), (cx + 12, cy - 18), (cx + 24, cy - 10),
                (cx + 34, cy + 2),
            ]
            lower = [
                (cx - 34, cy + 2), (cx - 22, cy + 13), (cx - 10, cy + 18),
                (cx, cy + 20), (cx + 10, cy + 18), (cx + 22, cy + 13),
                (cx + 34, cy + 2),
            ]
            crease = [
                (cx - 30, cy - 8), (cx - 16, cy - 23), (cx, cy - 27),
                (cx + 16, cy - 23), (cx + 30, cy - 8),
            ]
            lid_amp = 20
        else:  # narrow
            upper = [
                (cx - 40, cy + 1), (cx - 28, cy - 3), (cx - 14, cy - 7),
                (cx, cy - 8), (cx + 14, cy - 7), (cx + 28, cy - 3),
                (cx + 40, cy + 1),
            ]
            lower = [
                (cx - 40, cy + 1), (cx - 26, cy + 5), (cx - 12, cy + 8),
                (cx, cy + 9), (cx + 12, cy + 8), (cx + 26, cy + 5),
                (cx + 40, cy + 1),
            ]
            crease = [
                (cx - 36, cy - 2), (cx - 22, cy - 9), (cx - 8, cy - 11),
                (cx, cy - 11), (cx + 8, cy - 11), (cx + 22, cy - 9),
                (cx + 36, cy - 3),
            ]
            lid_amp = 8

        # Fill eye white
        eye_shape = upper + lower[::-1]
        cls._fspline(img, eye_shape, color=(255, 255, 255, 60))

        # Draw lid outlines with smooth splines
        cls._spline(img, upper, thick=2.8, color=(0, 0, 0, 240))
        cls._spline(img, lower, thick=1.6, color=(0, 0, 0, 160))
        cls._spline(img, crease, thick=1.0, color=(0, 0, 0, 75))

        # --- iris ---
        iris_r = 12 if style == 'round' else 10
        # Iris fill (light gray)
        cls._cir(img, cx, cy, iris_r, fill=True, color=(0, 0, 0, 40))
        # Iris outline
        cls._cir(img, cx, cy, iris_r, thick=1.8, color=(0, 0, 0, 200))
        # Radial iris texture
        for ang in range(0, 360, 12):
            rad = np.radians(ang)
            r1, r2 = 5, iris_r - 1.5
            alpha = 35 + (ang % 36)
            cls._ln(img, cx + r1 * np.cos(rad), cy + r1 * np.sin(rad),
                    cx + r2 * np.cos(rad), cy + r2 * np.sin(rad),
                    color=(0, 0, 0, alpha), thick=0.5)

        # --- pupil + highlight ---
        cls._cir(img, cx, cy, 5, fill=True, color=(0, 0, 0, 245))
        cls._cir(img, cx + 2.5, cy - 2.5, 2, fill=True,
                 color=(255, 255, 255, 230))
        cls._cir(img, cx - 1.5, cy + 1.5, 1, fill=True,
                 color=(255, 255, 255, 140))

        # --- inner corner (tear duct) ---
        cls._spline(img, [
            (cx - 38, cy + 4), (cx - 42, cy + 2), (cx - 38, cy + 0),
        ], thick=1.0, color=(0, 0, 0, 120))

        # --- outer corner ---
        cls._ln(img, cx + 38, cy + 2, cx + 40, cy + 1,
                thick=0.8, color=(0, 0, 0, 100))

        # --- upper lashes ---
        half_w = 34
        for i in range(16):
            lx = cx - half_w + i * (2 * half_w / 15)
            t_param = (lx - (cx - 38)) / 76.0
            t_param = max(0, min(1, t_param))
            lid_y = cy - lid_amp * np.sin(t_param * np.pi) + \
                    2 * (1 - np.sin(t_param * np.pi))
            lash_len = 3 + 3 * np.sin(t_param * np.pi)
            fan_angle = 80 + (t_param - 0.5) * 30
            rad = np.radians(fan_angle)
            cls._ln(img, lx, lid_y,
                    lx + lash_len * np.cos(rad),
                    lid_y - lash_len * np.sin(rad),
                    thick=0.8, color=(0, 0, 0, 200))

    @classmethod
    def generate_eye_templates(cls) -> List[np.ndarray]:
        eyes = []
        w, h = 260, 80
        gap = 60
        for style in ['almond', 'round', 'narrow']:
            img = cls._canvas(w, h)
            lcx = w // 2 - gap // 2 - 10
            rcx = w // 2 + gap // 2 + 10
            cls._draw_eye(img, lcx, h // 2, style)
            cls._draw_eye(img, rcx, h // 2, style)
            eyes.append(cls._finalize(img, w, h))
        return eyes

    # ── nose templates ─────────────────────────────────────────────────────

    @classmethod
    def generate_nose_templates(cls) -> List[np.ndarray]:
        noses = []
        w, h = 70, 100

        # --- Straight nose ---
        img = cls._canvas(w, h)
        # Light bridge lines
        cls._spline(img, [(32, 10), (31, 25), (30, 42), (28, 58), (26, 68)],
                    thick=1.3, color=(0, 0, 0, 110))
        cls._spline(img, [(38, 10), (39, 25), (40, 42), (42, 58), (44, 68)],
                    thick=1.3, color=(0, 0, 0, 110))
        # Nose tip / ball
        cls._spline(img, [
            (26, 68), (24, 74), (26, 80), (31, 84), (35, 86),
            (39, 84), (44, 80), (46, 74), (44, 68),
        ], thick=2.2, color=(0, 0, 0, 200))
        # Nostrils
        cls._ell(img, 29, 81, 7, 4.5, 20, 40, 200,
                 color=(0, 0, 0, 180), thick=1.5)
        cls._ell(img, 41, 81, 7, 4.5, 160, -20, 140,
                 color=(0, 0, 0, 180), thick=1.5)
        # Nostril shading
        cls._fspline(img, [(25, 79), (28, 84), (33, 83), (30, 78)],
                     color=(0, 0, 0, 50))
        cls._fspline(img, [(45, 79), (42, 84), (37, 83), (40, 78)],
                     color=(0, 0, 0, 50))
        noses.append(cls._finalize(img, w, h))

        # --- Button / upturned nose ---
        img = cls._canvas(w, h)
        cls._spline(img, [(33, 18), (32, 32), (31, 48)],
                    thick=1.0, color=(0, 0, 0, 90))
        cls._spline(img, [(37, 18), (38, 32), (39, 48)],
                    thick=1.0, color=(0, 0, 0, 90))
        # Wider, rounder tip
        cls._spline(img, [
            (24, 58), (21, 66), (23, 74), (29, 79), (35, 81),
            (41, 79), (47, 74), (49, 66), (46, 58),
        ], thick=2.2, color=(0, 0, 0, 200))
        # Upturned curve across tip
        cls._spline(img, [(26, 68), (31, 72), (35, 73), (39, 72), (44, 68)],
                    thick=1.5, color=(0, 0, 0, 150))
        # Nostrils
        cls._ell(img, 29, 75, 6, 4, 30, 30, 210,
                 color=(0, 0, 0, 170), thick=1.3)
        cls._ell(img, 41, 75, 6, 4, 150, -30, 150,
                 color=(0, 0, 0, 170), thick=1.3)
        cls._fspline(img, [(23, 73), (28, 78), (32, 77), (28, 72)],
                     color=(0, 0, 0, 45))
        cls._fspline(img, [(47, 73), (42, 78), (38, 77), (42, 72)],
                     color=(0, 0, 0, 45))
        noses.append(cls._finalize(img, w, h))

        # --- Aquiline / hooked nose ---
        img = cls._canvas(w, h)
        # Pronounced bridge with bump
        cls._spline(img, [
            (33, 6), (34, 16), (36, 28), (38, 36), (37, 46),
            (34, 56), (30, 66),
        ], thick=1.8, color=(0, 0, 0, 140))
        cls._spline(img, [
            (37, 6), (38, 16), (40, 28), (42, 36), (41, 46),
            (40, 56), (44, 66),
        ], thick=1.8, color=(0, 0, 0, 140))
        # Tip area
        cls._spline(img, [
            (30, 66), (27, 74), (29, 82), (34, 86), (38, 85),
            (43, 80), (44, 74), (44, 66),
        ], thick=2.2, color=(0, 0, 0, 200))
        # Nostrils
        cls._ell(img, 30, 83, 7, 4, 25, 25, 205,
                 color=(0, 0, 0, 175), thick=1.5)
        cls._ell(img, 40, 83, 7, 4, 155, -25, 155,
                 color=(0, 0, 0, 175), thick=1.5)
        # Bridge bump highlight
        cls._spline(img, [(35, 28), (37, 34), (36, 40)],
                    thick=0.8, color=(0, 0, 0, 60))
        cls._fspline(img, [(26, 81), (30, 86), (34, 85), (30, 80)],
                     color=(0, 0, 0, 50))
        cls._fspline(img, [(44, 81), (40, 86), (36, 85), (40, 80)],
                     color=(0, 0, 0, 50))
        noses.append(cls._finalize(img, w, h))

        return noses

    # ── mouth templates ────────────────────────────────────────────────────

    @classmethod
    def generate_mouth_templates(cls) -> List[np.ndarray]:
        mouths = []
        w, h = 120, 60

        # --- Neutral / closed ---
        img = cls._canvas(w, h)
        # Lip line (most prominent)
        cls._spline(img, [
            (22, 30), (35, 28), (48, 26), (55, 25), (60, 24.5),
            (65, 25), (72, 26), (85, 28), (98, 30),
        ], thick=2.5, color=(0, 0, 0, 220))
        # Cupid's bow (upper lip top edge)
        cls._spline(img, [
            (28, 29), (40, 24), (50, 20), (55, 18.5),
            (58, 17), (60, 16.5), (62, 17), (65, 18.5),
            (70, 20), (80, 24), (92, 29),
        ], thick=1.6, color=(0, 0, 0, 140))
        # Philtrum lines
        cls._ln(img, 57, 11, 58, 17, thick=0.8, color=(0, 0, 0, 60))
        cls._ln(img, 63, 11, 62, 17, thick=0.8, color=(0, 0, 0, 60))
        # Lower lip
        cls._spline(img, [
            (28, 31), (40, 36), (52, 39), (60, 40),
            (68, 39), (80, 36), (92, 31),
        ], thick=1.4, color=(0, 0, 0, 130))
        # Upper lip shading
        cls._fspline(img, [
            (35, 27), (48, 22), (55, 19), (60, 18), (65, 19),
            (72, 22), (85, 27), (80, 26), (65, 24), (60, 23.5),
            (55, 24), (40, 26),
        ], color=(0, 0, 0, 30))
        # Lower lip shading
        cls._fspline(img, [
            (35, 32), (48, 37), (60, 39), (72, 37), (85, 32),
            (80, 34), (68, 36), (60, 37), (52, 36), (40, 34),
        ], color=(0, 0, 0, 25))
        # Corner dimples
        cls._cir(img, 22, 30, 1, fill=True, color=(0, 0, 0, 50))
        cls._cir(img, 98, 30, 1, fill=True, color=(0, 0, 0, 50))
        mouths.append(cls._finalize(img, w, h))

        # --- Slightly-open / smiling ---
        img = cls._canvas(w, h)
        # Upper lip edge with smile curve
        cls._spline(img, [
            (18, 33), (28, 26), (40, 21), (50, 18), (55, 16.5),
            (58, 15), (60, 14.5), (62, 15), (65, 16.5),
            (70, 18), (80, 21), (92, 26), (102, 33),
        ], thick=2.2, color=(0, 0, 0, 220))
        # Lower lip
        cls._spline(img, [
            (18, 33), (30, 40), (45, 45), (60, 47),
            (75, 45), (90, 40), (102, 33),
        ], thick=1.8, color=(0, 0, 0, 180))
        # Teeth hint (light fill between lips)
        cls._fspline(img, [
            (30, 30), (45, 24), (60, 22), (75, 24), (90, 30),
            (85, 36), (70, 39), (60, 40), (50, 39), (35, 36),
        ], color=(255, 255, 255, 100))
        # Upper lip shading
        cls._fspline(img, [
            (28, 30), (45, 22), (60, 18), (75, 22), (92, 30),
            (85, 27), (72, 23), (60, 22), (48, 23), (35, 27),
        ], color=(0, 0, 0, 25))
        # Smile creases
        cls._spline(img, [(14, 36), (18, 33), (22, 31)],
                    thick=1.2, color=(0, 0, 0, 100))
        cls._spline(img, [(106, 36), (102, 33), (98, 31)],
                    thick=1.2, color=(0, 0, 0, 100))
        mouths.append(cls._finalize(img, w, h))

        # --- Small / pursed ---
        img = cls._canvas(w, h)
        # Lip line
        cls._spline(img, [
            (38, 30), (44, 27), (50, 25), (56, 24), (60, 23.5),
            (64, 24), (70, 25), (76, 27), (82, 30),
        ], thick=2.0, color=(0, 0, 0, 220))
        # Upper lip contour
        cls._spline(img, [
            (40, 29), (48, 25), (54, 22.5), (58, 22),
            (60, 21.5), (62, 22), (66, 22.5), (72, 25), (80, 29),
        ], thick=1.2, color=(0, 0, 0, 130))
        # Lower lip contour
        cls._spline(img, [
            (40, 31), (48, 34), (54, 36), (60, 37),
            (66, 36), (72, 34), (80, 31),
        ], thick=1.2, color=(0, 0, 0, 130))
        # Subtle shading
        cls._fspline(img, [
            (44, 28), (52, 24), (60, 23), (68, 24), (76, 28),
            (72, 26.5), (60, 25.5), (48, 26.5),
        ], color=(0, 0, 0, 28))
        cls._fspline(img, [
            (44, 32), (52, 35), (60, 36), (68, 35), (76, 32),
            (72, 33.5), (60, 34.5), (48, 33.5),
        ], color=(0, 0, 0, 22))
        mouths.append(cls._finalize(img, w, h))

        return mouths

    # ── eyebrow templates ──────────────────────────────────────────────────

    @classmethod
    def _draw_brow(cls, img, cx, cy, style='straight'):
        """Draw one eyebrow with filled shape and dense hair-stroke texture."""

        if style == 'straight':
            top = [
                (cx - 38, cy + 1), (cx - 25, cy - 2), (cx - 10, cy - 4),
                (cx, cy - 5), (cx + 10, cy - 5), (cx + 25, cy - 3),
                (cx + 38, cy + 0),
            ]
            bot = [
                (cx + 38, cy + 4), (cx + 20, cy + 3),
                (cx + 5, cy + 2), (cx - 10, cy + 2), (cx - 25, cy + 4),
                (cx - 38, cy + 4),
            ]
        elif style == 'arched':
            top = [
                (cx - 38, cy + 4), (cx - 25, cy - 2), (cx - 12, cy - 8),
                (cx, cy - 10), (cx + 12, cy - 9), (cx + 25, cy - 4),
                (cx + 38, cy + 2),
            ]
            bot = [
                (cx + 38, cy + 5), (cx + 20, cy + 0),
                (cx + 5, cy - 3), (cx - 10, cy - 2), (cx - 25, cy + 3),
                (cx - 38, cy + 7),
            ]
        else:  # angled
            top = [
                (cx - 38, cy + 5), (cx - 25, cy - 2), (cx - 10, cy - 9),
                (cx, cy - 11), (cx + 10, cy - 8), (cx + 25, cy - 3),
                (cx + 38, cy + 1),
            ]
            bot = [
                (cx + 38, cy + 4), (cx + 20, cy + 1),
                (cx + 5, cy - 2), (cx - 10, cy - 3), (cx - 25, cy + 4),
                (cx - 38, cy + 8),
            ]

        outline = top + bot
        # Brow body fill
        cls._fspline(img, outline, color=(0, 0, 0, 50))
        # Top edge
        cls._spline(img, top, thick=1.6, color=(0, 0, 0, 180))

        # Dense directional hair strokes
        rng = np.random.RandomState(42 + hash(style) % 100)
        for i in range(45):
            bx = cx - 36 + i * 1.65
            t_p = max(0, min(1, (bx - (cx - 38)) / 76.0))
            # Interpolate y along top edge
            by_top = cy - 5 + 3  # approximate baseline
            if style == 'arched':
                by_top = cy + 4 - 14 * np.sin(t_p * np.pi) + 4
            elif style == 'angled':
                if t_p < 0.5:
                    by_top = cy + 5 - 16 * (t_p / 0.5) + 4
                else:
                    by_top = cy - 11 + 12 * ((t_p - 0.5) / 0.5) + 4
            else:
                by_top = cy + 1 - 6 * np.sin(t_p * np.pi) + 4
            by = by_top + rng.uniform(-1, 1)
            angle = 78 + (bx - cx) * 0.3 + rng.uniform(-8, 8)
            rad = np.radians(angle)
            length = 5 + rng.random() * 3.5
            alpha = 120 + int(rng.random() * 100)
            cls._ln(img, bx, by,
                    bx + length * np.cos(rad), by - length * np.sin(rad),
                    thick=0.6, color=(0, 0, 0, alpha))

    @classmethod
    def generate_eyebrow_templates(cls) -> List[np.ndarray]:
        eyebrows = []
        w, h = 260, 40
        for style in ['straight', 'arched', 'angled']:
            img = cls._canvas(w, h)
            cls._draw_brow(img, 70, h // 2, style)
            cls._draw_brow(img, 190, h // 2, style)
            eyebrows.append(cls._finalize(img, w, h))
        return eyebrows

    # ── face shape templates ───────────────────────────────────────────────

    @classmethod
    def generate_face_shape_templates(cls) -> List[np.ndarray]:
        faces = []
        w, h = 400, 512

        # --- Oval ---
        img = cls._canvas(w, h)
        # Proper head shape: rounded forehead, tapered chin
        oval_pts = [
            (200, 40),
            (165, 50), (138, 70), (120, 100), (112, 140),
            (108, 185), (108, 230), (110, 275), (115, 315),
            (125, 350), (140, 380), (160, 405), (180, 425),
            (192, 438), (200, 445),
            (208, 438), (220, 425), (240, 405), (260, 380),
            (275, 350), (285, 315), (290, 275), (292, 230),
            (292, 185), (288, 140), (280, 100), (262, 70),
            (235, 50), (200, 40),
        ]
        cls._spline(img, oval_pts, thick=3.0, closed=True, color=(0, 0, 0, 220))
        # Jawline emphasis
        cls._spline(img, [
            (140, 380), (160, 405), (180, 425), (192, 438),
            (200, 445), (208, 438), (220, 425), (240, 405), (260, 380),
        ], thick=1.5, color=(0, 0, 0, 60))
        # Subtle cheekbone hints
        cls._spline(img, [(110, 240), (112, 260), (116, 280)],
                    thick=1.0, color=(0, 0, 0, 45))
        cls._spline(img, [(290, 240), (288, 260), (284, 280)],
                    thick=1.0, color=(0, 0, 0, 45))
        faces.append(cls._finalize(img, w, h))

        # --- Round ---
        img = cls._canvas(w, h)
        round_pts = [
            (200, 38),
            (162, 46), (130, 68), (108, 100), (96, 140),
            (90, 185), (88, 240), (90, 295), (96, 340),
            (108, 375), (128, 405), (155, 428), (180, 442),
            (200, 448),
            (220, 442), (245, 428), (272, 405), (292, 375),
            (304, 340), (310, 295), (312, 240), (310, 185),
            (304, 140), (292, 100), (270, 68), (238, 46),
            (200, 38),
        ]
        cls._spline(img, round_pts, thick=3.0, closed=True, color=(0, 0, 0, 220))
        cls._spline(img, [(90, 230), (89, 270), (92, 310)],
                    thick=1.0, color=(0, 0, 0, 50))
        cls._spline(img, [(310, 230), (311, 270), (308, 310)],
                    thick=1.0, color=(0, 0, 0, 50))
        faces.append(cls._finalize(img, w, h))

        # --- Square / angular ---
        img = cls._canvas(w, h)
        square_pts = [
            (200, 38),
            (165, 46), (136, 65), (118, 95), (110, 130),
            (106, 170), (104, 220), (104, 270), (104, 320),
            (106, 350), (115, 378), (135, 405), (160, 425),
            (180, 438), (200, 445),
            (220, 438), (240, 425), (265, 405), (285, 378),
            (294, 350), (296, 320), (296, 270), (296, 220),
            (294, 170), (290, 130), (282, 95), (264, 65),
            (235, 46), (200, 38),
        ]
        cls._spline(img, square_pts, thick=3.0, closed=True, color=(0, 0, 0, 220))
        # Jaw angle points
        cls._cir(img, 106, 350, 2, fill=True, color=(0, 0, 0, 120))
        cls._cir(img, 294, 350, 2, fill=True, color=(0, 0, 0, 120))
        # Strong jawline
        cls._spline(img, [
            (104, 320), (106, 350), (115, 378), (135, 405), (160, 425),
        ], thick=2.5, color=(0, 0, 0, 160))
        cls._spline(img, [
            (296, 320), (294, 350), (285, 378), (265, 405), (240, 425),
        ], thick=2.5, color=(0, 0, 0, 160))
        faces.append(cls._finalize(img, w, h))

        return faces

    # ── ear templates ──────────────────────────────────────────────────────

    @classmethod
    def _draw_ear_detail(cls, img, cx, cy, style='normal', flip=False):
        d = -1 if flip else 1

        if style == 'normal':
            outer = [
                (cx, cy - 30), (cx + d * 10, cy - 34),
                (cx + d * 18, cy - 32), (cx + d * 26, cy - 24),
                (cx + d * 31, cy - 12), (cx + d * 33, cy),
                (cx + d * 31, cy + 14), (cx + d * 26, cy + 24),
                (cx + d * 18, cy + 32), (cx + d * 8, cy + 36),
                (cx, cy + 32),
            ]
            cls._spline(img, outer, thick=2.2, color=(0, 0, 0, 210))
            # Antihelix (Y-shaped inner ridge)
            cls._spline(img, [
                (cx + d * 5, cy - 22), (cx + d * 14, cy - 20),
                (cx + d * 20, cy - 12), (cx + d * 22, cy - 2),
                (cx + d * 20, cy + 10), (cx + d * 14, cy + 20),
                (cx + d * 6, cy + 18),
            ], thick=1.2, color=(0, 0, 0, 110))
            # Tragus
            cls._spline(img, [
                (cx, cy - 6), (cx + d * 6, cy - 3),
                (cx + d * 7, cy + 3), (cx + d * 4, cy + 6),
                (cx, cy + 8),
            ], thick=1.0, color=(0, 0, 0, 130))
            # Earlobe
            cls._spline(img, [
                (cx, cy + 32), (cx + d * 3, cy + 37),
                (cx + d * 7, cy + 35), (cx + d * 8, cy + 30),
            ], thick=1.2, color=(0, 0, 0, 150))
        elif style == 'pointed':
            outer = [
                (cx, cy - 30), (cx + d * 12, cy - 38),
                (cx + d * 22, cy - 42), (cx + d * 30, cy - 30),
                (cx + d * 33, cy - 16), (cx + d * 29, cy + 4),
                (cx + d * 21, cy + 24), (cx + d * 10, cy + 32),
                (cx, cy + 28),
            ]
            cls._spline(img, outer, thick=2.2, color=(0, 0, 0, 210))
            # Inner fold
            cls._spline(img, [
                (cx + d * 6, cy - 24), (cx + d * 16, cy - 22),
                (cx + d * 22, cy - 12), (cx + d * 19, cy + 4),
                (cx + d * 10, cy + 16), (cx + d * 4, cy + 16),
            ], thick=1.2, color=(0, 0, 0, 110))
            # Tragus
            cls._spline(img, [
                (cx, cy - 4), (cx + d * 5, cy - 2),
                (cx + d * 5, cy + 4), (cx, cy + 6),
            ], thick=1.0, color=(0, 0, 0, 120))
        else:  # small / attached lobe
            outer = [
                (cx, cy - 20), (cx + d * 8, cy - 24),
                (cx + d * 16, cy - 20), (cx + d * 21, cy - 10),
                (cx + d * 22, cy + 2), (cx + d * 18, cy + 14),
                (cx + d * 10, cy + 22), (cx + d * 2, cy + 22),
                (cx, cy + 18),
            ]
            cls._spline(img, outer, thick=2.0, color=(0, 0, 0, 210))
            # Inner fold
            cls._spline(img, [
                (cx + d * 4, cy - 14), (cx + d * 10, cy - 12),
                (cx + d * 14, cy - 4), (cx + d * 12, cy + 6),
                (cx + d * 6, cy + 12),
            ], thick=1.0, color=(0, 0, 0, 100))
            # Tragus
            cls._spline(img, [
                (cx, cy - 2), (cx + d * 4, cy),
                (cx + d * 4, cy + 4), (cx, cy + 6),
            ], thick=0.9, color=(0, 0, 0, 110))

    @classmethod
    def generate_ear_templates(cls) -> List[np.ndarray]:
        ears = []
        w, h = 380, 120
        for style in ['normal', 'pointed', 'small']:
            img = cls._canvas(w, h)
            cls._draw_ear_detail(img, 40, h // 2, style, flip=False)
            cls._draw_ear_detail(img, w - 40, h // 2, style, flip=True)
            ears.append(cls._finalize(img, w, h))
        return ears

    # ── hair templates ─────────────────────────────────────────────────────

    @classmethod
    def generate_hair_templates(cls) -> List[np.ndarray]:
        hairs = []
        w, h = 360, 180

        # --- Short cropped ---
        img = cls._canvas(w, h)
        hairline = [
            (50, 165), (42, 125), (40, 92), (46, 62), (60, 40),
            (85, 24), (115, 14), (150, 7), (180, 5), (210, 7),
            (245, 14), (275, 24), (300, 40), (314, 62), (320, 92),
            (318, 125), (310, 165),
        ]
        cls._spline(img, hairline, thick=2.5, color=(0, 0, 0, 210))
        # Dense short hair texture
        rng = np.random.RandomState(50)
        for x in range(58, 302, 4):
            t_p = (x - 50) / 260.0
            y_top = 8 + 50 * abs(t_p - 0.5) ** 1.5 * 4
            for y_off in range(0, 60, 6):
                yo = y_top + y_off + rng.uniform(-2, 2)
                if yo < 158:
                    angle = 70 + rng.uniform(-15, 15)
                    # Hair radiates outward slightly from center
                    angle += (t_p - 0.5) * 20
                    rad = np.radians(angle)
                    ln = 5 + rng.uniform(0, 4)
                    alpha = 50 + int(rng.random() * 100)
                    cls._ln(img, x, yo,
                            x + ln * np.cos(rad), yo + ln * np.sin(rad),
                            thick=0.5, color=(0, 0, 0, alpha))
        hairs.append(cls._finalize(img, w, h))

        # --- Long flowing ---
        img = cls._canvas(w, h)
        hairline = [
            (30, h), (24, 148), (26, 112), (36, 78), (52, 50),
            (78, 30), (112, 16), (152, 8), (180, 5), (208, 8),
            (248, 16), (282, 30), (308, 50), (324, 78), (334, 112),
            (336, 148), (330, h),
        ]
        cls._spline(img, hairline, thick=2.5, color=(0, 0, 0, 210))
        # Long flowing strands
        rng = np.random.RandomState(51)
        for x in range(40, 320, 3):
            t_p = (x - 30) / 300.0
            y_top = 8 + 55 * abs(t_p - 0.5) ** 1.4 * 4
            jitter = rng.uniform(-2, 2)
            alpha = 40 + int(rng.random() * 70)
            # Use a wavy line instead of straight
            x_end = x + jitter + rng.uniform(-5, 5)
            # Draw multi-segment wavy strand
            strand_pts = []
            n_seg = 5
            for s in range(n_seg + 1):
                frac = s / n_seg
                sx = x + jitter + rng.uniform(-2, 2) * frac
                sy = y_top + 8 + (h - y_top - 10) * frac
                strand_pts.append((sx, sy))
            if len(strand_pts) >= 3:
                cls._spline(img, strand_pts, thick=0.5, color=(0, 0, 0, alpha))
        hairs.append(cls._finalize(img, w, h))

        # --- Curly / wavy ---
        img = cls._canvas(w, h)
        hairline = [
            (28, 168), (16, 118), (20, 74), (36, 44), (62, 22),
            (98, 10), (142, 4), (180, 2), (218, 4), (262, 10),
            (298, 22), (324, 44), (340, 74), (344, 118), (332, 168),
        ]
        cls._spline(img, hairline, thick=2.5, color=(0, 0, 0, 210))
        # Curly loops
        rng = np.random.RandomState(52)
        for x in range(42, 320, 8):
            t_p = (x - 28) / 304.0
            y_top = 6 + 55 * abs(t_p - 0.5) ** 1.4 * 4
            for y in range(int(y_top) + 8, min(h - 6, int(y_top) + 125), 10):
                r = 3.5 + rng.random() * 2.5
                alpha = 55 + int(rng.random() * 90)
                ox = rng.uniform(-2, 2)
                oy = rng.uniform(-2, 2)
                angle_start = rng.randint(0, 360)
                arc_span = 260 + rng.randint(0, 60)
                cls._ell(img,
                         x + ox, y + oy, r, r * 1.1,
                         rng.randint(-20, 20), angle_start,
                         angle_start + arc_span,
                         thick=0.6, color=(0, 0, 0, alpha))
        hairs.append(cls._finalize(img, w, h))

        return hairs

    # ── aggregation & persistence ──────────────────────────────────────────

    @staticmethod
    def generate_all_templates() -> Dict[str, List[np.ndarray]]:
        """Generate all feature templates."""
        return {
            'face_shapes': FeatureTemplateGenerator.generate_face_shape_templates(),
            'hair': FeatureTemplateGenerator.generate_hair_templates(),
            'ears': FeatureTemplateGenerator.generate_ear_templates(),
            'eyebrows': FeatureTemplateGenerator.generate_eyebrow_templates(),
            'eyes': FeatureTemplateGenerator.generate_eye_templates(),
            'noses': FeatureTemplateGenerator.generate_nose_templates(),
            'mouths': FeatureTemplateGenerator.generate_mouth_templates(),
        }

    @staticmethod
    def save_templates_to_disk(output_dir: str):
        """Save all generated templates to disk."""
        os.makedirs(output_dir, exist_ok=True)
        templates = FeatureTemplateGenerator.generate_all_templates()
        for feature_type, template_list in templates.items():
            feature_dir = os.path.join(output_dir, feature_type)
            os.makedirs(feature_dir, exist_ok=True)
            for idx, template in enumerate(template_list):
                filepath = os.path.join(feature_dir, f"{feature_type}_{idx + 1}.png")
                cv2.imwrite(filepath, template)
        print(f"Templates saved to {output_dir}")
        return templates

    @staticmethod
    def load_feature_from_disk(feature_type: str, feature_id: str,
                               base_dir: str) -> np.ndarray:
        """Load a specific feature template from disk."""
        try:
            parts = feature_id.split('_')
            idx = int(parts[-1])
            filepath = os.path.join(base_dir, feature_type,
                                    f"{feature_type}_{idx}.png")
            if os.path.exists(filepath):
                return cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
            return FeatureTemplateGenerator._generate_feature_by_index(
                feature_type, idx)
        except Exception as e:
            print(f"Error loading feature {feature_id}: {e}")
            return None

    @staticmethod
    def _generate_feature_by_index(feature_type: str, idx: int) -> np.ndarray:
        """Generate a specific feature by index on the fly."""
        generators = {
            'eyes': FeatureTemplateGenerator.generate_eye_templates,
            'noses': FeatureTemplateGenerator.generate_nose_templates,
            'mouths': FeatureTemplateGenerator.generate_mouth_templates,
            'eyebrows': FeatureTemplateGenerator.generate_eyebrow_templates,
            'face_shapes': FeatureTemplateGenerator.generate_face_shape_templates,
            'ears': FeatureTemplateGenerator.generate_ear_templates,
            'hair': FeatureTemplateGenerator.generate_hair_templates,
        }
        if feature_type in generators:
            templates = generators[feature_type]()
            if 0 < idx <= len(templates):
                return templates[idx - 1]
        return None
