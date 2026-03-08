import cv2
import numpy as np
from typing import Dict, List
import os

# ---------------------------------------------------------------------------
# Supersampling factor: all coordinates are multiplied by _K, drawn with
# cv2.LINE_AA, then the final image is down-scaled with INTER_AREA.
# This produces smooth, naturally anti-aliased pencil-sketch lines.
# ---------------------------------------------------------------------------
_K = 3


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
        return max(1, int(round(v * _K * 0.65)))

    @staticmethod
    def _canvas(w, h):
        return np.zeros((h * _K, w * _K, 4), dtype=np.uint8)

    @staticmethod
    def _finalize(img, w, h):
        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

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

    # ── eye templates ──────────────────────────────────────────────────────

    @classmethod
    def _draw_eye(cls, img, cx, cy, style='almond'):
        """Draw one detailed eye at (cx, cy)."""

        # --- lid outlines (style-dependent) ---
        if style == 'almond':
            upper = [
                (cx - 40, cy + 3), (cx - 32, cy - 5), (cx - 20, cy - 12),
                (cx - 8, cy - 15), (cx, cy - 16), (cx + 8, cy - 15),
                (cx + 20, cy - 12), (cx + 32, cy - 6), (cx + 40, cy + 2),
            ]
            lower = [
                (cx - 40, cy + 3), (cx - 28, cy + 10), (cx - 14, cy + 14),
                (cx, cy + 15), (cx + 14, cy + 14), (cx + 28, cy + 10),
                (cx + 40, cy + 2),
            ]
            crease = [
                (cx - 36, cy - 6), (cx - 22, cy - 18), (cx - 8, cy - 22),
                (cx, cy - 23), (cx + 8, cy - 22), (cx + 22, cy - 17),
                (cx + 36, cy - 9),
            ]
            lid_amp = 16
        elif style == 'round':
            upper = [
                (cx - 36, cy + 2), (cx - 26, cy - 10), (cx - 14, cy - 17),
                (cx, cy - 19), (cx + 14, cy - 17), (cx + 26, cy - 10),
                (cx + 36, cy + 2),
            ]
            lower = [
                (cx - 36, cy + 2), (cx - 26, cy + 12), (cx - 14, cy + 17),
                (cx, cy + 19), (cx + 14, cy + 17), (cx + 26, cy + 12),
                (cx + 36, cy + 2),
            ]
            crease = [
                (cx - 32, cy - 8), (cx - 18, cy - 22), (cx, cy - 26),
                (cx + 18, cy - 22), (cx + 32, cy - 8),
            ]
            lid_amp = 19
        else:  # narrow
            upper = [
                (cx - 42, cy + 1), (cx - 30, cy - 3), (cx - 16, cy - 7),
                (cx, cy - 8), (cx + 16, cy - 7), (cx + 30, cy - 3),
                (cx + 42, cy + 1),
            ]
            lower = [
                (cx - 42, cy + 1), (cx - 28, cy + 6), (cx - 14, cy + 9),
                (cx, cy + 10), (cx + 14, cy + 9), (cx + 28, cy + 6),
                (cx + 42, cy + 1),
            ]
            crease = [
                (cx - 38, cy - 1), (cx - 24, cy - 8), (cx - 10, cy - 10),
                (cx, cy - 10), (cx + 10, cy - 10), (cx + 24, cy - 8),
                (cx + 38, cy - 2),
            ]
            lid_amp = 8

        cls._poly(img, upper, thick=2.5, color=(0, 0, 0, 230))
        cls._poly(img, lower, thick=1.5, color=(0, 0, 0, 170))
        cls._poly(img, crease, thick=1, color=(0, 0, 0, 90))

        # --- iris with radial detail ---
        iris_r = 11 if style == 'round' else 10
        cls._cir(img, cx, cy, iris_r, thick=1.5, color=(0, 0, 0, 190))
        for ang in range(0, 360, 18):
            rad = np.radians(ang)
            r1, r2 = 4.5, iris_r - 1
            cls._ln(img, cx + r1 * np.cos(rad), cy + r1 * np.sin(rad),
                    cx + r2 * np.cos(rad), cy + r2 * np.sin(rad),
                    color=(0, 0, 0, 60), thick=0.5)

        # --- pupil + highlight ---
        cls._cir(img, cx, cy, 4.5, fill=True, color=(0, 0, 0, 240))
        cls._cir(img, cx + 2.5, cy - 2.5, 1.8, fill=True,
                 color=(255, 255, 255, 220))

        # --- tear duct ---
        cls._poly(img, [(cx - 40, cy + 2), (cx - 44, cy + 3), (cx - 40, cy + 5)],
                  thick=1, color=(0, 0, 0, 140))

        # --- upper lashes ---
        half_w = 32
        for i in range(13):
            lx = cx - half_w + i * 5
            t_param = (lx - (cx - 40)) / 80.0
            lid_y = cy - lid_amp * np.sin(t_param * np.pi) \
                    + 3 * (1 - np.sin(t_param * np.pi))
            lash_len = 3 + 2 * np.sin(t_param * np.pi)
            fan_x = (t_param - 0.5) * 4
            cls._ln(img, lx, lid_y, lx + fan_x, lid_y - lash_len,
                    thick=0.7, color=(0, 0, 0, 190))

        # --- subtle lower-lid shadow marks ---
        for lx in range(int(cx - 20), int(cx + 25), 8):
            base_y = cy + 14 - abs(lx - cx) * 0.15
            cls._ln(img, lx, base_y, lx, base_y + 2,
                    thick=0.5, color=(0, 0, 0, 80))

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
        cls._poly(img, [(30, 12), (29, 30), (28, 50), (26, 65)],
                  thick=1.5, color=(0, 0, 0, 130))
        cls._poly(img, [(40, 12), (41, 30), (42, 50), (44, 65)],
                  thick=1.5, color=(0, 0, 0, 130))
        cls._poly(img, [
            (26, 65), (24, 72), (25, 78), (30, 82), (35, 84),
            (40, 82), (45, 78), (46, 72), (44, 65),
        ], thick=2, color=(0, 0, 0, 200))
        cls._ell(img, 28, 80, 7, 4, 30, 30, 210, color=(0, 0, 0, 170), thick=1.5)
        cls._ell(img, 42, 80, 7, 4, 150, -30, 150, color=(0, 0, 0, 170), thick=1.5)
        cls._fpoly(img, [(24, 78), (28, 83), (33, 82), (30, 77)],
                   color=(0, 0, 0, 45))
        cls._fpoly(img, [(46, 78), (42, 83), (37, 82), (40, 77)],
                   color=(0, 0, 0, 45))
        noses.append(cls._finalize(img, w, h))

        # --- Button / upturned nose ---
        img = cls._canvas(w, h)
        cls._poly(img, [(32, 20), (31, 35), (30, 50)],
                  thick=1.2, color=(0, 0, 0, 110))
        cls._poly(img, [(38, 20), (39, 35), (40, 50)],
                  thick=1.2, color=(0, 0, 0, 110))
        cls._poly(img, [
            (25, 60), (22, 67), (24, 74), (30, 78), (35, 80),
            (40, 78), (46, 74), (48, 67), (45, 60),
        ], thick=2, color=(0, 0, 0, 200))
        cls._ell(img, 35, 72, 12, 8, -10, 20, 160,
                 color=(0, 0, 0, 170), thick=1.8)
        cls._ell(img, 28, 74, 6, 4, 40, 20, 200,
                 color=(0, 0, 0, 160), thick=1.2)
        cls._ell(img, 42, 74, 6, 4, 140, -20, 160,
                 color=(0, 0, 0, 160), thick=1.2)
        cls._fpoly(img, [(24, 73), (28, 77), (31, 76), (28, 72)],
                   color=(0, 0, 0, 40))
        cls._fpoly(img, [(46, 73), (42, 77), (39, 76), (42, 72)],
                   color=(0, 0, 0, 40))
        noses.append(cls._finalize(img, w, h))

        # --- Aquiline / hooked nose ---
        img = cls._canvas(w, h)
        cls._poly(img, [
            (32, 8), (33, 18), (35, 28), (38, 38), (37, 48), (34, 58), (30, 68),
        ], thick=1.8, color=(0, 0, 0, 150))
        cls._poly(img, [
            (38, 8), (39, 18), (41, 28), (43, 38), (42, 48), (41, 58), (44, 68),
        ], thick=1.8, color=(0, 0, 0, 150))
        cls._poly(img, [
            (30, 68), (28, 75), (30, 82), (35, 85), (40, 83),
            (44, 78), (44, 68),
        ], thick=2, color=(0, 0, 0, 200))
        cls._ell(img, 29, 82, 7, 4, 30, 20, 200,
                 color=(0, 0, 0, 170), thick=1.5)
        cls._ell(img, 41, 82, 7, 4, 150, -20, 160,
                 color=(0, 0, 0, 170), thick=1.5)
        cls._fpoly(img, [(25, 80), (29, 85), (33, 84), (30, 79)],
                   color=(0, 0, 0, 45))
        cls._fpoly(img, [(45, 80), (41, 85), (37, 84), (40, 79)],
                   color=(0, 0, 0, 45))
        noses.append(cls._finalize(img, w, h))

        return noses

    # ── mouth templates ────────────────────────────────────────────────────

    @classmethod
    def generate_mouth_templates(cls) -> List[np.ndarray]:
        mouths = []
        w, h = 120, 60

        # --- Neutral / closed ---
        img = cls._canvas(w, h)
        cls._poly(img, [
            (22, 30), (35, 28), (48, 26), (55, 25), (60, 24.5),
            (65, 25), (72, 26), (85, 28), (98, 30),
        ], thick=2.5, color=(0, 0, 0, 210))
        cls._poly(img, [
            (22, 30), (35, 26), (48, 22), (54, 21), (60, 19.5),
            (66, 21), (72, 22), (85, 26), (98, 30),
        ], thick=1.5, color=(0, 0, 0, 155))
        cls._poly(img, [
            (22, 30), (35, 35), (48, 38), (60, 39),
            (72, 38), (85, 35), (98, 30),
        ], thick=1.5, color=(0, 0, 0, 155))
        cls._fpoly(img, [
            (30, 28), (48, 23), (60, 20), (72, 23), (90, 28),
            (85, 27), (72, 25), (60, 24), (48, 25), (35, 27),
        ], color=(0, 0, 0, 30))
        cls._fpoly(img, [
            (30, 31), (48, 36), (60, 38), (72, 36), (90, 31),
            (85, 33), (72, 34), (60, 35), (48, 34), (35, 33),
        ], color=(0, 0, 0, 25))
        cls._ln(img, 56, 14, 57, 20, thick=0.8, color=(0, 0, 0, 70))
        cls._ln(img, 64, 14, 63, 20, thick=0.8, color=(0, 0, 0, 70))
        mouths.append(cls._finalize(img, w, h))

        # --- Slightly-open / smiling ---
        img = cls._canvas(w, h)
        cls._poly(img, [
            (18, 32), (30, 24), (42, 19), (52, 17), (60, 16),
            (68, 17), (78, 19), (90, 24), (102, 32),
        ], thick=2, color=(0, 0, 0, 210))
        cls._poly(img, [
            (18, 32), (30, 40), (45, 45), (60, 47),
            (75, 45), (90, 40), (102, 32),
        ], thick=1.8, color=(0, 0, 0, 175))
        cls._fpoly(img, [
            (28, 28), (45, 22), (60, 20), (75, 22), (92, 28),
            (85, 34), (70, 37), (60, 38), (50, 37), (35, 34),
        ], color=(230, 230, 230, 170))
        cls._ln(img, 60, 22, 60, 35, thick=0.6, color=(0, 0, 0, 70))
        cls._fpoly(img, [
            (25, 30), (45, 20), (60, 18), (75, 20), (95, 30),
            (85, 26), (72, 22), (60, 21), (48, 22), (35, 26),
        ], color=(0, 0, 0, 28))
        cls._poly(img, [(14, 35), (18, 32), (22, 30)],
                  thick=1, color=(0, 0, 0, 110))
        cls._poly(img, [(106, 35), (102, 32), (98, 30)],
                  thick=1, color=(0, 0, 0, 110))
        mouths.append(cls._finalize(img, w, h))

        # --- Small / pursed ---
        img = cls._canvas(w, h)
        cls._poly(img, [
            (38, 30), (45, 27), (52, 25), (58, 24), (62, 24),
            (68, 25), (75, 27), (82, 30),
        ], thick=2, color=(0, 0, 0, 210))
        cls._poly(img, [
            (38, 30), (48, 26), (56, 23.5), (60, 23),
            (64, 23.5), (72, 26), (82, 30),
        ], thick=1.2, color=(0, 0, 0, 145))
        cls._poly(img, [
            (38, 30), (48, 34), (56, 36), (60, 36.5),
            (64, 36), (72, 34), (82, 30),
        ], thick=1.2, color=(0, 0, 0, 145))
        cls._fpoly(img, [
            (42, 29), (52, 25), (60, 24), (68, 25), (78, 29),
            (72, 27), (60, 26), (48, 27),
        ], color=(0, 0, 0, 30))
        cls._fpoly(img, [
            (42, 31), (52, 34), (60, 35), (68, 34), (78, 31),
            (72, 33), (60, 34), (48, 33),
        ], color=(0, 0, 0, 25))
        mouths.append(cls._finalize(img, w, h))

        return mouths

    # ── eyebrow templates ──────────────────────────────────────────────────

    @classmethod
    def _draw_brow(cls, img, cx, cy, style='straight'):
        """Draw one eyebrow with filled shape and hair-stroke texture."""

        if style == 'straight':
            outline = [
                (cx - 38, cy + 3), (cx - 25, cy - 1), (cx - 10, cy - 3),
                (cx, cy - 4), (cx + 10, cy - 4), (cx + 25, cy - 2),
                (cx + 38, cy + 1), (cx + 35, cy + 4), (cx + 20, cy + 3),
                (cx + 5, cy + 2), (cx - 10, cy + 2), (cx - 25, cy + 4),
                (cx - 38, cy + 3),
            ]
            top_curve = outline[:7]
            base_y_func = lambda t: cy + 3 - 7 * np.sin(t * np.pi)
        elif style == 'arched':
            outline = [
                (cx - 38, cy + 5), (cx - 25, cy - 2), (cx - 12, cy - 8),
                (cx, cy - 10), (cx + 12, cy - 9), (cx + 25, cy - 4),
                (cx + 38, cy + 2), (cx + 35, cy + 5), (cx + 20, cy + 1),
                (cx + 5, cy - 3), (cx - 10, cy - 2), (cx - 25, cy + 3),
                (cx - 38, cy + 5),
            ]
            top_curve = outline[:7]
            base_y_func = lambda t: cy + 5 - 15 * np.sin(t * np.pi)
        else:  # angled
            outline = [
                (cx - 38, cy + 6), (cx - 25, cy - 2), (cx - 10, cy - 9),
                (cx, cy - 11), (cx + 10, cy - 8), (cx + 25, cy - 3),
                (cx + 38, cy + 1), (cx + 35, cy + 4), (cx + 20, cy + 2),
                (cx + 5, cy - 2), (cx - 10, cy - 3), (cx - 25, cy + 4),
                (cx - 38, cy + 6),
            ]
            top_curve = outline[:7]
            def base_y_func(t):
                if t < 0.5:
                    return cy + 6 - 17 * (t / 0.5)
                return cy - 11 + 12 * ((t - 0.5) / 0.5)

        cls._fpoly(img, outline, color=(0, 0, 0, 45))
        cls._poly(img, top_curve, thick=1.5, color=(0, 0, 0, 175))

        # hair strokes following growth direction
        rng = np.random.RandomState(42 + hash(style) % 100)
        for i in range(25):
            bx = cx - 34 + i * 2.8
            t_p = (bx - (cx - 38)) / 76.0
            by = base_y_func(t_p) + 3
            angle = 80 + (bx - cx) * 0.35
            rad = np.radians(angle)
            length = 5 + rng.random() * 2.5
            alpha = 140 + int(rng.random() * 70)
            cls._ln(img, bx, by,
                    bx + length * np.cos(rad), by - length * np.sin(rad),
                    thick=0.7, color=(0, 0, 0, alpha))

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
        cls._poly(img, [
            (200, 60), (185, 68), (170, 82), (158, 100), (149, 125),
            (143, 155), (140, 190), (138, 230), (139, 270), (142, 310),
            (148, 340), (158, 370), (170, 395), (184, 418), (195, 435),
            (200, 445),
            (205, 435), (216, 418), (230, 395), (242, 370), (252, 340),
            (258, 310), (261, 270), (262, 230), (260, 190), (257, 155),
            (251, 125), (242, 100), (230, 82), (215, 68), (200, 60),
        ], thick=3, closed=True, color=(0, 0, 0, 220))
        # jawline shadow
        cls._poly(img, [
            (158, 370), (170, 395), (184, 418), (195, 435), (200, 445),
            (205, 435), (216, 418), (230, 395), (242, 370),
        ], thick=1.5, color=(0, 0, 0, 70))
        # cheekbone hints
        cls._poly(img, [(143, 230), (145, 248), (148, 260)],
                  thick=1, color=(0, 0, 0, 55))
        cls._poly(img, [(257, 230), (255, 248), (252, 260)],
                  thick=1, color=(0, 0, 0, 55))
        faces.append(cls._finalize(img, w, h))

        # --- Round ---
        img = cls._canvas(w, h)
        cls._poly(img, [
            (200, 55), (180, 62), (162, 75), (148, 95), (137, 120),
            (130, 155), (126, 195), (125, 240), (127, 285), (132, 325),
            (140, 360), (152, 390), (168, 415), (185, 435), (200, 445),
            (215, 435), (232, 415), (248, 390), (260, 360), (268, 325),
            (273, 285), (275, 240), (274, 195), (270, 155), (263, 120),
            (252, 95), (238, 75), (220, 62), (200, 55),
        ], thick=3, closed=True, color=(0, 0, 0, 220))
        cls._poly(img, [(126, 220), (125, 260), (128, 300)],
                  thick=1.2, color=(0, 0, 0, 60))
        cls._poly(img, [(274, 220), (275, 260), (272, 300)],
                  thick=1.2, color=(0, 0, 0, 60))
        faces.append(cls._finalize(img, w, h))

        # --- Square / angular ---
        img = cls._canvas(w, h)
        cls._poly(img, [
            (200, 55), (182, 62), (165, 75), (152, 95), (144, 120),
            (140, 150), (138, 185), (137, 225), (137, 270), (137, 310),
            (137, 345), (140, 370), (148, 395), (162, 418), (180, 435),
            (192, 445), (200, 450),
            (208, 445), (220, 435), (238, 418), (252, 395), (260, 370),
            (263, 345), (263, 310), (263, 270), (263, 225), (262, 185),
            (260, 150), (256, 120), (248, 95), (235, 75), (218, 62),
            (200, 55),
        ], thick=3, closed=True, color=(0, 0, 0, 220))
        cls._cir(img, 140, 370, 3, fill=True, color=(0, 0, 0, 140))
        cls._cir(img, 260, 370, 3, fill=True, color=(0, 0, 0, 140))
        cls._poly(img, [(137, 345), (140, 370), (148, 395), (162, 418)],
                  thick=2.5, color=(0, 0, 0, 170))
        cls._poly(img, [(263, 345), (260, 370), (252, 395), (238, 418)],
                  thick=2.5, color=(0, 0, 0, 170))
        faces.append(cls._finalize(img, w, h))

        return faces

    # ── ear templates ──────────────────────────────────────────────────────

    @classmethod
    def _draw_ear_detail(cls, img, cx, cy, style='normal', flip=False):
        d = -1 if flip else 1

        if style == 'normal':
            cls._poly(img, [
                (cx, cy - 28), (cx + d * 10, cy - 32),
                (cx + d * 18, cy - 30), (cx + d * 25, cy - 22),
                (cx + d * 30, cy - 10), (cx + d * 32, cy),
                (cx + d * 30, cy + 12), (cx + d * 25, cy + 22),
                (cx + d * 18, cy + 30), (cx + d * 8, cy + 34),
                (cx, cy + 30),
            ], thick=2, color=(0, 0, 0, 200))
            cls._poly(img, [
                (cx + d * 5, cy - 20), (cx + d * 14, cy - 18),
                (cx + d * 20, cy - 10), (cx + d * 22, cy),
                (cx + d * 20, cy + 10), (cx + d * 14, cy + 18),
                (cx + d * 5, cy + 16),
            ], thick=1.2, color=(0, 0, 0, 120))
            cls._poly(img, [
                (cx, cy - 4), (cx + d * 6, cy - 2),
                (cx + d * 6, cy + 4), (cx, cy + 6),
            ], thick=1, color=(0, 0, 0, 130))
            cls._poly(img, [
                (cx, cy + 30), (cx + d * 3, cy + 34),
                (cx + d * 6, cy + 32), (cx + d * 8, cy + 28),
            ], thick=1.2, color=(0, 0, 0, 150))
        elif style == 'pointed':
            cls._poly(img, [
                (cx, cy - 28), (cx + d * 12, cy - 35),
                (cx + d * 22, cy - 38), (cx + d * 30, cy - 28),
                (cx + d * 32, cy - 14), (cx + d * 28, cy + 5),
                (cx + d * 20, cy + 22), (cx + d * 10, cy + 30),
                (cx, cy + 26),
            ], thick=2, color=(0, 0, 0, 200))
            cls._poly(img, [
                (cx + d * 6, cy - 22), (cx + d * 16, cy - 20),
                (cx + d * 22, cy - 10), (cx + d * 18, cy + 5),
                (cx + d * 10, cy + 15), (cx + d * 4, cy + 14),
            ], thick=1.2, color=(0, 0, 0, 120))
            cls._poly(img, [
                (cx, cy - 2), (cx + d * 5, cy - 1),
                (cx + d * 5, cy + 5), (cx, cy + 6),
            ], thick=1, color=(0, 0, 0, 130))
        else:  # small
            cls._poly(img, [
                (cx, cy - 18), (cx + d * 8, cy - 22),
                (cx + d * 16, cy - 18), (cx + d * 20, cy - 8),
                (cx + d * 20, cy + 4), (cx + d * 16, cy + 14),
                (cx + d * 8, cy + 22), (cx, cy + 18),
            ], thick=2, color=(0, 0, 0, 200))
            cls._poly(img, [
                (cx + d * 4, cy - 12), (cx + d * 10, cy - 10),
                (cx + d * 13, cy - 2), (cx + d * 10, cy + 8),
                (cx + d * 4, cy + 10),
            ], thick=1, color=(0, 0, 0, 110))
            cls._poly(img, [
                (cx, cy + 18), (cx + d * 2, cy + 22),
                (cx + d * 5, cy + 20), (cx + d * 6, cy + 16),
            ], thick=1, color=(0, 0, 0, 130))

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
        cls._poly(img, [
            (50, 160), (42, 120), (40, 90), (48, 60), (65, 38),
            (90, 22), (120, 12), (155, 6), (180, 4), (205, 6),
            (240, 12), (270, 22), (295, 38), (312, 60), (320, 90),
            (318, 120), (310, 160),
        ], thick=2.5, color=(0, 0, 0, 200))
        rng = np.random.RandomState(50)
        for x in range(70, 290, 6):
            t_p = (x - 50) / 260
            y_top = 8 + 50 * abs(t_p - 0.5) ** 1.5 * 4
            for y_off in range(0, 50, 10):
                yo = y_top + y_off + rng.uniform(-2, 2)
                if yo < 155:
                    angle = 75 + rng.uniform(-10, 10)
                    rad = np.radians(angle)
                    ln = 8 + rng.uniform(0, 5)
                    alpha = 70 + int(rng.random() * 80)
                    cls._ln(img, x, yo,
                            x + ln * np.cos(rad), yo + ln * np.sin(rad),
                            thick=0.6, color=(0, 0, 0, alpha))
        hairs.append(cls._finalize(img, w, h))

        # --- Long flowing ---
        img = cls._canvas(w, h)
        cls._poly(img, [
            (30, h), (25, 145), (28, 110), (38, 75), (55, 48),
            (80, 28), (115, 14), (155, 6), (180, 4), (205, 6),
            (245, 14), (280, 28), (305, 48), (322, 75), (332, 110),
            (335, 145), (330, h),
        ], thick=2.5, color=(0, 0, 0, 200))
        rng = np.random.RandomState(51)
        for x in range(45, 315, 5):
            t_p = (x - 30) / 300
            y_top = 8 + 55 * abs(t_p - 0.5) ** 1.4 * 4
            jitter = rng.uniform(-3, 3)
            alpha = 50 + int(rng.random() * 80)
            cls._ln(img, x + jitter, y_top + 8,
                    x + jitter + rng.uniform(-4, 4), h - 2,
                    thick=0.5, color=(0, 0, 0, alpha))
        hairs.append(cls._finalize(img, w, h))

        # --- Curly / wavy ---
        img = cls._canvas(w, h)
        cls._poly(img, [
            (30, 165), (18, 115), (22, 72), (38, 42), (65, 20),
            (100, 8), (145, 2), (180, 0), (215, 2), (260, 8),
            (295, 20), (322, 42), (338, 72), (342, 115), (330, 165),
        ], thick=2.5, color=(0, 0, 0, 200))
        rng = np.random.RandomState(52)
        for x in range(45, 318, 10):
            t_p = (x - 30) / 300
            y_top = 6 + 55 * abs(t_p - 0.5) ** 1.4 * 4
            for y in range(int(y_top) + 8, min(h - 8, int(y_top) + 120), 12):
                angle_start = rng.randint(0, 360)
                r = 4 + rng.random() * 3
                alpha = 70 + int(rng.random() * 80)
                cls._ell(img,
                         x + rng.uniform(-3, 3), y, r, r + 1,
                         rng.randint(-30, 30), angle_start, angle_start + 280,
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
