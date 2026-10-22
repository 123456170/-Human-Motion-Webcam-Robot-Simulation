from __future__ import annotations

import csv
import hashlib
import io
import math
import os
import shutil
import tempfile
import time
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import streamlit as st

try:
    import mediapipe as mp

    MP_AVAILABLE = True
except Exception:
    mp = None
    MP_AVAILABLE = False


st.set_page_config(
    page_title="Human Motion Webcam to Robot Simulation - Windows",
    page_icon="🤖",
    layout="wide",
)


PANEL_W = 480
PANEL_H = 540
VIDEO_W = 2 * PANEL_W
VIDEO_H = PANEL_H

PANEL_BG = (24, 27, 33)
HUMAN_COLOR = (255, 190, 90)
ROBOT_COLOR = (90, 220, 255)
OK_COLOR = (90, 220, 130)
WARN_COLOR = (0, 190, 255)
BAD_COLOR = (0, 80, 255)
TEXT_COLOR = (225, 225, 225)


ACTION_OPTIONS = [
    "Routine",
    "Wave",
    "Reach",
    "Squat",
    "Step",
    "Extreme lunge",
]

HUMAN_BONES = [
    ("pelvis", "torso"),
    ("torso", "neck"),
    ("neck", "head"),
    ("torso", "left_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("torso", "right_shoulder"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("pelvis", "left_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_toe"),
    ("pelvis", "right_hip"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_toe"),
]

ROBOT_BONES = HUMAN_BONES

HUMAN_UPPER_ARM = 0.28
HUMAN_FOREARM = 0.25
HUMAN_THIGH = 0.45
HUMAN_SHANK = 0.45

ROBOT_UPPER_ARM = 0.26
ROBOT_FOREARM = 0.23
ROBOT_THIGH = 0.34
ROBOT_SHANK = 0.32

ARM_SCALE = (ROBOT_UPPER_ARM + ROBOT_FOREARM) / (HUMAN_UPPER_ARM + HUMAN_FOREARM)
LEG_SCALE = (ROBOT_THIGH + ROBOT_SHANK) / (HUMAN_THIGH + HUMAN_SHANK)
TORSO_SCALE = 0.95
FOOT_HEIGHT = 0.05

ROBOT_SHOULDER_REL = {
    "left": np.array([-0.17, 0.27, 0.0], dtype=float),
    "right": np.array([0.17, 0.27, 0.0], dtype=float),
}

ROBOT_HIP_REL = {
    "left": np.array([-0.09, 0.0, 0.0], dtype=float),
    "right": np.array([0.09, 0.0, 0.0], dtype=float),
}

JOINT_LIMITS = {
    "left_shoulder": 2.60,
    "right_shoulder": 2.60,
    "left_elbow": 2.30,
    "right_elbow": 2.30,
    "left_hip": 1.80,
    "right_hip": 1.80,
    "left_knee": 2.20,
    "right_knee": 2.20,
    "torso": 0.80,
}

MP_KEY_INDICES = [
    0,
    11,
    12,
    13,
    14,
    15,
    16,
    23,
    24,
    25,
    26,
    27,
    28,
]

MP_BONES = [
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (25, 27),
    (24, 26),
    (26, 28),
]

MP_JOINT_INDEX = {
    "head": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}


def make_run_id() -> str:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rnd = int(np.random.default_rng().integers(1000, 9999))
    return f"run_{stamp}_{rnd}"


def open_camera(index: int) -> cv2.VideoCapture:
    """
    Open a webcam on Windows with several backend fallbacks.

    Windows commonly needs DSHOW or MSMF. Linux/macOS backends are also tried
    if present, but this function is optimized for Windows desktop use.
    """
    backends: List[Optional[int]] = [None]

    for name in (
        "CAP_DSHOW",
        "CAP_MSMF",
        "CAP_V4L2",
        "CAP_AVFOUNDATION",
        "CAP_GSTREAMER",
    ):
        value = getattr(cv2, name, None)
        if value is not None:
            backends.append(int(value))

    last_cap: Optional[cv2.VideoCapture] = None

    for backend in backends:
        try:
            if backend is None:
                cap = cv2.VideoCapture(index)
            else:
                cap = cv2.VideoCapture(index, backend)

            if not cap.isOpened():
                cap.release()
                continue

            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 15)

            ok, frame = cap.read()
            if ok and frame is not None:
                return cap

            cap.release()
            last_cap = cap
        except Exception:
            pass

    if last_cap is not None:
        return last_cap

    return cv2.VideoCapture(index)


def find_working_camera(
    preferred: int,
    max_scan: int = 4,
) -> Tuple[Optional[cv2.VideoCapture], int]:
    """
    Try the preferred camera index, then scan nearby indexes.

    Returns:
        cap: opened VideoCapture or None
        used_index: index that worked, or preferred if none worked
    """
    candidates = [preferred] + [i for i in range(max_scan) if i != preferred]

    for idx in candidates:
        cap = open_camera(idx)
        try:
            if cap is not None and cap.isOpened():
                ok, frame = cap.read()
                if ok and frame is not None:
                    return cap, idx
        except Exception:
            pass
        finally:
            if cap is not None and not (cap.isOpened() and ok):
                try:
                    cap.release()
                except Exception:
                    pass

    return None, preferred


def baseline_human() -> Dict[str, np.ndarray]:
    return {
        "pelvis": np.array([0.0, 0.95, 0.0], dtype=float),
        "torso": np.array([0.0, 1.25, 0.0], dtype=float),
        "neck": np.array([0.0, 1.42, 0.0], dtype=float),
        "head": np.array([0.0, 1.58, 0.0], dtype=float),
        "left_shoulder": np.array([-0.20, 1.38, 0.0], dtype=float),
        "right_shoulder": np.array([0.20, 1.38, 0.0], dtype=float),
        "left_elbow": np.array([-0.28, 1.08, 0.0], dtype=float),
        "right_elbow": np.array([0.28, 1.08, 0.0], dtype=float),
        "left_wrist": np.array([-0.34, 0.82, 0.0], dtype=float),
        "right_wrist": np.array([0.34, 0.82, 0.0], dtype=float),
        "left_hip": np.array([-0.10, 0.93, 0.0], dtype=float),
        "right_hip": np.array([0.10, 0.93, 0.0], dtype=float),
        "left_knee": np.array([-0.12, 0.50, 0.0], dtype=float),
        "right_knee": np.array([0.12, 0.50, 0.0], dtype=float),
        "left_ankle": np.array([-0.14, 0.08, 0.0], dtype=float),
        "right_ankle": np.array([0.14, 0.08, 0.0], dtype=float),
        "left_toe": np.array([-0.14, 0.05, 0.14], dtype=float),
        "right_toe": np.array([0.14, 0.05, 0.14], dtype=float),
    }


def envelope(progress: float, cycles: float) -> float:
    ramp = min(1.0, progress / 0.12, (1.0 - progress) / 0.12)
    ramp = max(0.0, ramp)
    osc = 0.5 + 0.5 * math.sin(2.0 * math.pi * cycles * progress - math.pi / 2.0)
    return float(ramp * osc)


def bump(progress: float, start: float, end: float) -> float:
    if end <= start:
        return 0.0
    if progress <= start or progress >= end:
        return 0.0
    x = (progress - start) / (end - start)
    return float(math.sin(math.pi * x) ** 2)


def place_two_bone(
    root: np.ndarray,
    tip: np.ndarray,
    l1: float,
    l2: float,
    bend_ref: np.ndarray,
    sign: float = 1.0,
    margin: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray]:
    root = np.asarray(root, dtype=float)
    tip = np.asarray(tip, dtype=float)
    bend_ref = np.asarray(bend_ref, dtype=float)

    dvec = tip - root
    d = float(np.linalg.norm(dvec))

    min_d = abs(l1 - l2) + margin
    max_d = l1 + l2 - margin

    if d < 1e-6:
        u = np.array([1.0, 0.0, 0.0], dtype=float)
        d = min_d
    else:
        u = dvec / d
        d = float(np.clip(d, min_d, max_d))

    tip2 = root + u * d

    a = (l1 * l1 - l2 * l2 + d * d) / (2.0 * d)
    h = math.sqrt(max(0.0, l1 * l1 - a * a))

    v = bend_ref - float(np.dot(bend_ref, u)) * u
    nv = float(np.linalg.norm(v))

    if nv < 1e-5:
        tmp = np.array([0.0, 1.0, 0.0], dtype=float)
        if abs(float(u[1])) > 0.9:
            tmp = np.array([1.0, 0.0, 0.0], dtype=float)
        v = np.cross(u, tmp)
        nv = float(np.linalg.norm(v))

    if nv < 1e-6:
        v = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        v = v / nv

    joint = root + u * a + sign * h * v
    return joint, tip2


def clamp_vector_length(v: np.ndarray, min_len: float, max_len: float) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))

    if n < 1e-6:
        return np.array([0.0, -min_len, 0.0], dtype=float)

    if n < min_len:
        return v * (min_len / n)

    if n > max_len:
        return v * (max_len / n)

    return v


def generate_human(
    action: str,
    progress: float,
    speed: float,
    rng: np.random.Generator,
) -> Dict[str, np.ndarray]:
    p = baseline_human()
    phase = 2.0 * math.pi * speed * progress
    a = action.lower()

    env = {
        "wave": 0.0,
        "reach": 0.0,
        "squat": 0.0,
        "step": 0.0,
        "extreme": 0.0,
    }

    if a == "wave":
        env["wave"] = envelope(progress, 4.0 * speed)
    elif a == "reach":
        env["reach"] = envelope(progress, 2.0 * speed)
    elif a == "squat":
        env["squat"] = envelope(progress, 1.0 * speed)
    elif a == "step":
        env["step"] = envelope(progress, 3.0 * speed)
    elif a == "extreme lunge":
        env["extreme"] = bump(progress, 0.08, 0.92)
    else:
        env["wave"] = bump(progress, 0.02, 0.28) * (
            0.55 + 0.45 * math.sin(2.0 * math.pi * speed * progress * 8.0)
        )
        env["reach"] = bump(progress, 0.28, 0.52) * (
            0.55 + 0.45 * math.sin(2.0 * math.pi * speed * progress * 4.0)
        )
        env["squat"] = bump(progress, 0.52, 0.78)
        env["step"] = bump(progress, 0.78, 0.99)

    for k in env:
        env[k] = float(np.clip(env[k], 0.0, 1.0))

    sq = env["squat"]
    stp = env["step"]
    ex = env["extreme"]

    p["pelvis"][1] -= 0.36 * sq
    p["pelvis"][1] += 0.02 * math.cos(phase * 12.0) * stp
    p["pelvis"][0] += 0.05 * math.sin(phase * 6.0) * stp
    p["pelvis"][0] += 0.32 * ex
    p["pelvis"][2] += 0.10 * ex

    lean_z = 0.10 * sq + 0.18 * env["reach"] + 0.22 * ex

    p["torso"] = p["pelvis"] + np.array([0.0, 0.30 - 0.04 * sq, lean_z], dtype=float)
    p["neck"] = p["pelvis"] + np.array([0.0, 0.47 - 0.05 * sq, 1.1 * lean_z], dtype=float)
    p["head"] = p["pelvis"] + np.array([0.0, 0.62 - 0.06 * sq, 1.1 * lean_z], dtype=float)

    p["left_hip"] = p["pelvis"] + np.array([-0.10, -0.02, 0.0], dtype=float)
    p["right_hip"] = p["pelvis"] + np.array([0.10, -0.02, 0.0], dtype=float)

    p["left_shoulder"] = p["torso"] + np.array([-0.20, 0.10, 0.0], dtype=float)
    p["right_shoulder"] = p["torso"] + np.array([0.20, 0.10, 0.0], dtype=float)

    base_rw = np.array([0.34, 0.82, 0.0], dtype=float)
    wave_rw = np.array(
        [0.58, 1.50 + 0.10 * math.sin(phase * 8.0), 0.12 * math.sin(phase * 4.0)],
        dtype=float,
    )
    reach_rw = np.array([0.36, 1.20, 0.62 + 0.08 * math.sin(phase * 2.0)], dtype=float)
    extreme_rw = np.array([0.92, 1.00, 0.78], dtype=float)

    rw = base_rw.copy()
    rw += env["wave"] * (wave_rw - base_rw)
    rw += env["reach"] * (reach_rw - base_rw)
    rw += env["extreme"] * (extreme_rw - base_rw)
    rw += 0.004 * rng.normal(size=3)
    p["right_wrist"] = rw

    lw = np.array([-0.34, 0.82, 0.0], dtype=float)
    lw[0] -= 0.08 * env["wave"] + 0.12 * env["extreme"]
    lw[1] -= 0.05 * sq
    lw[2] += 0.05 * env["reach"]
    lw += 0.003 * rng.normal(size=3)
    p["left_wrist"] = lw

    la = np.array(
        [-0.14 - 0.05 * sq - 0.22 * ex, 0.08, 0.04 * sq + 0.10 * ex],
        dtype=float,
    )
    ra = np.array(
        [0.14 + 0.05 * sq + 0.08 * ex, 0.08, 0.04 * sq - 0.18 * ex],
        dtype=float,
    )

    swing = math.sin(phase * 6.0)
    la[1] += 0.07 * max(0.0, swing) * stp
    ra[1] += 0.07 * max(0.0, -swing) * stp
    la[2] += 0.28 * swing * stp
    ra[2] += -0.28 * swing * stp

    p["left_ankle"] = la
    p["right_ankle"] = ra

    for side in ["left", "right"]:
        sh = p[f"{side}_shoulder"]
        wr = p[f"{side}_wrist"]
        bend_ref = np.array([0.0, 0.0, -1.0], dtype=float)
        el, wr_clamped = place_two_bone(
            sh,
            wr,
            HUMAN_UPPER_ARM,
            HUMAN_FOREARM,
            bend_ref,
            sign=1.0,
            margin=0.01,
        )
        p[f"{side}_elbow"] = el
        p[f"{side}_wrist"] = wr_clamped

    for side in ["left", "right"]:
        hip = p[f"{side}_hip"]
        ankle = p[f"{side}_ankle"]
        bend_ref = np.array([0.0, 0.0, 1.0], dtype=float)
        knee, ankle_clamped = place_two_bone(
            hip,
            ankle,
            HUMAN_THIGH,
            HUMAN_SHANK,
            bend_ref,
            sign=1.0,
            margin=0.01,
        )
        p[f"{side}_knee"] = knee
        p[f"{side}_ankle"] = ankle_clamped
        p[f"{side}_toe"] = ankle_clamped + np.array([0.0, -0.03, 0.14], dtype=float)

    return p


def make_raw_state(human: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    hp = np.asarray(human["pelvis"], dtype=float)

    pelvis_y = FOOT_HEIGHT + max(0.0, float(hp[1]) - 0.08) * LEG_SCALE
    pelvis = np.array([hp[0] * 0.45, pelvis_y, hp[2] * 0.45], dtype=float)

    torso_rel = (np.asarray(human["torso"], dtype=float) - hp) * TORSO_SCALE
    torso_rel[1] = float(np.clip(torso_rel[1], 0.18, 0.36))

    state: Dict[str, np.ndarray] = {
        "pelvis": pelvis,
        "torso_rel": torso_rel,
    }

    for side in ["left", "right"]:
        human_vec = (
            np.asarray(human[f"{side}_wrist"], dtype=float)
            - np.asarray(human[f"{side}_shoulder"], dtype=float)
        )
        raw_rel = human_vec * ARM_SCALE
        raw_rel = clamp_vector_length(
            raw_rel,
            0.08,
            ROBOT_UPPER_ARM + ROBOT_FOREARM - 0.01,
        )
        state[f"{side}_wrist_rel"] = raw_rel

        ha = np.asarray(human[f"{side}_ankle"], dtype=float)
        ankle_world = np.array(
            [
                ha[0] * LEG_SCALE,
                FOOT_HEIGHT + max(0.0, float(ha[1]) - 0.08) * LEG_SCALE,
                ha[2] * LEG_SCALE,
            ],
            dtype=float,
        )
        hip_world = pelvis + ROBOT_HIP_REL[side]
        raw_rel = ankle_world - hip_world
        raw_rel = clamp_vector_length(
            raw_rel,
            0.08,
            ROBOT_THIGH + ROBOT_SHANK - 0.01,
        )
        state[f"{side}_ankle_rel"] = raw_rel

    return state


def update_state_smooth(
    raw_state: Dict[str, np.ndarray],
    state: Dict[str, np.ndarray],
    dt: float,
    tau: float,
    max_vel: float,
) -> Dict[str, np.ndarray]:
    if not state:
        return {k: np.asarray(v, dtype=float).copy() for k, v in raw_state.items()}

    alpha = 1.0 - math.exp(-dt / max(tau, 1e-3))
    max_delta = max_vel * dt

    for k, raw_v in raw_state.items():
        raw_v = np.asarray(raw_v, dtype=float)
        old = np.asarray(state.get(k, raw_v), dtype=float)
        new = old + alpha * (raw_v - old)

        delta = new - old
        n = float(np.linalg.norm(delta))
        if n > max_delta and n > 1e-9:
            new = old + delta * (max_delta / n)

        state[k] = new

    return state


def compute_robot_from_state(
    pelvis: np.ndarray,
    state: Dict[str, np.ndarray],
    human: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    robot: Dict[str, np.ndarray] = {}
    pelvis = np.asarray(pelvis, dtype=float)

    robot["pelvis"] = pelvis.copy()
    robot["torso"] = pelvis + np.asarray(state["torso_rel"], dtype=float)
    robot["neck"] = robot["torso"] + np.array([0.0, 0.14, 0.0], dtype=float)
    robot["head"] = robot["neck"] + np.array([0.0, 0.12, 0.0], dtype=float)

    for side in ["left", "right"]:
        sh = pelvis + ROBOT_SHOULDER_REL[side]
        hip = pelvis + ROBOT_HIP_REL[side]

        robot[f"{side}_shoulder"] = sh
        robot[f"{side}_hip"] = hip

        wrist_tip = sh + np.asarray(state[f"{side}_wrist_rel"], dtype=float)
        elbow_bend = (
            np.asarray(human[f"{side}_elbow"], dtype=float)
            - 0.5
            * (
                np.asarray(human[f"{side}_shoulder"], dtype=float)
                + np.asarray(human[f"{side}_wrist"], dtype=float)
            )
        )
        elbow, wrist = place_two_bone(
            sh,
            wrist_tip,
            ROBOT_UPPER_ARM,
            ROBOT_FOREARM,
            elbow_bend,
            sign=1.0,
            margin=0.01,
        )
        robot[f"{side}_elbow"] = elbow
        robot[f"{side}_wrist"] = wrist

        ankle_tip = hip + np.asarray(state[f"{side}_ankle_rel"], dtype=float)
        knee_bend = (
            np.asarray(human[f"{side}_knee"], dtype=float)
            - 0.5
            * (
                np.asarray(human[f"{side}_hip"], dtype=float)
                + np.asarray(human[f"{side}_ankle"], dtype=float)
            )
        )
        knee, ankle = place_two_bone(
            hip,
            ankle_tip,
            ROBOT_THIGH,
            ROBOT_SHANK,
            knee_bend,
            sign=1.0,
            margin=0.01,
        )
        robot[f"{side}_knee"] = knee
        robot[f"{side}_ankle"] = ankle
        robot[f"{side}_toe"] = ankle + np.array([0.0, -0.03, 0.10], dtype=float)

    return robot


def convex_hull(points: np.ndarray) -> np.ndarray:
    pts = sorted(set((round(float(x), 6), round(float(y), 6)) for x, y in points))

    if len(pts) <= 1:
        return np.array(pts, dtype=float).reshape(-1, 2)

    def cross(
        o: Tuple[float, float],
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: List[Tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-9:
            lower.pop()
        lower.append(p)

    upper: List[Tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-9:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if not hull:
        hull = pts[:1]

    return np.array(hull, dtype=float)


def dist_point_segment(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    ap = p - a
    denom = float(np.dot(ab, ab))
    if denom < 1e-9:
        return float(np.linalg.norm(ap))
    t = float(np.dot(ap, ab) / denom)
    t = float(np.clip(t, 0.0, 1.0))
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def polygon_margin(poly: np.ndarray, point: np.ndarray) -> float:
    if len(poly) == 0:
        return -1.0

    if len(poly) < 3:
        min_d = min(float(np.linalg.norm(point - q)) for q in poly)
        return -min_d

    area = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    area *= 0.5

    if area < 0:
        poly = poly[::-1]

    if abs(area) < 1e-6:
        min_d = min(
            dist_point_segment(point, poly[i], poly[(i + 1) % n]) for i in range(n)
        )
        return -min_d

    inside = True
    min_d = float("inf")

    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        edge = b - a
        w = point - a
        c = float(edge[0] * w[1] - edge[1] * w[0])
        if c < -1e-5:
            inside = False
        min_d = min(min_d, dist_point_segment(point, a, b))

    return min_d if inside else -min_d


def compute_com(robot: Dict[str, np.ndarray]) -> np.ndarray:
    segments = [
        ("pelvis", "torso", 0.24),
        ("torso", "neck", 0.08),
        ("neck", "head", 0.07),
        ("left_shoulder", "left_elbow", 0.035),
        ("left_elbow", "left_wrist", 0.030),
        ("right_shoulder", "right_elbow", 0.035),
        ("right_elbow", "right_wrist", 0.030),
        ("left_hip", "left_knee", 0.075),
        ("left_knee", "left_ankle", 0.060),
        ("left_ankle", "left_toe", 0.020),
        ("right_hip", "right_knee", 0.075),
        ("right_knee", "right_ankle", 0.060),
        ("right_ankle", "right_toe", 0.020),
    ]

    com = np.zeros(3, dtype=float)
    total_mass = 0.0

    for a, b, m in segments:
        mid = 0.5 * (np.asarray(robot[a], dtype=float) + np.asarray(robot[b], dtype=float))
        com += m * mid
        total_mass += m

    if total_mass < 1e-9:
        return np.asarray(robot["pelvis"], dtype=float)

    return com / total_mass


def support_margin(robot: Dict[str, np.ndarray]) -> float:
    feet_points = np.array(
        [
            [robot["left_ankle"][0], robot["left_ankle"][2]],
            [robot["left_toe"][0], robot["left_toe"][2]],
            [robot["right_toe"][0], robot["right_toe"][2]],
            [robot["right_ankle"][0], robot["right_ankle"][2]],
        ],
        dtype=float,
    )

    hull = convex_hull(feet_points)
    com = compute_com(robot)
    com_xz = np.array([com[0], com[2]], dtype=float)

    return float(polygon_margin(hull, com_xz))


def angle_between(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))

    if na < 1e-9 or nb < 1e-9:
        return 0.0

    cos = float(np.dot(a, b) / (na * nb))
    cos = float(np.clip(cos, -1.0, 1.0))
    return float(math.acos(cos))


def extract_angles(pos: Dict[str, np.ndarray]) -> Dict[str, float]:
    down = np.array([0.0, -1.0, 0.0], dtype=float)
    up = np.array([0.0, 1.0, 0.0], dtype=float)
    angles: Dict[str, float] = {}

    for side in ["left", "right"]:
        sh = np.asarray(pos[f"{side}_shoulder"], dtype=float)
        el = np.asarray(pos[f"{side}_elbow"], dtype=float)
        wr = np.asarray(pos[f"{side}_wrist"], dtype=float)

        hip = np.asarray(pos[f"{side}_hip"], dtype=float)
        kn = np.asarray(pos[f"{side}_knee"], dtype=float)
        an = np.asarray(pos[f"{side}_ankle"], dtype=float)

        angles[f"{side}_shoulder"] = angle_between(el - sh, down)
        angles[f"{side}_elbow"] = angle_between(sh - el, wr - el)
        angles[f"{side}_hip"] = angle_between(kn - hip, down)
        angles[f"{side}_knee"] = angle_between(hip - kn, an - kn)

    angles["torso"] = angle_between(
        np.asarray(pos["torso"], dtype=float) - np.asarray(pos["pelvis"], dtype=float),
        up,
    )

    return angles


def angle_vector(robot: Dict[str, np.ndarray]) -> np.ndarray:
    angles = extract_angles(robot)
    keys = sorted(angles.keys())
    return np.array([angles[k] for k in keys], dtype=float)


def compute_joint_error(
    human: Dict[str, np.ndarray],
    robot: Dict[str, np.ndarray],
) -> float:
    ah = extract_angles(human)
    ar = extract_angles(robot)
    diffs = [ah[k] - ar[k] for k in ah if k in ar]

    if not diffs:
        return 0.0

    return float(np.sqrt(np.mean(np.square(diffs))))


def compute_ee_error(
    human: Dict[str, np.ndarray],
    state: Dict[str, np.ndarray],
    robot: Dict[str, np.ndarray],
) -> float:
    errors = []
    pelvis = np.asarray(state["pelvis"], dtype=float)

    for side in ["left", "right"]:
        sh = pelvis + ROBOT_SHOULDER_REL[side]
        human_vec = (
            np.asarray(human[f"{side}_wrist"], dtype=float)
            - np.asarray(human[f"{side}_shoulder"], dtype=float)
        )
        ref_wrist = sh + human_vec * ARM_SCALE
        errors.append(
            float(np.linalg.norm(np.asarray(robot[f"{side}_wrist"], dtype=float) - ref_wrist))
        )

        hip = pelvis + ROBOT_HIP_REL[side]
        ha = np.asarray(human[f"{side}_ankle"], dtype=float)
        ref_ankle = np.array(
            [
                ha[0] * LEG_SCALE,
                FOOT_HEIGHT + max(0.0, float(ha[1]) - 0.08) * LEG_SCALE,
                ha[2] * LEG_SCALE,
            ],
            dtype=float,
        )
        errors.append(
            float(np.linalg.norm(np.asarray(robot[f"{side}_ankle"], dtype=float) - ref_ankle))
        )

    if not errors:
        return 0.0

    return float(np.mean(errors) * 100.0)


def check_joint_violations(robot: Dict[str, np.ndarray]) -> List[str]:
    angles = extract_angles(robot)
    violations = []
    for name, value in angles.items():
        limit = JOINT_LIMITS.get(name, 3.0)
        if value > limit:
            violations.append(name)
    return violations


def check_self_collision(robot: Dict[str, np.ndarray]) -> bool:
    pairs = [
        ("left_wrist", "torso"),
        ("right_wrist", "torso"),
        ("left_wrist", "head"),
        ("right_wrist", "head"),
        ("left_wrist", "right_wrist"),
        ("left_ankle", "right_ankle"),
    ]

    for a, b in pairs:
        d = float(
            np.linalg.norm(
                np.asarray(robot[a], dtype=float) - np.asarray(robot[b], dtype=float)
            )
        )
        if d < 0.07:
            return True

    return False


def scale_state(state: Dict[str, np.ndarray], factor: float) -> Dict[str, np.ndarray]:
    out = {k: np.asarray(v, dtype=float).copy() for k, v in state.items()}

    out["pelvis"][0] *= factor
    out["pelvis"][2] *= factor
    out["torso_rel"][2] *= factor

    for key in [
        "left_wrist_rel",
        "right_wrist_rel",
        "left_ankle_rel",
        "right_ankle_rel",
    ]:
        out[key][0] *= factor
        out[key][2] *= factor

    return out


def freeze_state(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    out = {k: np.asarray(v, dtype=float).copy() for k, v in state.items()}

    out["pelvis"][0] *= 0.25
    out["pelvis"][2] *= 0.25
    out["torso_rel"][2] *= 0.15

    out["left_wrist_rel"] = np.array([-0.05, -0.42, 0.02], dtype=float)
    out["right_wrist_rel"] = np.array([0.05, -0.42, 0.02], dtype=float)

    out["left_ankle_rel"] = np.array([-0.05, -0.62, 0.0], dtype=float)
    out["right_ankle_rel"] = np.array([0.05, -0.62, 0.0], dtype=float)

    return out


def process_frame(
    human: Dict[str, np.ndarray],
    state: Dict[str, np.ndarray],
    dt: float,
    params: Dict[str, object],
) -> Tuple[
    Dict[str, np.ndarray],
    float,
    float,
    float,
    str,
    Dict[str, np.ndarray],
    Dict[str, object],
]:
    raw = make_raw_state(human)
    state = update_state_smooth(
        raw,
        state,
        dt=dt,
        tau=float(params["smoothing"]),
        max_vel=float(params["max_vel"]),
    )

    robot = compute_robot_from_state(state["pelvis"], state, human)
    margin = support_margin(robot)
    violations = check_joint_violations(robot)
    collision = check_self_collision(robot)

    intervention = ""
    min_margin = float(params["balance_margin"])

    if margin < min_margin or violations or collision:
        parts = []

        if margin < min_margin:
            parts.append("balance")
        if violations:
            parts.append("joint_limit")
        if collision:
            parts.append("collision")

        fixed = False

        for factor in (0.82, 0.65, 0.45, 0.25):
            trial_state = scale_state(state, factor)
            trial_robot = compute_robot_from_state(trial_state["pelvis"], trial_state, human)
            trial_margin = support_margin(trial_robot)
            trial_violations = check_joint_violations(trial_robot)
            trial_collision = check_self_collision(trial_robot)

            if trial_margin >= min_margin and not trial_violations and not trial_collision:
                state = trial_state
                robot = trial_robot
                margin = trial_margin
                violations = trial_violations
                collision = trial_collision
                fixed = True
                break

        if not fixed:
            state = freeze_state(state)
            robot = compute_robot_from_state(state["pelvis"], state, human)
            margin = support_margin(robot)
            violations = check_joint_violations(robot)
            collision = check_self_collision(robot)
            parts.append("freeze")

        intervention = "+".join(parts)

    ee_err = compute_ee_error(human, state, robot)
    joint_err = compute_joint_error(human, robot)

    flags = {
        "violations": violations,
        "collision": bool(collision),
    }

    return robot, margin, ee_err, joint_err, intervention, state, flags


def project_point(
    pt: np.ndarray,
    center: Tuple[int, int],
    scale: float,
    focal: float = 2.2,
) -> Tuple[int, int]:
    x, y, z = np.asarray(pt, dtype=float)
    denom = max(focal + float(z), 0.75)
    u = center[0] + scale * float(x) / denom
    v = center[1] - scale * float(y) / denom
    return int(round(u)), int(round(v))


def draw_skeleton(
    img: np.ndarray,
    pose: Dict[str, np.ndarray],
    bones: List[Tuple[str, str]],
    center: Tuple[int, int],
    scale: float,
    color: Tuple[int, int, int],
) -> None:
    for a, b in bones:
        if a not in pose or b not in pose:
            continue
        pa = project_point(pose[a], center, scale)
        pb = project_point(pose[b], center, scale)
        cv2.line(img, pa, pb, color, 4, cv2.LINE_AA)

    for name, pt in pose.items():
        if name in {"pelvis", "torso", "neck", "head"}:
            continue
        p = project_point(pt, center, scale)
        cv2.circle(img, p, 4, color, -1, cv2.LINE_AA)

    if "head" in pose:
        hp = project_point(pose["head"], center, scale)
        cv2.circle(img, hp, 10, color, 2, cv2.LINE_AA)


def draw_ground(
    img: np.ndarray,
    center: Tuple[int, int],
    scale: float,
) -> None:
    g1 = project_point(np.array([-0.95, 0.0, 0.0]), center, scale)
    g2 = project_point(np.array([0.95, 0.0, 0.0]), center, scale)
    cv2.line(img, g1, g2, (70, 78, 92), 2, cv2.LINE_AA)


def draw_progress_bar(img: np.ndarray, progress: float) -> None:
    x0 = 20
    x1 = PANEL_W - 20
    y = PANEL_H - 18
    cv2.rectangle(img, (x0, y), (x1, y + 6), (55, 58, 68), -1)
    w = int((x1 - x0) * float(np.clip(progress, 0.0, 1.0)))
    cv2.rectangle(img, (x0, y), (x0 + w, y + 6), (90, 200, 255), -1)


def draw_skeleton_panel(
    pose: Dict[str, np.ndarray],
    title: str,
    color: Tuple[int, int, int],
    progress: float = 0.0,
    show_metrics: bool = False,
    metrics: Optional[Dict[str, float]] = None,
) -> np.ndarray:
    panel = np.full((PANEL_H, PANEL_W, 3), PANEL_BG, dtype=np.uint8)

    center = (PANEL_W // 2, PANEL_H - 120)
    scale = 185.0

    draw_ground(panel, center, scale)
    draw_skeleton(panel, pose, HUMAN_BONES, center, scale, color)

    cv2.rectangle(panel, (0, 0), (PANEL_W, 38), (38, 42, 50), -1)
    cv2.putText(
        panel,
        title,
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    if show_metrics and metrics is not None:
        cv2.putText(
            panel,
            f"Joint err: {metrics.get('Joint', 0.0):.3f} rad",
            (15, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            f"EE err: {metrics.get('EE', 0.0):.1f} cm",
            (15, 95),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            TEXT_COLOR,
            1,
            cv2.LINE_AA,
        )

    draw_progress_bar(panel, progress)
    return panel


def draw_human_panel(
    human: Dict[str, np.ndarray],
    action: str,
    progress: float,
    show_metrics: bool,
    ee_err: float,
    joint_err: float,
) -> np.ndarray:
    panel = draw_skeleton_panel(
        human,
        "HUMAN MOTION",
        HUMAN_COLOR,
        progress=progress,
        show_metrics=show_metrics,
        metrics={"EE": ee_err, "Joint": joint_err},
    )

    cv2.putText(
        panel,
        f"Action: {action}",
        (15, PANEL_H - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    return panel


def draw_robot_panel(
    robot: Dict[str, np.ndarray],
    margin: float,
    ee_err: float,
    joint_err: float,
    action: str,
    progress: float,
    intervention: str,
    show_metrics: bool,
) -> np.ndarray:
    panel = draw_skeleton_panel(
        robot,
        "ROBOT SIMULATION",
        ROBOT_COLOR,
        progress=progress,
        show_metrics=show_metrics,
        metrics={"EE": ee_err, "Joint": joint_err},
    )

    if margin >= 0.03:
        margin_color = OK_COLOR
    elif margin >= 0.0:
        margin_color = WARN_COLOR
    else:
        margin_color = BAD_COLOR

    cv2.putText(
        panel,
        f"Driver: {action}",
        (15, PANEL_H - 95),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        panel,
        f"Balance margin: {margin * 100.0:.1f} cm",
        (15, PANEL_H - 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        margin_color,
        1,
        cv2.LINE_AA,
    )

    if intervention:
        cv2.putText(
            panel,
            f"Safety: {intervention}",
            (15, PANEL_H - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            BAD_COLOR,
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            panel,
            "Safety: nominal",
            (15, PANEL_H - 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            OK_COLOR,
            1,
            cv2.LINE_AA,
        )

    return panel


def draw_camera_panel(
    frame: np.ndarray,
    landmarks: Optional[object],
    title: str = "WEBCAM HUMAN",
) -> np.ndarray:
    panel = cv2.resize(frame, (PANEL_W, PANEL_H))

    if landmarks is not None:
        pts = []
        for lm in landmarks:
            pts.append((int(lm.x * PANEL_W), int(lm.y * PANEL_H)))

        for a, b in MP_BONES:
            if a < len(pts) and b < len(pts):
                cv2.line(panel, pts[a], pts[b], HUMAN_COLOR, 3, cv2.LINE_AA)

        for i in MP_KEY_INDICES:
            if i < len(pts):
                cv2.circle(panel, pts[i], 4, HUMAN_COLOR, -1, cv2.LINE_AA)
    else:
        cv2.putText(
            panel,
            "Waiting for pose...",
            (20, PANEL_H // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            WARN_COLOR,
            1,
            cv2.LINE_AA,
        )

    cv2.rectangle(panel, (0, 0), (PANEL_W, 38), (0, 0, 0), -1)
    cv2.putText(
        panel,
        title,
        (14, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )

    return panel


def put_image(placeholder: st.delta_generator.DeltaGenerator, img: np.ndarray) -> None:
    try:
        placeholder.image(img, channels="BGR", use_container_width=True)
    except Exception:
        try:
            placeholder.image(img, channels="BGR", use_column_width=True)
        except Exception:
            placeholder.image(img, channels="BGR")


def try_create_video_writer(
    prefix: str,
    fps: float,
    size: Tuple[int, int],
) -> Tuple[Optional[cv2.VideoWriter], str, str]:
    """
    Try several OpenCV codecs.

    On Windows, availability depends on the installed OpenCV build and system
    codecs. This function prefers browser-friendly formats first, then falls
    back to AVI.
    """
    candidates = [
        ("VP80", ".webm", "video/webm"),
        ("VP90", ".webm", "video/webm"),
        ("avc1", ".mp4", "video/mp4"),
        ("H264", ".mp4", "video/mp4"),
        ("mp4v", ".mp4", "video/mp4"),
        ("MJPG", ".avi", "video/x-msvideo"),
        ("XVID", ".avi", "video/x-msvideo"),
    ]

    for fourcc_text, ext, mime in candidates:
        path = prefix + ext
        try:
            fourcc = cv2.VideoWriter_fourcc(*fourcc_text)
            writer = cv2.VideoWriter(path, fourcc, float(fps), size)
            if writer.isOpened():
                return writer, path, mime
            writer.release()
        except Exception:
            pass

    return None, "", ""


def summarize_run(
    margins: List[float],
    ee_errors: List[float],
    joint_errors: List[float],
    jerks: List[float],
    latencies: List[float],
    intervention_count: int,
    fall_count: int,
    joint_limit_count: int,
    collision_count: int,
    pose_lost_count: int,
    frames: int,
    mode: str,
    action: str,
    run_id: str,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "run_id": run_id,
        "mode": mode,
        "action": action,
        "frames": int(frames),
        "margin_mean": 0.0,
        "margin_min": 0.0,
        "margin_p05": 0.0,
        "ee_err_mean": 0.0,
        "ee_err_p95": 0.0,
        "joint_err_mean": 0.0,
        "joint_err_p95": 0.0,
        "jerk_mean": 0.0,
        "jerk_p95": 0.0,
        "latency_p50_ms": 0.0,
        "latency_p95_ms": 0.0,
        "interventions": int(intervention_count),
        "intervention_rate": 0.0,
        "falls": int(fall_count),
        "fall_rate": 0.0,
        "joint_limit_events": int(joint_limit_count),
        "collision_events": int(collision_count),
        "pose_lost_frames": int(pose_lost_count),
    }

    if frames <= 0:
        return summary

    m = np.array(margins, dtype=float)
    e = np.array(ee_errors, dtype=float)
    j = np.array(joint_errors, dtype=float)
    k = np.array(jerks, dtype=float)
    l = np.array(latencies, dtype=float)

    summary["margin_mean"] = float(np.mean(m))
    summary["margin_min"] = float(np.min(m))
    summary["margin_p05"] = float(np.percentile(m, 5))

    summary["ee_err_mean"] = float(np.mean(e))
    summary["ee_err_p95"] = float(np.percentile(e, 95))

    summary["joint_err_mean"] = float(np.mean(j))
    summary["joint_err_p95"] = float(np.percentile(j, 95))

    summary["jerk_mean"] = float(np.mean(k))
    summary["jerk_p95"] = float(np.percentile(k, 95))

    summary["latency_p50_ms"] = float(np.percentile(l, 50))
    summary["latency_p95_ms"] = float(np.percentile(l, 95))

    summary["intervention_rate"] = float(intervention_count / max(1, frames))
    summary["fall_rate"] = float(fall_count / max(1, frames))

    return summary


def generate_auto_demo(params: Dict[str, object]) -> Tuple[bytes, str, Dict[str, object]]:
    fps = int(params["fps"])
    duration = float(params["duration"])
    n_frames = max(2, int(duration * fps))
    dt = 1.0 / float(fps)

    tmpdir = tempfile.mkdtemp(prefix="motion_auto_")
    prefix = os.path.join(tmpdir, "demo")

    writer: Optional[cv2.VideoWriter] = None

    try:
        writer, path, mime = try_create_video_writer(prefix, float(fps), (VIDEO_W, VIDEO_H))
        if writer is None:
            raise RuntimeError(
                "OpenCV video encoder unavailable on this Windows machine. "
                "Try Webcam live or Browser webcam mode. "
                "If you need downloadable Auto demo video, reinstall opencv-python "
                "in a fresh virtual environment."
            )

        rng = np.random.default_rng(int(params["seed"]))

        state: Dict[str, np.ndarray] = {}
        margins: List[float] = []
        ee_errors: List[float] = []
        joint_errors: List[float] = []
        jerks: List[float] = []
        latencies: List[float] = []
        interventions_list: List[str] = []

        prev_angles: Optional[np.ndarray] = None
        prev_vel: Optional[np.ndarray] = None

        fall_count = 0
        joint_limit_count = 0
        collision_count = 0

        for i in range(n_frames):
            progress = i / float(max(1, n_frames - 1))
            human = generate_human(
                str(params["action"]),
                progress,
                float(params["speed"]),
                rng,
            )

            t0 = time.perf_counter()
            robot, margin, ee_err, joint_err, intervention, state, flags = process_frame(
                human,
                state,
                dt,
                params,
            )
            latency = (time.perf_counter() - t0) * 1000.0

            av = angle_vector(robot)
            if prev_angles is None:
                jerk = 0.0
                vel = np.zeros_like(av)
            else:
                vel = (av - prev_angles) / dt
                acc = (vel - prev_vel) / dt if prev_vel is not None else np.zeros_like(vel)
                jerk = float(np.linalg.norm(acc))

            prev_angles = av
            prev_vel = vel

            margins.append(margin)
            ee_errors.append(ee_err)
            joint_errors.append(joint_err)
            jerks.append(jerk)
            latencies.append(latency)
            interventions_list.append(intervention)

            if margin < 0.0:
                fall_count += 1
            if flags["violations"]:
                joint_limit_count += 1
            if flags["collision"]:
                collision_count += 1

            left = draw_human_panel(
                human,
                str(params["action"]),
                progress,
                bool(params["show_metrics"]),
                ee_err,
                joint_err,
            )

            right = draw_robot_panel(
                robot,
                margin,
                ee_err,
                joint_err,
                str(params["action"]),
                progress,
                intervention,
                bool(params["show_metrics"]),
            )

            writer.write(np.hstack([left, right]))

        writer.release()
        writer = None

        with open(path, "rb") as f:
            video_bytes = f.read()

        summary = summarize_run(
            margins=margins,
            ee_errors=ee_errors,
            joint_errors=joint_errors,
            jerks=jerks,
            latencies=latencies,
            intervention_count=sum(1 for x in interventions_list if x),
            fall_count=fall_count,
            joint_limit_count=joint_limit_count,
            collision_count=collision_count,
            pose_lost_count=0,
            frames=n_frames,
            mode="auto_demo",
            action=str(params["action"]),
            run_id=make_run_id(),
        )

        return video_bytes, mime, summary

    finally:
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)


@st.cache_data(show_spinner=False)
def generate_auto_demo_cached(
    action: str,
    duration: float,
    fps: int,
    speed: float,
    smoothing: float,
    max_vel: float,
    balance_margin: float,
    seed: int,
    show_metrics: bool,
) -> Tuple[bytes, str, Dict[str, object]]:
    params = {
        "action": action,
        "duration": duration,
        "fps": fps,
        "speed": speed,
        "smoothing": smoothing,
        "max_vel": max_vel,
        "balance_margin": balance_margin,
        "seed": seed,
        "show_metrics": show_metrics,
        "min_conf": 0.5,
        "hold_frames": 0,
        "model_complexity": 0,
        "mirror": False,
        "camera_index": 0,
        "live_seconds": duration,
    }
    return generate_auto_demo(params)


def mp_key_visibility(landmarks: object) -> float:
    if landmarks is None:
        return 0.0

    vals = []
    for i in MP_KEY_INDICES:
        if i < len(landmarks):
            vals.append(float(getattr(landmarks[i], "visibility", 1.0)))

    return float(np.mean(vals)) if vals else 0.0


def mp_world_to_pose(results: object, min_conf: float) -> Optional[Dict[str, np.ndarray]]:
    if results is None:
        return None

    pose_landmarks = getattr(results, "pose_landmarks", None)
    world_landmarks = getattr(results, "pose_world_landmarks", None)

    if pose_landmarks is None or world_landmarks is None:
        return None

    landmarks = pose_landmarks.landmark
    if mp_key_visibility(landmarks) < min_conf:
        return None

    wl = world_landmarks.landmark

    def get(i: int) -> Optional[np.ndarray]:
        if i >= len(wl):
            return None
        p = np.array([wl[i].x, wl[i].y, wl[i].z], dtype=float)
        if not np.all(np.isfinite(p)):
            return None
        return p

    raw: Dict[str, np.ndarray] = {}
    for name, i in MP_JOINT_INDEX.items():
        p = get(i)
        if p is None:
            return None
        raw[name] = p

    left_hip = raw["left_hip"]
    right_hip = raw["right_hip"]
    pelvis_rel = 0.5 * (left_hip + right_hip)
    floor = np.array([0.0, 0.95, 0.0], dtype=float)

    joints: Dict[str, np.ndarray] = {"pelvis": floor.copy()}
    for name, p in raw.items():
        joints[name] = p - pelvis_rel + floor

    ls = joints["left_shoulder"]
    rs = joints["right_shoulder"]

    joints["torso"] = 0.5 * (ls + rs) + np.array([0.0, -0.08, 0.0], dtype=float)
    joints["neck"] = 0.5 * (ls + rs) + np.array([0.0, 0.06, 0.0], dtype=float)

    head = joints.get("head")
    if head is None or head[1] < joints["neck"][1]:
        joints["head"] = joints["neck"] + np.array([0.0, 0.18, 0.0], dtype=float)

    for side in ["left", "right"]:
        ankle = joints[f"{side}_ankle"]
        joints[f"{side}_toe"] = ankle + np.array([0.0, -0.03, 0.12], dtype=float)

    if joints["head"][1] < joints["pelvis"][1]:
        py = joints["pelvis"][1]
        for k in joints:
            joints[k][1] = 2.0 * py - joints[k][1]

    if joints["left_shoulder"][0] > joints["right_shoulder"][0]:
        swap_pairs = [
            ("left_shoulder", "right_shoulder"),
            ("left_elbow", "right_elbow"),
            ("left_wrist", "right_wrist"),
            ("left_hip", "right_hip"),
            ("left_knee", "right_knee"),
            ("left_ankle", "right_ankle"),
            ("left_toe", "right_toe"),
        ]
        for a, b in swap_pairs:
            joints[a], joints[b] = joints[b], joints[a]

    foot_keys = ["left_ankle", "right_ankle", "left_toe", "right_toe"]
    min_y = min(float(joints[k][1]) for k in foot_keys)
    if min_y < 0.02:
        shift = 0.02 - min_y
        for k in joints:
            joints[k][1] += shift

    return joints


def run_webcam_session(
    cap: cv2.VideoCapture,
    params: Dict[str, object],
    image_slot: Optional[st.delta_generator.DeltaGenerator] = None,
    status_slot: Optional[st.delta_generator.DeltaGenerator] = None,
    source_label: str = "Webcam live",
) -> Tuple[bytes, str, Dict[str, object], int, str]:
    if not cap.isOpened():
        raise RuntimeError("Webcam did not open. Check camera index and Windows camera permission.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    use_fps = float(params["fps"])
    dt = 1.0 / max(1.0, use_fps)
    max_seconds = float(params["live_seconds"])
    total_frames = max(1, int(max_seconds * use_fps))

    tmpdir = tempfile.mkdtemp(prefix="motion_webcam_")
    prefix = os.path.join(tmpdir, "session")

    writer: Optional[cv2.VideoWriter] = None
    path = ""
    mime = ""

    pose_detector = None
    pose_error: Optional[str] = None
    pose_failures = 0

    rng = np.random.default_rng(int(params["seed"]))

    try:
        if MP_AVAILABLE:
            try:
                pose_detector = mp.solutions.pose.Pose(
                    static_image_mode=False,
                    model_complexity=int(params["model_complexity"]),
                    min_detection_confidence=float(params["min_conf"]),
                    min_tracking_confidence=float(params["min_conf"]),
                )
            except Exception as exc:
                pose_error = f"MediaPipe init failed: {exc}"
        else:
            pose_error = "MediaPipe not installed; using synthetic fallback"

        writer, path, mime = try_create_video_writer(prefix, use_fps, (VIDEO_W, VIDEO_H))

        state: Dict[str, np.ndarray] = {}
        last_pose: Optional[Dict[str, np.ndarray]] = None
        miss = 0

        margins: List[float] = []
        ee_errors: List[float] = []
        joint_errors: List[float] = []
        jerks: List[float] = []
        latencies: List[float] = []
        interventions_list: List[str] = []

        prev_angles: Optional[np.ndarray] = None
        prev_vel: Optional[np.ndarray] = None

        fall_count = 0
        joint_limit_count = 0
        collision_count = 0
        pose_lost_count = 0

        for _ in range(5):
            ok, _ = cap.read()
            if ok:
                break

        start = time.time()
        frame_idx = 0
        empty_reads = 0
        last_status = 0.0

        while True:
            elapsed = time.time() - start
            if elapsed >= max_seconds or frame_idx >= total_frames:
                break

            target_time = start + frame_idx * dt
            now = time.time()
            if now < target_time:
                time.sleep(min(0.02, target_time - now))
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                empty_reads += 1
                if empty_reads > 30:
                    break
                continue

            empty_reads = 0

            if bool(params["mirror"]):
                frame = cv2.flip(frame, 1)

            t0 = time.perf_counter()

            landmarks = None
            pose3d = None
            valid = False
            driver = "pose"

            if pose_detector is not None:
                try:
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = pose_detector.process(rgb)

                    if results is not None and getattr(results, "pose_landmarks", None) is not None:
                        landmarks = results.pose_landmarks.landmark
                        pose3d = mp_world_to_pose(results, float(params["min_conf"]))
                        valid = pose3d is not None
                except Exception as exc:
                    pose_failures += 1
                    pose_error = f"Pose processing failed: {exc}"
                    if pose_failures >= 3:
                        pose_detector = None

            if pose3d is None:
                miss += 1
                pose_lost_count += 1

                if last_pose is not None and miss <= int(params["hold_frames"]):
                    pose3d = {k: v.copy() for k, v in last_pose.items()}
                    driver = "hold"
                else:
                    progress_fb = elapsed / max(1.0, max_seconds)
                    pose3d = generate_human("Routine", progress_fb, 1.0, rng)
                    driver = "fallback"
            else:
                last_pose = {k: v.copy() for k, v in pose3d.items()}
                miss = 0
                driver = "pose"

            robot, margin, ee_err, joint_err, intervention, state, flags = process_frame(
                pose3d,
                state,
                dt,
                params,
            )

            if not valid:
                intervention = f"{intervention}+{driver}" if intervention else driver

            latency = (time.perf_counter() - t0) * 1000.0

            av = angle_vector(robot)
            if prev_angles is None:
                jerk = 0.0
                vel = np.zeros_like(av)
            else:
                vel = (av - prev_angles) / dt
                acc = (vel - prev_vel) / dt if prev_vel is not None else np.zeros_like(vel)
                jerk = float(np.linalg.norm(acc))

            prev_angles = av
            prev_vel = vel

            margins.append(margin)
            ee_errors.append(ee_err)
            joint_errors.append(joint_err)
            jerks.append(jerk)
            latencies.append(latency)
            interventions_list.append(intervention)

            if margin < 0.0:
                fall_count += 1
            if flags["violations"]:
                joint_limit_count += 1
            if flags["collision"]:
                collision_count += 1

            progress = frame_idx / float(max(1, total_frames - 1))

            left = draw_camera_panel(
                frame,
                landmarks if valid else None,
                title="WEBCAM HUMAN" if valid else f"WEBCAM ({driver})",
            )

            right = draw_robot_panel(
                robot,
                margin,
                ee_err,
                joint_err,
                driver,
                progress,
                intervention,
                bool(params["show_metrics"]),
            )

            combined = np.hstack([left, right])

            if writer is not None:
                writer.write(combined)

            if image_slot is not None:
                put_image(image_slot, combined)

            if status_slot is not None and (
                time.time() - last_status > 0.2 or frame_idx == total_frames - 1
            ):
                fps_est = (frame_idx + 1) / max(elapsed, 1e-3)
                rec = "recording" if writer is not None else "live only"
                pose_txt = "pose OK" if valid else driver
                err_txt = pose_error or ""
                status_slot.text(
                    f"Running... frame {frame_idx + 1}/{total_frames} | "
                    f"{fps_est:.1f} fps | {pose_txt} | {rec} | {err_txt}"
                )
                last_status = time.time()

            frame_idx += 1

        if frame_idx == 0:
            raise RuntimeError(
                "No webcam frames captured. "
                f"camera_opened={cap.isOpened()}, "
                f"pose_error={pose_error}, "
                f"empty_reads={empty_reads}"
            )

        if writer is not None:
            writer.release()
            writer = None
            with open(path, "rb") as f:
                video_bytes = f.read()
        else:
            video_bytes = b""

        summary = summarize_run(
            margins=margins,
            ee_errors=ee_errors,
            joint_errors=joint_errors,
            jerks=jerks,
            latencies=latencies,
            intervention_count=sum(1 for x in interventions_list if x),
            fall_count=fall_count,
            joint_limit_count=joint_limit_count,
            collision_count=collision_count,
            pose_lost_count=pose_lost_count,
            frames=frame_idx,
            mode=source_label,
            action=source_label,
            run_id=make_run_id(),
        )

        diagnostics = (
            f"pose_error={pose_error or 'none'}; "
            f"recorder={'on' if path else 'off'}; "
            f"frames={frame_idx}"
        )

        return video_bytes, mime, summary, frame_idx, diagnostics

    finally:
        if pose_detector is not None:
            try:
                pose_detector.close()
            except Exception:
                pass
        if writer is not None:
            try:
                writer.release()
            except Exception:
                pass
        shutil.rmtree(tmpdir, ignore_errors=True)


def read_camera_bytes(obj: object) -> Optional[bytes]:
    """
    Convert Streamlit camera_input object to raw bytes.

    Streamlit versions differ slightly. This handles common object shapes.
    """
    if obj is None:
        return None

    if isinstance(obj, bytes):
        return obj

    if isinstance(obj, bytearray):
        return bytes(obj)

    if isinstance(obj, memoryview):
        return obj.tobytes()

    getvalue = getattr(obj, "getvalue", None)
    if callable(getvalue):
        try:
            data = getvalue()
            if isinstance(data, bytes):
                return data
            if isinstance(data, bytearray):
                return bytes(data)
        except Exception:
            pass

    read = getattr(obj, "read", None)
    seek = getattr(obj, "seek", None)
    if callable(read):
        try:
            if callable(seek):
                seek(0)
            data = read()
            if isinstance(data, bytes):
                return data
            if isinstance(data, bytearray):
                return bytes(data)
        except Exception:
            pass

    path = getattr(obj, "path", None)
    if isinstance(path, str) and os.path.exists(path):
        try:
            with open(path, "rb") as f:
                return f.read()
        except Exception:
            pass

    return None


def get_mp_pose_detector(params: Dict[str, object]) -> Tuple[Optional[object], str]:
    """
    Create or reuse a MediaPipe pose detector for browser snapshots.
    """
    if not MP_AVAILABLE:
        return None, "MediaPipe not installed; using synthetic fallback"

    key = f"{int(params['model_complexity'])}|{float(params['min_conf'])}"
    old_key = st.session_state.get("mp_pose_key")
    detector = st.session_state.get("mp_pose_detector")

    if old_key == key and detector is not None:
        return detector, ""

    if detector is not None:
        try:
            detector.close()
        except Exception:
            pass

    try:
        detector = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=int(params["model_complexity"]),
            min_detection_confidence=float(params["min_conf"]),
            min_tracking_confidence=float(params["min_conf"]),
        )
        st.session_state["mp_pose_detector"] = detector
        st.session_state["mp_pose_key"] = key
        return detector, ""
    except Exception as exc:
        st.session_state["mp_pose_detector"] = None
        st.session_state["mp_pose_key"] = None
        return None, f"MediaPipe init failed: {exc}"


def reset_browser_session() -> None:
    """
    Clear browser-webcam simulation state and history.
    """
    keys = [
        "browser_state",
        "browser_last_time",
        "browser_start",
        "browser_history",
        "browser_prev_angles",
        "browser_prev_vel",
        "browser_last_hash",
        "browser_last_combined",
    ]

    for key in keys:
        st.session_state.pop(key, None)


def render_browser_webcam(params: Dict[str, object]) -> None:
    """
    Browser-camera snapshot mode using st.camera_input.

    This avoids OpenCV server-side webcam permission problems on Windows.
    """
    st.subheader("Browser webcam snapshot → robot simulation")
    st.caption(
        "This mode uses the browser camera permission instead of OpenCV's "
        "server-side webcam capture. Click the camera button, capture one frame, "
        "and the humanoid robot simulation updates on the right."
    )

    if st.button("Reset browser session"):
        reset_browser_session()
        st.success("Browser session reset.")

    image_obj = st.camera_input("Capture one webcam frame", key="browser_cam_input")

    if image_obj is None:
        st.info("No frame captured yet. Click the camera button and take a picture.")
        return

    data = read_camera_bytes(image_obj)
    if data is None or len(data) == 0:
        st.error("Could not read the browser camera image.")
        return

    img_hash = hashlib.sha1(data).hexdigest()
    last_hash = st.session_state.get("browser_last_hash")

    if last_hash == img_hash:
        last_img = st.session_state.get("browser_last_combined")
        if last_img is not None:
            try:
                st.image(last_img, channels="BGR", use_container_width=True)
            except Exception:
                st.image(last_img, channels="BGR")
        st.info("Same frame already processed. Capture a new frame to update the robot.")
        return

    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if frame is None:
        st.error("Could not decode the browser camera image.")
        return

    if bool(params["mirror"]):
        frame = cv2.flip(frame, 1)

    now = time.time()

    start = st.session_state.get("browser_start")
    if start is None:
        start = now
        st.session_state["browser_start"] = start

    last = st.session_state.get("browser_last_time")
    if last is None:
        dt = 1.0 / max(1.0, float(params["fps"]))
    else:
        dt = max(0.033, min(0.20, now - float(last)))

    progress = ((now - float(start)) % 6.0) / 6.0

    t0 = time.perf_counter()

    detector, mp_error = get_mp_pose_detector(params)

    landmarks = None
    pose3d = None
    valid = False
    driver = "fallback"
    pose_error = mp_error

    if detector is not None:
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(rgb)

            if results is not None and getattr(results, "pose_landmarks", None) is not None:
                landmarks = results.pose_landmarks.landmark
                pose3d = mp_world_to_pose(results, float(params["min_conf"]))
                valid = pose3d is not None
        except Exception as exc:
            pose_error = f"Pose processing failed: {exc}"

    if pose3d is None:
        rng = np.random.default_rng(int(params["seed"]))
        pose3d = generate_human("Routine", progress, 1.0, rng)
        driver = "fallback"
    else:
        driver = "pose"

    state = st.session_state.get("browser_state", {})

    robot, margin, ee_err, joint_err, intervention, state, flags = process_frame(
        pose3d,
        state,
        dt,
        params,
    )

    if not valid:
        intervention = f"{intervention}+{driver}" if intervention else driver

    latency = (time.perf_counter() - t0) * 1000.0

    av = angle_vector(robot)
    prev_angles = st.session_state.get("browser_prev_angles")
    prev_vel = st.session_state.get("browser_prev_vel")

    if prev_angles is None:
        jerk = 0.0
        vel = np.zeros_like(av)
    else:
        vel = (av - np.asarray(prev_angles, dtype=float)) / dt

        if prev_vel is None:
            acc = np.zeros_like(vel)
        else:
            acc = (vel - np.asarray(prev_vel, dtype=float)) / dt

        jerk = float(np.linalg.norm(acc))

    st.session_state["browser_prev_angles"] = av
    st.session_state["browser_prev_vel"] = vel
    st.session_state["browser_state"] = state
    st.session_state["browser_last_time"] = now
    st.session_state["browser_last_hash"] = img_hash

    hist = st.session_state.get("browser_history")
    if hist is None:
        hist = {
            "margins": [],
            "ee_errors": [],
            "joint_errors": [],
            "jerks": [],
            "latencies": [],
            "interventions": [],
            "falls": [],
            "joint_limit_events": [],
            "collision_events": [],
            "pose_lost_frames": [],
        }
        st.session_state["browser_history"] = hist

    hist["margins"].append(margin)
    hist["ee_errors"].append(ee_err)
    hist["joint_errors"].append(joint_err)
    hist["jerks"].append(jerk)
    hist["latencies"].append(latency)
    hist["interventions"].append(intervention)
    hist["falls"].append(1 if margin < 0.0 else 0)
    hist["joint_limit_events"].append(1 if flags["violations"] else 0)
    hist["collision_events"].append(1 if flags["collision"] else 0)
    hist["pose_lost_frames"].append(0 if valid else 1)

    for key in list(hist.keys()):
        hist[key] = hist[key][-300:]

    left = draw_camera_panel(
        frame,
        landmarks if valid else None,
        title="BROWSER WEBCAM" if valid else f"BROWSER ({driver})",
    )

    right = draw_robot_panel(
        robot,
        margin,
        ee_err,
        joint_err,
        driver,
        progress,
        intervention,
        bool(params["show_metrics"]),
    )

    combined = np.hstack([left, right])
    st.session_state["browser_last_combined"] = combined

    try:
        st.image(combined, channels="BGR", use_container_width=True)
    except Exception:
        st.image(combined, channels="BGR")

    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Driver", driver)
    c2.metric("Balance margin", f"{margin * 100.0:.1f} cm")
    c3.metric("EE error", f"{ee_err:.1f} cm")
    c4.metric("Latency", f"{latency:.1f} ms")

    if pose_error:
        st.caption(f"Pose diagnostic: {pose_error}")

    frames = len(hist["margins"])

    if frames > 0:
        summary = summarize_run(
            margins=hist["margins"],
            ee_errors=hist["ee_errors"],
            joint_errors=hist["joint_errors"],
            jerks=hist["jerks"],
            latencies=hist["latencies"],
            intervention_count=sum(1 for x in hist["interventions"] if x),
            fall_count=sum(hist["falls"]),
            joint_limit_count=sum(hist["joint_limit_events"]),
            collision_count=sum(hist["collision_events"]),
            pose_lost_count=sum(hist["pose_lost_frames"]),
            frames=frames,
            mode="browser_webcam",
            action="browser_webcam",
            run_id=make_run_id(),
        )

        st.subheader("Browser session metrics")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Frames", f"{summary['frames']}")
        m2.metric("Mean margin", f"{float(summary['margin_mean']):.3f} m")
        m3.metric("Min margin", f"{float(summary['margin_min']):.3f} m")
        m4.metric("Interventions", f"{summary['interventions']}")

        st.download_button(
            label="Download browser metrics CSV",
            data=summary_to_csv(summary),
            file_name="metrics_browser_webcam.csv",
            mime="text/csv",
            key="dl_browser_metrics",
        )


def summary_to_csv(summary: Dict[str, object]) -> str:
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["metric", "value"])
    for k, v in summary.items():
        w.writerow([k, v])
    return buf.getvalue()


def display_result(
    video_bytes: bytes,
    mime: str,
    summary: Dict[str, object],
    label: str,
    diagnostics: str = "",
) -> None:
    col_video, col_metrics = st.columns([2, 1], gap="large")

    with col_video:
        st.subheader("Human webcam / synthetic motion → robot simulation")

        if video_bytes:
            ext = "webm" if "webm" in mime else ("mp4" if "mp4" in mime else "avi")
            try:
                st.video(
                    video_bytes,
                    format=mime,
                    loop=True,
                    autoplay=True,
                    muted=True,
                )
            except Exception as exc:
                st.warning(f"Video player could not display this file: {exc}")
                st.info("Use the download button below. The file may still be valid.")

            st.download_button(
                label="Download result video",
                data=video_bytes,
                file_name=f"motion_transfer_{label}.{ext}",
                mime=mime,
                key=f"dl_video_{label}",
            )
        else:
            st.warning(
                "Live session ran, but video encoding was unavailable. "
                "The live preview above still worked."
            )

        if diagnostics:
            st.caption(diagnostics)

    with col_metrics:
        st.subheader("Run metrics")
        st.metric("Run ID", str(summary.get("run_id", "n/a")))
        st.metric("Frames", f"{summary.get('frames', 0)}")
        st.metric("Mean balance margin", f"{float(summary.get('margin_mean', 0.0)):.3f} m")
        st.metric("Min balance margin", f"{float(summary.get('margin_min', 0.0)):.3f} m")
        st.metric(
            "Fall frames",
            f"{summary.get('falls', 0)} ({float(summary.get('fall_rate', 0.0)) * 100.0:.1f}%)",
        )
        st.metric(
            "Safety interventions",
            f"{summary.get('interventions', 0)} ({float(summary.get('intervention_rate', 0.0)) * 100.0:.1f}%)",
        )
        st.metric("Joint-limit events", f"{summary.get('joint_limit_events', 0)}")
        st.metric("Collision events", f"{summary.get('collision_events', 0)}")
        st.metric("Pose-lost frames", f"{summary.get('pose_lost_frames', 0)}")
        st.metric("Mean EE error", f"{float(summary.get('ee_err_mean', 0.0)):.1f} cm")
        st.metric("P95 EE error", f"{float(summary.get('ee_err_p95', 0.0)):.1f} cm")
        st.metric("Mean joint error", f"{float(summary.get('joint_err_mean', 0.0)):.3f} rad")
        st.metric("P95 joint error", f"{float(summary.get('joint_err_p95', 0.0)):.3f} rad")
        st.metric("Mean jerk", f"{float(summary.get('jerk_mean', 0.0)):.3f}")
        st.metric("Latency p50", f"{float(summary.get('latency_p50_ms', 0.0)):.1f} ms")
        st.metric("Latency p95", f"{float(summary.get('latency_p95_ms', 0.0)):.1f} ms")

        st.download_button(
            label="Download metrics CSV",
            data=summary_to_csv(summary),
            file_name=f"metrics_{label}.csv",
            mime="text/csv",
            key=f"dl_csv_{label}",
        )

        st.divider()
        st.caption(
            "All numbers above are measured from this simulated run. "
            "No physical robot is commanded."
        )


def render_auto_demo(params: Dict[str, object]) -> None:
    try:
        with st.spinner("Rendering automatic demo video..."):
            video_bytes, mime, summary = generate_auto_demo_cached(
                action=str(params["action"]),
                duration=float(params["duration"]),
                fps=int(params["fps"]),
                speed=float(params["speed"]),
                smoothing=float(params["smoothing"]),
                max_vel=float(params["max_vel"]),
                balance_margin=float(params["balance_margin"]),
                seed=int(params["seed"]),
                show_metrics=bool(params["show_metrics"]),
            )
        display_result(video_bytes, mime, summary, "auto_demo")
    except Exception as exc:
        st.error(f"Auto demo failed: {exc}")
        st.info("Try Webcam live mode or Browser webcam mode.")


def main() -> None:
    st.title("🤖 Human Motion Webcam → Robot Simulation - Windows")
    st.caption(
        "Live webcam human pose is mapped to a simulated humanoid robot side-by-side. "
        "Use Browser webcam if OpenCV cannot access the Windows camera."
    )

    if not MP_AVAILABLE:
        st.warning(
            "MediaPipe is not available. Webcam live and Browser webcam will still run, "
            "but they will use synthetic fallback motion instead of real pose estimation. "
            "Install requirements.txt for real webcam pose tracking."
        )

    with st.sidebar:
        st.header("Controls")

        mode = st.radio(
            "Mode",
            ["Auto demo", "Webcam live", "Browser webcam"],
            index=0,
        )

        if mode == "Browser webcam":
            fps_default = 10
        elif mode == "Webcam live":
            fps_default = 15
        else:
            fps_default = 24

        fps = st.select_slider(
            "Target FPS",
            options=[10, 15, 24, 30],
            value=fps_default,
        )

        smoothing = st.slider("Smoothing (s)", 0.01, 0.30, 0.08, 0.01)
        max_vel = st.slider("Max end-effector speed (m/s)", 0.2, 2.0, 0.9, 0.1)
        balance_margin = st.slider("Min balance margin (m)", 0.0, 0.10, 0.03, 0.005)
        min_conf = st.slider("Pose confidence", 0.10, 0.90, 0.35, 0.05)
        hold_frames = st.slider("Hold invalid pose frames", 0, 30, 10, 1)
        model_complexity = st.select_slider("Pose model complexity", options=[0, 1, 2], value=0)
        show_metrics = st.checkbox("Show metrics overlay", value=True)
        seed = st.number_input("Random seed", value=0, step=1)

        if mode == "Auto demo":
            action = st.selectbox("Motion", ACTION_OPTIONS, index=0)
            duration = st.slider("Duration (s)", 2.0, 8.0, 4.0, 0.5)
            speed = st.slider("Motion speed", 0.5, 2.0, 1.0, 0.1)
            live_seconds = duration
            camera_index = 0
            mirror = True
        elif mode == "Webcam live":
            action = "Webcam live"
            duration = 5.0
            speed = 1.0
            live_seconds = st.slider("Live seconds", 3.0, 20.0, 8.0, 1.0)
            camera_index = st.number_input("Camera index", 0, 10, 0, 1)
            mirror = st.checkbox("Mirror webcam", value=True)
        else:
            action = "Browser webcam"
            duration = 5.0
            speed = 1.0
            live_seconds = 5.0
            camera_index = 0
            mirror = st.checkbox("Mirror browser webcam", value=True)

        params = {
            "action": action,
            "duration": float(duration),
            "fps": int(fps),
            "speed": float(speed),
            "smoothing": float(smoothing),
            "max_vel": float(max_vel),
            "balance_margin": float(balance_margin),
            "min_conf": float(min_conf),
            "hold_frames": int(hold_frames),
            "model_complexity": int(model_complexity),
            "seed": int(seed),
            "show_metrics": bool(show_metrics),
            "mirror": bool(mirror),
            "camera_index": int(camera_index),
            "live_seconds": float(live_seconds),
        }

        with st.expander("Windows camera help"):
            st.markdown(
                """
                **If Webcam live fails:**
                1. Close Zoom, Teams, OBS, browser camera tabs, and other camera apps.
                2. Try camera index 0, 1, 2.
                3. Check Windows Settings → Privacy → Camera.
                4. Enable camera access for desktop apps.
                5. Restart the terminal and rerun `streamlit run app.py`.
                6. If OpenCV still cannot open the camera, use **Browser webcam** mode.

                **Browser webcam uses the browser permission.**
                Open the app on:

                ```text
                http://localhost:8501
                ```

                or:

                ```text
                http://127.0.0.1:8501
                ```
                """
            )

    if mode == "Auto demo":
        render_auto_demo(params)

    elif mode == "Webcam live":
        st.subheader("Webcam human motion → robot simulation")
        st.caption(
            "Press start to run a short live session. You should see the webcam human on "
            "the left and the simulated humanoid robot on the right, updating frame by frame."
        )

        if st.button("Start webcam session", type="primary"):
            image_slot = st.empty()
            status_slot = st.empty()

            cap = None
            try:
                cap, used_index = find_working_camera(int(params["camera_index"]), max_scan=4)

                if cap is None:
                    raise RuntimeError(
                        "Could not open webcam. Try camera index 1 or 2, "
                        "close other camera apps, check Windows camera permission, "
                        "or switch to Browser webcam mode."
                    )

                if used_index != int(params["camera_index"]):
                    st.info(f"Preferred camera index failed. Using camera index {used_index}.")

                video_bytes, mime, summary, frames, diagnostics = run_webcam_session(
                    cap=cap,
                    params=params,
                    image_slot=image_slot,
                    status_slot=status_slot,
                    source_label="Webcam live",
                )

                st.session_state["webcam_result"] = (
                    video_bytes,
                    mime,
                    summary,
                    diagnostics,
                )
                st.success(f"Webcam session finished: {frames} frames.")

            except Exception as exc:
                st.error(f"Webcam session failed: {exc}")
                st.session_state["webcam_result"] = None
            finally:
                if cap is not None:
                    try:
                        cap.release()
                    except Exception:
                        pass

        result = st.session_state.get("webcam_result")
        if result is not None:
            display_result(
                result[0],
                result[1],
                result[2],
                "webcam_live",
                result[3],
            )
        else:
            st.info("No webcam session yet. Showing automatic demo below.")
            render_auto_demo(params)

    else:
        render_browser_webcam(params)

    st.divider()
    st.markdown(
        "**Safety note:** This project is simulation-only. Before any real humanoid robot test, "
        "use reduced speed/torque limits, an emergency stop within reach, a harness or gantry for "
        "balance tests, enable feasibility filtering, and keep a human operator monitoring the system."
    )


if __name__ == "__main__":
    main()