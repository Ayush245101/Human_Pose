import argparse
import math
import os
import sys
from typing import List, Optional, Tuple

import cv2

# OpenPose COCO indices (subset)
PAIRS = [(1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13)]
NEEDED = sorted(set([i for a, b in PAIRS for i in (a, b)] + [0, 8, 11, 9, 12, 14, 15, 16, 17]))

DEFAULT_PROTOTXT = "pose_deploy_linevec.prototxt"
DEFAULT_MODEL = "pose_iter_440000.caffemodel"

_net: Optional[cv2.dnn_Net] = None
_net_key: Optional[Tuple[str, str]] = None


def _load_net(prototxt: str, model: str) -> cv2.dnn_Net:
    global _net, _net_key
    key = (os.path.abspath(prototxt), os.path.abspath(model))
    if _net is not None and _net_key == key:
        return _net
    if not (os.path.exists(prototxt) and os.path.exists(model)):
        raise FileNotFoundError(
            "Missing model files. Ensure pose_deploy_linevec.prototxt and pose_iter_440000.caffemodel are in the Space repo."
        )
    _net = cv2.dnn.readNetFromCaffe(prototxt, model)
    _net_key = key
    return _net


def _infer_points(
    frame_bgr,
    net: cv2.dnn_Net,
    thr: float = 0.15,
    in_w: int = 192,
    in_h: int = 192,
) -> List[Optional[Tuple[int, int]]]:
    H, W = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1 / 255.0, (in_w, in_h), (0, 0, 0), swapRB=False, crop=False)
    net.setInput(blob)
    out = net.forward()

    pts: List[Optional[Tuple[int, int]]] = [None] * out.shape[1]
    for pid in NEEDED:
        if pid >= out.shape[1]:
            continue
        hm = out[0, pid]
        _, conf, _, mp = cv2.minMaxLoc(hm)
        if conf >= thr:
            pts[pid] = (int(W * mp[0] / hm.shape[1]), int(H * mp[1] / hm.shape[0]))
    return pts


def _labels(pts: List[Optional[Tuple[int, int]]], H: int, mirrored: bool = False) -> str:
    lab: List[str] = []

    def _angle_deg(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
        # angle ABC (at point b)
        bax, bay = a[0] - b[0], a[1] - b[1]
        bcx, bcy = c[0] - b[0], c[1] - b[1]
        na = (bax * bax + bay * bay) ** 0.5
        nc = (bcx * bcx + bcy * bcy) ** 0.5
        if na <= 1e-6 or nc <= 1e-6:
            return 180.0
        cosang = (bax * bcx + bay * bcy) / (na * nc)
        cosang = max(-1.0, min(1.0, cosang))
        return float(math.degrees(math.acos(cosang)))

    hips = [p for p in (pts[8] if 8 < len(pts) else None, pts[11] if 11 < len(pts) else None) if p]
    knees = [p for p in (pts[9] if 9 < len(pts) else None, pts[12] if 12 < len(pts) else None) if p]
    # Standing/Sitting: prefer knee bend angle if hip-knee-ankle points exist;
    # otherwise fall back to hip->knee vertical separation.
    knee_angles: List[float] = []
    # Right side: hip(8), knee(9), ankle(10); Left side: hip(11), knee(12), ankle(13)
    rhip = pts[8] if 8 < len(pts) else None
    rknee = pts[9] if 9 < len(pts) else None
    rank = pts[10] if 10 < len(pts) else None
    lhip = pts[11] if 11 < len(pts) else None
    lknee = pts[12] if 12 < len(pts) else None
    lank = pts[13] if 13 < len(pts) else None

    if rhip and rknee and rank:
        knee_angles.append(_angle_deg(rhip, rknee, rank))
    if lhip and lknee and lank:
        knee_angles.append(_angle_deg(lhip, lknee, lank))

    if knee_angles:
        # Bent knee => sitting (or crouching). Straight knee => standing.
        # Typical: standing ~160-180 deg, sitting ~70-120 deg.
        min_angle = min(knee_angles)
        lab.append("Sitting" if min_angle < 140.0 else "Standing")
    elif hips and knees:
        dy = (sum(y for _, y in knees) / len(knees)) - (sum(y for _, y in hips) / len(hips))
        # Larger hip->knee distance usually indicates standing.
        # Use a slightly more forgiving threshold than before.
        stand_thr = max(40.0, 0.18 * H)
        lab.append("Standing" if dy >= stand_thr else "Sitting")

    m = 10
    rs, ls = (pts[2] if 2 < len(pts) else None), (pts[5] if 5 < len(pts) else None)
    rw, lw = (pts[4] if 4 < len(pts) else None), (pts[7] if 7 < len(pts) else None)
    r_up_img = bool(rw and rs and rw[1] < rs[1] - m)
    l_up_img = bool(lw and ls and lw[1] < ls[1] - m)
    r_up = l_up_img if mirrored else r_up_img
    l_up = r_up_img if mirrored else l_up_img
    if r_up and l_up:
        lab.append("Both Hands Raised")
    elif r_up:
        lab.append("Right Hand Raised")
    elif l_up:
        lab.append("Left Hand Raised")

    le, re = (pts[15] if 15 < len(pts) else None), (pts[14] if 14 < len(pts) else None)
    la, ra = (pts[17] if 17 < len(pts) else None), (pts[16] if 16 < len(pts) else None)
    if (le is None) ^ (re is None) or (la is None) ^ (ra is None):
        lab.append("Not Looking")

    return " | ".join(lab) if lab else "Pose Uncertain"


def _draw(frame_bgr, pts: List[Optional[Tuple[int, int]]], label: str):
    for p in pts:
        if p:
            cv2.circle(frame_bgr, p, 4, (0, 255, 255), -1, cv2.LINE_AA)
    for a, b in PAIRS:
        pa = pts[a] if a < len(pts) else None
        pb = pts[b] if b < len(pts) else None
        if pa and pb:
            cv2.line(frame_bgr, pa, pb, (0, 255, 0), 2, cv2.LINE_AA)

    cv2.putText(frame_bgr, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame_bgr


def _center_and_move(
    pts: List[Optional[Tuple[int, int]]],
    prev_center: Optional[Tuple[int, int]],
    mirrored: bool = False,
) -> Tuple[Optional[Tuple[int, int]], float, str]:
    centers = [p for p in (pts[1] if 1 < len(pts) else None, pts[8] if 8 < len(pts) else None, pts[11] if 11 < len(pts) else None) if p]
    center = (int(sum(x for x, _ in centers) / len(centers)), int(sum(y for _, y in centers) / len(centers))) if centers else None
    if center and prev_center:
        dx, dy = center[0] - prev_center[0], center[1] - prev_center[1]
        if mirrored:
            dx = -dx
        mv = float((dx * dx + dy * dy) ** 0.5)
        if mv >= 8:
            if abs(dx) >= abs(dy):
                direction = "Moving Right" if dx > 0 else "Moving Left"
            else:
                direction = "Moving Down" if dy > 0 else "Moving Up"
        else:
            direction = ""
        return center, mv, direction
    return center, 0.0, ""

def _open_capture(source: str, cam: int, max_cams: int) -> Optional[cv2.VideoCapture]:
    if source.lower() != "webcam":
        c = cv2.VideoCapture(source)
        return c if c.isOpened() else None

    max_cams = max(1, int(max_cams))
    cam_indices = [cam] + [i for i in range(max_cams) if i != cam]
    backends = [cv2.CAP_ANY]
    if sys.platform.startswith("win"):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]

    for idx in cam_indices:
        for backend in backends:
            c = cv2.VideoCapture(idx, backend)
            if not c.isOpened():
                c.release()
                continue
            ok, _ = c.read()
            if ok:
                return c
            c.release()
    return None


def _run_local(
    source: str,
    cam: int,
    max_cams: int,
    flip: bool,
    prototxt: str,
    model: str,
    thr: float,
    in_w: int,
    in_h: int,
    infer_every: int,
) -> int:
    if not (os.path.exists(prototxt) and os.path.exists(model)):
        print("ERROR: model files missing")
        return 2

    cap = _open_capture(source, cam, max_cams=max_cams)
    if cap is None or not cap.isOpened():
        if source.lower() == "webcam":
            print(
                "ERROR: Could not open any webcam. Close other apps using the camera and try --cam 0/1/2/... or increase --max-cams."
            )
        else:
            print(f"ERROR: Could not open source: {source}")
        return 2

    net = _load_net(prototxt, model)
    every = max(1, int(infer_every))
    last_pts: Optional[List[Optional[Tuple[int, int]]]] = None
    last_lbl = "Warming up"
    i = 0
    prev_center: Optional[Tuple[int, int]] = None
    last_move = 0.0
    last_dir = ""
    mirrored = bool(flip and source.lower() == "webcam")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if mirrored:
            frame = cv2.flip(frame, 1)

        if (i % every) == 0 or last_pts is None:
            last_pts = _infer_points(frame, net, thr=thr, in_w=in_w, in_h=in_h)
            last_lbl = _labels(last_pts, frame.shape[0], mirrored=mirrored)
            prev_center, mv, direction = _center_and_move(last_pts, prev_center, mirrored=mirrored)
            if mv:
                last_move = mv
            last_dir = direction

        frame = _draw(frame, last_pts, last_lbl)
        if prev_center is not None:
            cv2.circle(frame, prev_center, 6, (255, 0, 0), -1, cv2.LINE_AA)
            cv2.putText(
                frame,
                f"Move: {last_move:.1f}px" + (f" ({last_dir})" if last_dir else ""),
                (10, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        cv2.imshow("Pose", frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()
    return 0


def main() -> int:
    p = argparse.ArgumentParser("Human Pose (VS Code)")
    p.add_argument("--source", default="webcam")
    p.add_argument("--cam", type=int, default=0)
    p.add_argument("--max-cams", type=int, default=10, help="How many camera indices to try (0..max-cams-1)")
    p.add_argument("--flip", action="store_true")
    p.add_argument("--prototxt", default=DEFAULT_PROTOTXT)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--thr", type=float, default=0.15)
    p.add_argument("--in-w", type=int, default=192)
    p.add_argument("--in-h", type=int, default=192)
    p.add_argument("--infer-every", type=int, default=3)
    a = p.parse_args()

    return _run_local(
        source=a.source,
        cam=a.cam,
        max_cams=a.max_cams,
        flip=a.flip,
        prototxt=a.prototxt,
        model=a.model,
        thr=a.thr,
        in_w=a.in_w,
        in_h=a.in_h,
        infer_every=a.infer_every,
    )


if __name__ == "__main__":
    raise SystemExit(main())
