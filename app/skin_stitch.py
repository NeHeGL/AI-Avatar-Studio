"""Stitch a seamless skin texture from the saved scan photos.

For each photo: warp every visible, front-facing face-mesh triangle into a common (frontal) layout, weighted by how
front-on, sharp and un-occluded it is. Photos are colour-matched to each other, then blended with multiband (Laplacian
pyramid) blending so there are no hard seams. Gaps are filled by inpainting; nostrils stay dark.

Input : <scan dir>/skin_atlas.json (layout written by the scan: points, triangles) + the saved scan photos
Output: <scan dir>/skin_stitched.png (RGBA, mesh area opaque) and skin_stitched_preview.jpg
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from app.landmarks import SCRIPT, _model, _python
from app.storage import MEDIA_DIR, get_working_profile
from app.walkthrough import media_url

LEVELS = 5


def _scan_dir(canonical_name: str) -> Path:
    from app.capture import _user_media_dir

    return MEDIA_DIR / _user_media_dir(canonical_name) / "scan"


def _ensure_landmarks(paths: list[Path]) -> None:
    todo = [p for p in paths if not p.with_suffix(".lm.json").exists()]
    if not todo:
        return
    py, model = _python(), _model()
    if py is None or model is None:
        raise RuntimeError("landmark helper unavailable (needs a Python with mediapipe and face_landmarker.task)")
    script = SCRIPT.parent / "detect_landmarks_batch.py"
    subprocess.run(
        [str(py), str(script), str(model), *[str(p) for p in todo]],
        capture_output=True,
        text=True,
        timeout=600,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )


def _area2(a, b, c):
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def stitch(canonical_name: str, tone: list[float] | None = None) -> dict:
    t_start = time.time()
    sd = _scan_dir(canonical_name)
    layout_path = sd / "skin_atlas.json"
    if not layout_path.exists():
        return {"status": "error", "message": "No scan layout yet. Run the face scan and tap Done first."}
    layout = json.loads(layout_path.read_text(encoding="utf-8"))
    S = int(layout["size"])
    canon = np.array(layout["points"], dtype=np.float64) * S  # (N,2) atlas coordinates
    tris = np.array(layout["triangles"], dtype=np.int32)
    N = len(canon)
    n_real = 478 if N >= 478 else N  # points beyond this are synthetic eye centres

    # Eye-centre points are the mean of the contour vertices they connect to.
    centres = {}
    for t in tris:
        for v in t:
            if v >= n_real:
                centres.setdefault(int(v), set()).update(int(x) for x in t if x < n_real)

    # the real-landmark photos (hero = "Center", plus the grid photos)
    profile = get_working_profile(canonical_name)
    paths = []
    for a in profile.scan_assets:
        p = Path(a.path)
        if p.name.startswith("angle_") and p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.exists():
            paths.append((p, (a.caption or "").strip().lower() == "center"))
    if not paths:
        return {"status": "error", "message": "No scan photos found."}
    _ensure_landmarks([p for p, _ in paths])

    # front-facing orientation of every triangle in the frontal layout
    sign0 = np.array([1 if _area2(canon[a], canon[b], canon[c]) >= 0 else -1 for a, b, c in tris])
    mesh_mask = np.zeros((S, S), np.uint8)
    for a, b, c in tris:
        cv2.fillConvexPoly(mesh_mask, np.round(canon[[a, b, c]]).astype(np.int32), 255)

    photos = []  # per photo: atlas image (float32 BGR), weight map (float32), yaw, sharpness
    for path, is_hero in paths:
        lm_file = path.with_suffix(".lm.json")
        if not lm_file.exists():
            continue
        data = json.loads(lm_file.read_text(encoding="utf-8"))
        if "landmarks" not in data:
            continue
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            continue
        H, W = img.shape[:2]
        L = np.array(data["landmarks"], dtype=np.float64)
        if len(L) < n_real:
            continue
        P = np.zeros((N, 3))
        P[:n_real] = L[:n_real] * [W, H, W]
        for ci, verts in centres.items():
            P[ci] = P[sorted(verts)].mean(axis=0)

        # winding vs the frontal layout: a mirrored photo flips every sign, so take the majority.
        signs = np.array([1 if _area2(P[a], P[b], P[c]) >= 0 else -1 for a, b, c in tris])
        agree = float((signs == sign0).mean())
        orient = 1 if agree >= 0.5 else -1
        front = (signs * orient) == sign0

        # facing = how directly the triangle faces the camera
        v1 = P[tris[:, 1]] - P[tris[:, 0]]
        v2 = P[tris[:, 2]] - P[tris[:, 0]]
        nrm = np.cross(v1, v2)
        facing = np.abs(nrm[:, 2]) / (np.linalg.norm(nrm, axis=1) + 1e-9)

        # visibility: paint triangle ids far-to-near, then see which survive at their centre (z-buffer)
        zs = P[tris, 2].mean(axis=1)
        order = [i for i in np.argsort(-zs) if front[i]]
        idmap = np.zeros((H, W, 3), np.uint8)
        for i in order:
            idv = int(i) + 1
            cv2.fillConvexPoly(idmap, np.round(P[tris[i], :2]).astype(np.int32), (idv & 255, (idv >> 8) & 255, (idv >> 16) & 255))
        visible = np.zeros(len(tris), bool)
        for i in order:
            a, b, c = tris[i]
            cen = P[[a, b, c], :2].mean(axis=0)
            for q in (cen, (cen + P[a, :2]) / 2, (cen + P[b, :2]) / 2, (cen + P[c, :2]) / 2):
                x, y = int(round(q[0])), int(round(q[1]))
                if 0 <= x < W and 0 <= y < H:
                    px = idmap[y, x]
                    if (int(px[0]) | (int(px[1]) << 8) | (int(px[2]) << 16)) == i + 1:
                        visible[i] = True
                        break

        # sharpness of the face area of this photo
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        x0, y0 = np.floor(P[:n_real, :2].min(axis=0)).astype(int)
        x1, y1 = np.ceil(P[:n_real, :2].max(axis=0)).astype(int)
        crop = gray[max(0, y0) : min(H, y1), max(0, x0) : min(W, x1)]
        sharp = float(cv2.Laplacian(crop, cv2.CV_64F).var()) if crop.size else 0.0

        atlas = np.zeros((S, S, 3), np.float32)
        wmap = np.zeros((S, S), np.float32)
        for i, (a, b, c) in enumerate(tris):
            if not (front[i] and visible[i]) or facing[i] < 0.22:
                continue
            dst = canon[[a, b, c]]
            src = P[[a, b, c], :2]
            bx0 = max(0, int(np.floor(dst[:, 0].min())) - 2)
            by0 = max(0, int(np.floor(dst[:, 1].min())) - 2)
            bx1 = min(S, int(np.ceil(dst[:, 0].max())) + 3)
            by1 = min(S, int(np.ceil(dst[:, 1].max())) + 3)
            if bx1 - bx0 < 2 or by1 - by0 < 2 or bx1 - bx0 > 600 or by1 - by0 > 600:
                continue
            M = cv2.getAffineTransform(src.astype(np.float32), (dst - [bx0, by0]).astype(np.float32))
            warped = cv2.warpAffine(img, M, (bx1 - bx0, by1 - by0), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            tmask = np.zeros((by1 - by0, bx1 - bx0), np.uint8)
            cen = (dst - [bx0, by0]).mean(axis=0)
            grown = (dst - [bx0, by0]) + np.sign((dst - [bx0, by0]) - cen) * 0.6  # close hairline gaps between triangles
            cv2.fillConvexPoly(tmask, np.round(grown).astype(np.int32), 255)
            sel = tmask > 0
            roi_a = atlas[by0:by1, bx0:bx1]
            roi_w = wmap[by0:by1, bx0:bx1]
            roi_a[sel] = warped[sel]
            roi_w[sel] = float(facing[i] ** 4)  # the most front-on photo of each patch dominates, which keeps detail sharp
        photos.append({"atlas": atlas, "w": wmap, "yaw": float(P[1, 0] - P[[33, 263], 0].mean()), "sharp": sharp, "hero": is_hero, "name": path.name})

    if not photos:
        return {"status": "error", "message": "Could not find the face in any saved photo."}
    smax = max(p["sharp"] for p in photos) or 1.0
    for p in photos:
        p["w"] *= float(np.clip(p["sharp"] / smax, 0.25, 1.0)) ** 2

    # ---- colour matching: bring every photo to the colour of the mosaic built so far (hero first) ----
    hero = next((p for p in photos if p["hero"]), max(photos, key=lambda p: p["w"].sum()))
    rest = sorted([p for p in photos if p is not hero], key=lambda p: abs(p["yaw"]))
    mos_num = hero["atlas"] * hero["w"][..., None]
    mos_den = hero["w"].copy()
    gains_log = []
    for p in rest:
        ref = mos_num / np.maximum(mos_den, 1e-6)[..., None]
        ov = (p["w"] > 0.08) & (mos_den > 0.08) & (mesh_mask > 0)
        if int(ov.sum()) > 1500:
            g = np.array([np.median(ref[..., k][ov]) / max(np.median(p["atlas"][..., k][ov]), 1.0) for k in range(3)])
            g = np.clip(g, 0.7, 1.4)
            p["atlas"] = np.clip(p["atlas"] * g, 0, 255)
            gains_log.append([round(float(x), 3) for x in g])
        mos_num += p["atlas"] * p["w"][..., None]
        mos_den += p["w"]

    # ---- multiband blend (Laplacian pyramids, weights smoothed through a Gaussian pyramid) ----
    def gpyr(im, n):
        out = [im]
        for _ in range(n):
            out.append(cv2.pyrDown(out[-1]))
        return out

    num = [None] * (LEVELS + 1)
    den = [None] * (LEVELS + 1)
    for p in photos:
        gi = gpyr(p["atlas"], LEVELS)
        gw = gpyr(p["w"], LEVELS)
        for k in range(LEVELS + 1):
            if k < LEVELS:
                lap = gi[k] - cv2.pyrUp(gi[k + 1], dstsize=(gi[k].shape[1], gi[k].shape[0]))
            else:
                lap = gi[k]
            w3 = gw[k][..., None]
            num[k] = lap * w3 if num[k] is None else num[k] + lap * w3
            den[k] = gw[k].copy() if den[k] is None else den[k] + gw[k]
    res = [num[k] / np.maximum(den[k], 1e-6)[..., None] for k in range(LEVELS + 1)]
    out = res[LEVELS]
    for k in range(LEVELS - 1, -1, -1):
        out = cv2.pyrUp(out, dstsize=(res[k].shape[1], res[k].shape[0])) + res[k]
    out = np.clip(out, 0, 255)
    # gentle sharpening to bring back skin detail lost to blending
    blur = cv2.GaussianBlur(out, (0, 0), 2.0)
    out = np.clip(out + 0.7 * (out - blur), 0, 255)

    covered = (sum(p["w"] for p in photos) > 0.02) & (mesh_mask > 0)
    coverage = float(covered.sum() / max(1, (mesh_mask > 0).sum()))

    # ---- fill whatever no photo saw: inpaint at low resolution, nostrils stay dark ----
    out8 = out.astype(np.uint8)
    hole = ((~covered) & (mesh_mask > 0)).astype(np.uint8) * 255
    if hole.any() and covered.any():
        sm = 512
        small = cv2.resize(out8, (sm, sm), interpolation=cv2.INTER_AREA)
        smask = cv2.resize(hole, (sm, sm), interpolation=cv2.INTER_NEAREST)
        smask = cv2.dilate(smask, np.ones((3, 3), np.uint8))
        filled = cv2.inpaint(small, smask, 6, cv2.INPAINT_TELEA)
        filled = cv2.GaussianBlur(cv2.resize(filled, (S, S), interpolation=cv2.INTER_CUBIC), (0, 0), 3)
        out8[hole > 0] = filled[hole > 0]
    if n_real > 327:
        d = float(np.hypot(*(canon[98] - canon[327])))
        r = 0.2 * d
        nos = np.zeros((S, S), np.uint8)
        for a, b, c in tris:
            cen = canon[[a, b, c]].mean(axis=0)
            if np.hypot(*(cen - canon[98])) < r or np.hypot(*(cen - canon[327])) < r:
                cv2.fillConvexPoly(nos, np.round(canon[[a, b, c]]).astype(np.int32), 255)
        dark = (nos > 0) & (hole > 0)
        out8[dark] = (34, 38, 70)  # BGR

    if tone and len(tone) == 3:  # the user's "less pink / warmer" sliders, as R,G,B gains
        out8 = np.clip(out8.astype(np.float32) * np.array([tone[2], tone[1], tone[0]], np.float32), 0, 255).astype(np.uint8)
    rgba = cv2.cvtColor(out8, cv2.COLOR_BGR2BGRA)
    rgba[..., 3] = mesh_mask
    cv2.imwrite(str(sd / "skin_stitched.png"), rgba)
    prev = cv2.resize(out8, (1024, 1024), interpolation=cv2.INTER_AREA)
    prev[cv2.resize(mesh_mask, (1024, 1024)) == 0] = 0
    cv2.imwrite(str(sd / "skin_stitched_preview.jpg"), prev, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return {
        "status": "ok",
        "url": media_url(str(sd / "skin_stitched.png")),
        "preview_url": media_url(str(sd / "skin_stitched_preview.jpg")),
        "photos_used": len(photos),
        "coverage_percent": round(100 * coverage),
        "colour_gains": gains_log[:6],
        "seconds": round(time.time() - t_start, 1),
    }
