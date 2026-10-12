#!/usr/bin/env python3
"""Evaluate the output of kitti_dynamic_eval (C++) against SemanticKITTI MOS labels.

Metrics are computed in two ways:
  sampled : only the points the rejection actually classified (exact original indices).
  full    : every original point; its label is taken from the nearest classified point
            within --radius (else static).
Moving = label ids 251..259 (lower 16 bits of the .label file): 252..259 SemanticKITTI, 251 the
generic MOS "moving" label used by HeLiMOS. --ignore drops points with the given label ids
(e.g. 0 = unlabeled, as in the MOS benchmark).

Frames without a .label file are skipped (sparse GT, e.g. DOALS converted by tools/doals/doals_to_kitti.py (FIDO_glim repo)).

Usage: kitti_dynamic_eval.py --seq <sequences/NN> --pred <dir> [--radius 0.5] [--json out.json]
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

MOVING = np.arange(251, 260)
BINS = [(0, 10), (10, 20), (20, 50), (0, 20), (0, 50), (0, float("inf"))]
CLASS_NAMES = {
    0: "unlabeled", 1: "outlier", 10: "car", 11: "bicycle", 13: "bus", 15: "motorcycle", 16: "on-rails",
    18: "truck", 20: "other-vehicle", 30: "person", 31: "bicyclist", 32: "motorcyclist", 40: "road",
    44: "parking", 48: "sidewalk", 49: "other-ground", 50: "building", 51: "fence", 52: "other-structure",
    60: "lane-marking", 70: "vegetation", 71: "trunk", 72: "terrain", 80: "pole", 81: "traffic-sign",
    99: "other-object",
}


def read_pred(path):
    raw = path.read_bytes()
    n = int(np.frombuffer(raw[:4], dtype=np.int32)[0])
    rec = np.frombuffer(raw[4:4 + 5 * n], dtype=np.dtype([("idx", "<i4"), ("dyn", "u1")]))
    return rec["idx"].astype(np.int64), rec["dyn"].astype(bool)


class Acc:
    def __init__(self):
        self.c = {b: np.zeros(3, dtype=np.int64) for b in BINS}  # TP FP FN

    def add(self, rng, pred, gt):
        for lo, hi in BINS:
            m = (rng >= lo) & (rng < hi)
            p, g = pred[m], gt[m]
            self.c[(lo, hi)] += [np.sum(p & g), np.sum(p & ~g), np.sum(~p & g)]

    def summary(self):
        out = {}
        for b, (tp, fp, fn) in self.c.items():
            P = tp / (tp + fp) if tp + fp else float("nan")
            R = tp / (tp + fn) if tp + fn else float("nan")
            iou = tp / (tp + fp + fn) if tp + fp + fn else float("nan")
            f1 = 2 * P * R / (P + R) if P + R > 0 else float("nan")
            out["all" if b[1] == float("inf") else f"{b[0]}-{b[1]}m"] = dict(P=P, R=R, IoU=iou, F1=f1, TP=int(tp), FP=int(fp), FN=int(fn))
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--radius", type=float, default=0.5)
    ap.add_argument("--json", default=None)
    ap.add_argument("--ignore", type=int, nargs="*", default=[], help="label ids excluded from the metrics")
    ap.add_argument("--frames", default=None, help="file with the frame numbers to evaluate (e.g. a test split)")
    args = ap.parse_args()

    seq, pred_dir = Path(args.seq), Path(args.pred)
    files = sorted(pred_dir.glob("*.pred"))
    if args.frames:
        keep = {int(l) for l in Path(args.frames).read_text().split() if l.strip()}
        files = [f for f in files if int(f.stem) in keep]
    sampled, full = Acc(), Acc()
    fp_classes = Counter()
    dyn_frac, n_frames = [], 0
    frame_iou = {20.0: [], float("inf"): []}   # per-frame IoU on the full scan (DOALS/Dynablox protocol)
    for f in files:
        i = int(f.stem)
        label_path = seq / "labels" / f"{i:06d}.label"
        if not label_path.exists():
            continue  # sparsely annotated datasets (e.g. DOALS): only frames with a .label are evaluated
        pts = np.fromfile(seq / "velodyne" / f"{i:06d}.bin", dtype=np.float32).reshape(-1, 4)[:, :3]
        lab = np.fromfile(label_path, dtype=np.uint32) & 0xFFFF
        gt = np.isin(lab, MOVING)
        rng = np.linalg.norm(pts, axis=1)
        idx, dyn = read_pred(f)
        if idx.size == 0:
            continue
        n_frames += 1
        dyn_frac.append(dyn.mean())
        keep = ~np.isin(lab, args.ignore)
        sampled.add(rng[idx][keep[idx]], dyn[keep[idx]], gt[idx][keep[idx]])

        d, nn = cKDTree(pts[idx]).query(pts, k=1, distance_upper_bound=args.radius)
        pred_full = np.zeros(len(pts), dtype=bool)
        ok = np.isfinite(d)
        pred_full[ok] = dyn[nn[ok]]
        full.add(rng[keep], pred_full[keep], gt[keep])
        for rmax, lst in frame_iou.items():
            m = keep & (rng < rmax)
            tp, fp, fn = np.sum(pred_full & gt & m), np.sum(pred_full & ~gt & m), np.sum(~pred_full & gt & m)
            if tp + fp + fn:
                lst.append(tp / (tp + fp + fn))
        fp_classes.update(lab[pred_full & ~gt & keep & (rng < 50)].tolist())

    res = dict(frames=n_frames, mean_dynamic_fraction=float(np.mean(dyn_frac)),
               frames_over_10pct_dynamic=int(np.sum(np.array(dyn_frac) > 0.10)),
               sampled=sampled.summary(), full=full.summary(),
               full_frame_mean_iou={("all" if r == float("inf") else f"0-{r:g}m"): (float(np.mean(v)) if v else float("nan"))
                                    for r, v in frame_iou.items()})
    tot = sum(fp_classes.values()) or 1
    res["fp_by_class_full_0-50m"] = {CLASS_NAMES.get(k, str(k)): round(v / tot, 3) for k, v in fp_classes.most_common(8)}

    print(f"frames={n_frames}  mean dyn frac={res['mean_dynamic_fraction']:.4f}  frames>10%={res['frames_over_10pct_dynamic']}")
    for kind in ("sampled", "full"):
        print(f"  [{kind}]")
        for b, m in res[kind].items():
            print(f"    {b:>7}: P={m['P']:.3f} R={m['R']:.3f} IoU={m['IoU']:.3f} F1={m['F1']:.3f}  (TP={m['TP']} FP={m['FP']} FN={m['FN']})")
    print("  FP by class (full, <50m):", res["fp_by_class_full_0-50m"])
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
