"""Train and compare crashed-vehicle classifiers, then keep the best.

Model choice here is measured, not asserted, because the obvious default was
wrong. Reaching for a small YOLO classification CNN is the reflex, but the
ACCIDENT benchmark evaluated this exact model family on this exact data and
found the ordering inverted:

    Qwen2.5-VL-7B  zero-shot .............. 0.119   (crop)
    Molmo-7B       zero-shot .............. 0.262
    DINOv2         linear probe ........... 0.363
    SigLIP2        linear probe ........... 0.471   <- best, and on CROPS

A frozen self-supervised backbone with a linear head beat 7B vision-language
models by roughly 4x. Two properties of our setting say the same thing:

* the training set is a few thousand crops, where end-to-end CNN training
  overfits and a frozen backbone plus light head is far more data-efficient;
* classification runs only on *candidate* stationary objects -- a handful per
  incident -- so a ViT is affordable even on CPU. It is triggered work, not
  per-frame work, which is the same shape as the rest of the pipeline.

So this script trains several candidates and selects on held-out validation.
Validation is split by clip upstream, so the same vehicle never appears in both
halves and the score is not inflated by near-duplicate crops.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

BACKBONES = {
    # timm names; the first that resolves in each family is used
    "siglip2": ["vit_base_patch16_siglip_224.v2_webli",
                "vit_base_patch16_siglip_224",
                "vit_base_patch16_clip_224.openai"],
    "dinov3": ["vit_small_patch16_dinov3.lvd1689m",
               "vit_small_patch14_dinov2.lvd142m"],
    "dinov2": ["vit_small_patch14_dinov2.lvd142m"],
    "convnext": ["convnext_tiny.fb_in22k"],
}


def load_split(root: Path, split: str):
    paths, labels = [], []
    for cls, y in (("normal", 0), ("crash", 1)):
        for p in sorted((root / split / cls).glob("*.jpg")):
            paths.append(p)
            labels.append(y)
    return paths, np.asarray(labels, dtype=np.int64)


def extract(paths, model, transform, device, batch: int = 64) -> np.ndarray:
    """Frozen forward pass -> pooled feature vectors."""
    from PIL import Image
    feats = []
    model.eval()
    with torch.inference_mode():
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i:i + batch]:
                im = Image.open(p).convert("RGB")
                imgs.append(transform(im))
            x = torch.stack(imgs).to(device)
            f = model(x)
            if f.ndim > 2:
                f = f.mean(dim=tuple(range(1, f.ndim - 1)))
            feats.append(f.float().cpu().numpy())
    return np.concatenate(feats, axis=0)


def probe(name: str, root: Path, device: str, out: Path) -> dict | None:
    """Frozen backbone + logistic-regression head."""
    import timm
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

    model = None
    used = None
    for cand in BACKBONES[name]:
        try:
            model = timm.create_model(cand, pretrained=True, num_classes=0)
            used = cand
            break
        except Exception:
            continue
    if model is None:
        print(f"  {name}: no checkpoint resolved, skipping")
        return None

    cfg = timm.data.resolve_model_data_config(model)
    transform = timm.data.create_transform(**cfg, is_training=False)
    model = model.to(device)

    tr_p, tr_y = load_split(root, "train")
    va_p, va_y = load_split(root, "val")
    if len(tr_p) < 40 or len(va_p) < 10:
        print(f"  {name}: not enough data ({len(tr_p)}/{len(va_p)})")
        return None

    t0 = time.perf_counter()
    Xtr = extract(tr_p, model, transform, device)
    Xva = extract(va_p, model, transform, device)
    feat_s = time.perf_counter() - t0

    clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
    clf.fit(Xtr, tr_y)
    pv = clf.predict_proba(Xva)[:, 1]

    res = {
        "model": f"{name} ({used}) + linear probe",
        "backbone": used,
        "feature_dim": int(Xtr.shape[1]),
        "train_n": len(tr_p), "val_n": len(va_p),
        "val_accuracy": round(float(accuracy_score(va_y, pv >= 0.5)), 4),
        "val_auc": round(float(roc_auc_score(va_y, pv)), 4) if len(set(va_y)) > 1 else None,
        "val_ap": round(float(average_precision_score(va_y, pv)), 4) if len(set(va_y)) > 1 else None,
        "feature_seconds": round(feat_s, 1),
    }

    # keep the artefacts so the winner can be wired straight into the pipeline
    import pickle
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    with open(d / "head.pkl", "wb") as f:
        pickle.dump({"clf": clf, "backbone": used, "cfg": cfg}, f)
    (d / "metrics.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    return res


def yolo_cls(root: Path, device: str, epochs: int, imgsz: int) -> dict | None:
    """The obvious default, kept as a baseline to measure the others against."""
    try:
        from ultralytics import YOLO
        m = YOLO("yolo11n-cls.pt")
        r = m.train(data=str(root.resolve()), epochs=epochs, imgsz=imgsz,
                    device=0 if device.startswith("cuda") else "cpu",
                    verbose=False, plots=False, project="models", name="crash_cls_yolo",
                    exist_ok=True)
        top1 = float(getattr(r, "top1", 0.0) or 0.0)
        if not top1:
            top1 = float(getattr(getattr(m, "metrics", None), "top1", 0.0) or 0.0)
        return {"model": "yolo11n-cls (fine-tuned end-to-end)",
                "val_accuracy": round(top1, 4), "val_auc": None, "val_ap": None}
    except Exception as exc:
        print(f"  yolo11n-cls failed: {type(exc).__name__}: {exc}")
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data/crash_cls")
    ap.add_argument("--out", default="models/crash_cls")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--models", nargs="+",
                    default=["siglip2", "dinov3", "convnext", "yolo"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--imgsz", type=int, default=128)
    args = ap.parse_args()

    root = Path(args.data)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    tr_p, tr_y = load_split(root, "train")
    va_p, va_y = load_split(root, "val")
    print(f"train {len(tr_p)} crops ({int(tr_y.sum())} crash / {int((1 - tr_y).sum())} normal)")
    print(f"val   {len(va_p)} crops ({int(va_y.sum())} crash / {int((1 - va_y).sum())} normal)")
    print(f"device: {args.device}\n")

    results = []
    for name in args.models:
        print(f"-- {name}")
        r = yolo_cls(root, args.device, args.epochs, args.imgsz) if name == "yolo" \
            else probe(name, root, args.device, out)
        if r:
            results.append(r)
            print(f"   acc={r['val_accuracy']}  auc={r.get('val_auc')}  ap={r.get('val_ap')}")

    if not results:
        print("no model trained successfully")
        return 1

    results.sort(key=lambda r: (r.get("val_auc") or 0, r["val_accuracy"]), reverse=True)
    print("\n" + "=" * 66)
    print(f"{'model':44s} {'acc':>6s} {'auc':>6s} {'ap':>6s}")
    print("-" * 66)
    for r in results:
        print(f"{r['model'][:44]:44s} {r['val_accuracy']:>6.3f} "
              f"{(r.get('val_auc') or 0):>6.3f} {(r.get('val_ap') or 0):>6.3f}")
    print("=" * 66)
    print(f"\nbest: {results[0]['model']}")

    (out / "comparison.json").write_text(
        json.dumps({"results": results, "best": results[0]}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
