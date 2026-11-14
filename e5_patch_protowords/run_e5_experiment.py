"""Run the complete teammate-provided E5 patch-ProtoWords experiment.

This single-file version combines:
  A) sample patch tokens from training images;
  B) fit MiniBatchKMeans patch codebooks;
  C) create per-image patch histograms for train/val/test;
  D/E) classify patch histograms and compare with continuous DINOv2 features.

The experimental operations and hyperparameters from the four source scripts
are preserved. Portability changes: project-root discovery, CUDA/MPS/CPU device
selection, clearer file checks, output-directory creation, and a unified CLI.
"""
import argparse
import json
import os
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode

CACHE_NAME = "cifar10_dinov2_vits14_224px_train45000_val5000_test10000_seed42.npz"


def project_root_from(script_file: str, explicit_root: str | None) -> Path:
    if explicit_root:
        root = Path(explicit_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Project root does not exist: {root}")
        return root
    env_root = os.environ.get("PROTOWORDS_ROOT")
    if env_root:
        root = Path(env_root).expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"PROTOWORDS_ROOT does not exist: {root}")
        return root
    starts = [Path(script_file).resolve().parent, Path.cwd().resolve()]
    seen = set()
    for start in starts:
        for candidate in (start, *start.parents):
            if candidate in seen:
                continue
            seen.add(candidate)
            if (candidate / "outputs" / CACHE_NAME).is_file():
                return candidate
    raise FileNotFoundError(
        f"Could not locate the ProtoWords project root. Expected outputs/{CACHE_NAME}. "
        "Pass --project-root /path/to/ProtoWords or set PROTOWORDS_ROOT."
    )


def resolve_under_root(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            selected = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            selected = "mps"
        else:
            selected = "cpu"
    else:
        selected = requested
    if selected == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if selected == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available in this PyTorch installation.")
    return torch.device(selected)


def load_dinov2(device: torch.device):
    print("Loading DINOv2 ViT-S/14 (first run may download the model weights)...")
    return torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device).eval()


def image_transform():
    # Keep the exact image preprocessing used in the supplied experiment.
    return transforms.Compose([
        transforms.Resize(224, interpolation=InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


def stage_a(root: Path, outdir: Path, cache_path: Path, model, device,
            n_images: int | None, patches_per_image: int, batch_size: int, seed: int):
    print("\n================ E5 STAGE A: PATCH SAMPLING ================")
    if not (root / "data").is_dir():
        raise FileNotFoundError(f"CIFAR-10 data directory not found: {root / 'data'}")
    with np.load(cache_path) as z:
        if "train_indices" not in z.files:
            raise KeyError("train_indices is missing from the shared feature cache")
        train_idx = z["train_indices"].copy()
    if n_images is not None:
        train_idx = train_idx[:n_images]
    print("train images used:", len(train_idx))

    ds = datasets.CIFAR10(str(root / "data"), train=True, download=False,
                          transform=image_transform())
    loader = DataLoader(Subset(ds, train_idx.tolist()), batch_size=batch_size,
                        shuffle=False, num_workers=0)
    rng = np.random.default_rng(seed)
    n_total = len(train_idx) * patches_per_image
    sample = np.empty((n_total, 384), dtype=np.float16)
    pos, t0 = 0, time.time()
    with torch.no_grad():
        for b, (x, _) in enumerate(loader):
            tokens = model.forward_features(x.to(device))["x_norm_patchtokens"]
            tokens = torch.nn.functional.normalize(tokens, dim=-1).float().cpu().numpy()
            for t in tokens:
                keep = rng.choice(256, size=patches_per_image, replace=False)
                sample[pos:pos + patches_per_image] = t[keep].astype(np.float16)
                pos += patches_per_image
            if b % 50 == 0:
                print(f"batch {b}/{len(loader)}  patches so far {pos}  {time.time()-t0:.0f}s")
    assert pos == n_total
    sample_path = outdir / "patch_sample_train.npy"
    np.save(sample_path, sample)
    print("saved", sample_path, sample.shape, sample.dtype, f"{time.time()-t0:.0f}s")


def stage_b(outdir: Path, ks: list[int], seed: int):
    print("\n================ E5 STAGE B: CODEBOOK FITTING ================")
    sample_path = outdir / "patch_sample_train.npy"
    if not sample_path.is_file():
        raise FileNotFoundError(f"Patch sample not found: {sample_path}. Run Stage A first.")
    X = np.load(sample_path).astype(np.float32)
    print("sample:", X.shape)
    for K in ks:
        if K > len(X):
            raise ValueError(f"K={K} exceeds number of sampled patches ({len(X)})")
        t0 = time.time()
        km = MiniBatchKMeans(n_clusters=K, batch_size=10000, n_init=3,
                             max_iter=100, random_state=seed)
        km.fit(X)
        counts = np.bincount(km.predict(X), minlength=K)
        np.save(outdir / f"centroids_K{K}.npy", km.cluster_centers_.astype(np.float32))
        info = {"K": K, "inertia": float(km.inertia_), "seed": seed,
                "min_cluster_size": int(counts.min()), "max_cluster_size": int(counts.max()),
                "empty_clusters": int((counts == 0).sum()),
                "fit_seconds": round(time.time() - t0)}
        with open(outdir / f"codebook_K{K}_meta.json", "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2)
        print(info)


def stage_c(root: Path, outdir: Path, cache_path: Path, model, device,
            ks: list[int], n_images: int | None, batch_size: int):
    print("\n================ E5 STAGE C: IMAGE HISTOGRAMS ================")
    if not (root / "data").is_dir():
        raise FileNotFoundError(f"CIFAR-10 data directory not found: {root / 'data'}")
    z = np.load(cache_path)
    print("cache keys:", z.files)
    required = {"train_indices", "val_indices"}
    missing = required.difference(z.files)
    if missing:
        raise KeyError(f"Missing cache keys: {sorted(missing)}")
    splits = {
        "train": (True, z["train_indices"]),
        "val": (True, z["val_indices"]),
        "test": (False, z["test_indices"] if "test_indices" in z.files else np.arange(10000)),
    }
    cent = {}
    for K in ks:
        centroid_path = outdir / f"centroids_K{K}.npy"
        if not centroid_path.is_file():
            raise FileNotFoundError(f"Centroids not found: {centroid_path}. Run Stage B first.")
        cent[K] = torch.from_numpy(np.load(centroid_path).astype(np.float32)).to(device)
    cnorm = {K: (c ** 2).sum(1) for K, c in cent.items()}
    suffix = "_smoke" if n_images else ""
    tf = image_transform()

    for name, (is_train, idx) in splits.items():
        if n_images is not None:
            idx = idx[:n_images]
        ds = datasets.CIFAR10(str(root / "data"), train=is_train,
                              download=False, transform=tf)
        loader = DataLoader(Subset(ds, idx.tolist()), batch_size=batch_size,
                            shuffle=False, num_workers=0)
        hists = {K: [] for K in ks}
        labels, t0 = [], time.time()
        with torch.no_grad():
            for b, (x, y) in enumerate(loader):
                tok = model.forward_features(x.to(device))["x_norm_patchtokens"]
                tok = torch.nn.functional.normalize(tok, dim=-1)
                B, P, D = tok.shape
                flat = tok.reshape(B * P, D)
                for K in ks:
                    d = cnorm[K][None, :] - 2 * flat @ cent[K].T
                    a = d.argmin(1).reshape(B, P).cpu()
                    h = torch.zeros(B, K).scatter_add_(1, a, torch.ones(B, P))
                    assert (h.sum(1) == P).all()
                    hists[K].append(h.numpy().astype(np.int16))
                labels.append(y.numpy())
                if b % 50 == 0:
                    print(f"{name} batch {b}/{len(loader)}  {time.time()-t0:.0f}s")
        labels = np.concatenate(labels)
        np.save(outdir / f"labels_{name}{suffix}.npy", labels)
        for K in ks:
            H = np.concatenate(hists[K])
            np.save(outdir / f"hist_K{K}_{name}{suffix}.npy", H)
            print(f"saved {name} K={K}", H.shape)

    with open(outdir / f"histograms_meta{suffix}.json", "w", encoding="utf-8") as f:
        json.dump({"stage": "E5-C", "ks": ks, "patch_norm": "L2 per patch",
                   "assignment": "nearest centroid, Euclidean, centroids not renormalized",
                   "histogram": "raw counts (int16), 256 patches per image; normalize in Stage D"},
                  f, indent=2)


def stage_d_e(root: Path, outdir: Path, cache_path: Path, ks: list[int],
              norm: str, cs: list[float]):
    print("\n================ E5 STAGES D+E: CLASSIFICATION ================")
    z = np.load(cache_path)
    required_cache = {"y_train", "y_val", "y_test", "X_train", "X_val", "X_test"}
    missing = required_cache.difference(z.files)
    if missing:
        raise KeyError(f"Missing cache keys required for classification: {sorted(missing)}")
    y = {"train": z["y_train"], "val": z["y_val"], "test": z["y_test"]}

    for s in ["train", "val", "test"]:
        labels_path = outdir / f"labels_{s}.npy"
        if not labels_path.is_file():
            raise FileNotFoundError(f"Stage C labels not found: {labels_path}. Run Stage C first.")
        assert np.array_equal(np.load(labels_path), y[s]), f"label mismatch in {s}"
    print("label alignment with shared cache: OK")

    def normalize(H, kind):
        H = H.astype(np.float32)
        if kind == "l1":
            return H / H.sum(1, keepdims=True)
        return H / np.linalg.norm(H, axis=1, keepdims=True)

    def fit_select_test(Xtr, Xva, Xte, name):
        # This intentionally preserves the source code's train-only fitting.
        best = (-1, None)
        for C in cs:
            t0 = time.time()
            clf = LogisticRegression(C=C, max_iter=2000, random_state=42).fit(Xtr, y["train"])
            acc = accuracy_score(y["val"], clf.predict(Xva))
            print(f"  {name} C={C:<7} val_acc={acc:.4f}  ({time.time()-t0:.0f}s)")
            if acc > best[0]:
                best = (acc, C, clf)
        val_acc, C, clf = best
        pred = clf.predict(Xte)
        res = {"method": name, "C": C, "val_acc": round(val_acc, 4),
               "test_acc": round(accuracy_score(y["test"], pred), 4),
               "test_macro_f1": round(f1_score(y["test"], pred, average="macro"), 4)}
        print(f"  -> {res}")
        return res, clf, pred

    results = []
    for K in ks:
        print(f"\n=== Patch ProtoWords K={K} (norm={norm}) ===")
        H = {}
        for s in ["train", "val", "test"]:
            hist_path = outdir / f"hist_K{K}_{s}.npy"
            if not hist_path.is_file():
                raise FileNotFoundError(f"Histogram not found: {hist_path}. Run Stage C first.")
            H[s] = normalize(np.load(hist_path), norm)
        res, clf, pred = fit_select_test(H["train"], H["val"], H["test"],
                                          f"patch_protowords_K{K}")
        results.append(res)
        joblib.dump(clf, outdir / f"lr_patch_K{K}.joblib")
        np.save(outdir / f"confusion_patch_K{K}.npy", confusion_matrix(y["test"], pred))
        with open(outdir / f"report_patch_K{K}.txt", "w", encoding="utf-8") as f:
            f.write(classification_report(y["test"], pred, digits=4))

    print("\n=== Continuous DINOv2 + LR (E1 reference, recomputed) ===")
    def l2(X):
        X = X.astype(np.float32)
        return X / np.linalg.norm(X, axis=1, keepdims=True)
    res, clf, pred = fit_select_test(l2(z["X_train"]), l2(z["X_val"]),
                                     l2(z["X_test"]), "continuous_dinov2_LR")
    results.append(res)

    print("\n================ COMPARISON ================")
    print(f"{'method':<28}{'C':>8}{'val_acc':>10}{'test_acc':>10}{'macro_F1':>10}")
    for r in results:
        print(f"{r['method']:<28}{r['C']:>8}{r['val_acc']:>10}{r['test_acc']:>10}{r['test_macro_f1']:>10}")
    print("\nAdd your friend's E4 (global codebook) numbers to this table by hand or from their results file.")
    with open(outdir / "e5_results.json", "w", encoding="utf-8") as f:
        json.dump({"norm": norm, "cs": cs, "results": results}, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Run E5 patch ProtoWords experiment (Stages A-D/E).")
    p.add_argument("--project-root", default=None, help="ProtoWords folder; auto-detected by default")
    p.add_argument("--device", choices=["auto", "cuda", "mps", "cpu"], default="auto")
    p.add_argument("--stages", nargs="+", choices=["A", "B", "C", "D"],
                   default=["A", "B", "C", "D"], help="stages to run, in order; default: all")
    p.add_argument("--ks", type=int, nargs="+", default=[64, 128, 256])
    p.add_argument("--n-images", type=int, default=None,
                   help="smoke test for A/C only; do not include D when set")
    p.add_argument("--patches-per-image", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--norm", choices=["l1", "l2"], default="l2")
    p.add_argument("--cs", type=float, nargs="+", default=[1, 10, 100, 1000])
    p.add_argument("--outdir", default="outputs/e5")
    return p.parse_args()


def main():
    args = parse_args()
    if len(set(args.stages)) != len(args.stages):
        raise ValueError("Do not repeat a stage in --stages")
    if any(k <= 0 for k in args.ks):
        raise ValueError("All K values must be positive integers")
    if args.batch_size <= 0 or not 1 <= args.patches_per_image <= 256:
        raise ValueError("batch-size must be positive and patches-per-image must be 1..256")
    if args.n_images is not None and args.n_images <= 0:
        raise ValueError("--n-images must be a positive integer")
    if not args.cs or any(c <= 0 for c in args.cs):
        raise ValueError("--cs must contain positive values")
    if args.n_images is not None and "D" in args.stages:
        raise ValueError("--n-images is for smoke testing A/C; Stage D requires full train/val/test histograms")

    root = project_root_from(__file__, args.project_root)
    cache_path = root / "outputs" / CACHE_NAME
    if not cache_path.is_file():
        raise FileNotFoundError(f"Feature cache not found: {cache_path}")
    outdir = resolve_under_root(root, args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    device = choose_device(args.device)
    print("project root:", root)
    print("output directory:", outdir)
    print("device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(device))
    print("stages:", " -> ".join(args.stages))

    needs_model = any(stage in args.stages for stage in ["A", "C"])
    model = load_dinov2(device) if needs_model else None
    for stage in args.stages:
        if stage == "A":
            stage_a(root, outdir, cache_path, model, device, args.n_images,
                    args.patches_per_image, args.batch_size, args.seed)
        elif stage == "B":
            stage_b(outdir, args.ks, args.seed)
        elif stage == "C":
            stage_c(root, outdir, cache_path, model, device, args.ks,
                    args.n_images, args.batch_size)
        elif stage == "D":
            stage_d_e(root, outdir, cache_path, args.ks, args.norm, args.cs)
    print("\nSelected E5 stages finished.")
    print("Outputs:", outdir)


if __name__ == "__main__":
    main()
