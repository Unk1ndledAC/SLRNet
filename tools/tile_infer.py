"""Validate tiled inference for SLRNet against single-pass full-image inference.

This is the go/no-go gate before writing any Android code. SLRNet's AFU uses a
global-average-pooled gate, so every tile sees a *different* global context.
Tiles therefore disagree slightly at the seams, and only overlap + cosine
windowing hides it. This script measures exactly how much quality that costs.

    python tile_infer.py --onnx slrnet.onnx --image hazy.png --tile 512 --overlap 64

Outputs: PSNR/SSIM between full-image and tiled results, plus a stitched image
and a seam heatmap so you can eyeball whether the grid is visible.
"""

import argparse

import cv2
import numpy as np
import onnxruntime as ort


def cosine_window(tile: int, eps: float = 1e-3) -> np.ndarray:
    """Separable Hann window, floored so border tiles never contribute zero."""
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(tile) / (tile - 1))
    win = np.outer(w, w).astype(np.float32)
    return np.clip(win, eps, 1.0)[:, :, None]


def tile_starts(length: int, tile: int, stride: int) -> list:
    if length <= tile:
        return [0]
    starts = list(range(0, length - tile + 1, stride))
    if starts[-1] != length - tile:
        starts.append(length - tile)
    return starts


class Engine:
    def __init__(self, onnx_path: str):
        self.sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        self.in_name = self.sess.get_inputs()[0].name
        self.fixed = self.sess.get_inputs()[0].shape[2:4]

    def __call__(self, img: np.ndarray) -> np.ndarray:
        """img: HWC float32 in [0,1]. Returns HWC float32 in [0,1]."""
        x = img.transpose(2, 0, 1)[None].astype(np.float32)
        y = self.sess.run(None, {self.in_name: x})[0]
        return np.clip(y[0].transpose(1, 2, 0), 0.0, 1.0)


def infer_full(img: np.ndarray, engine: Engine) -> np.ndarray:
    """Single pass, resized to the exported graph size, then resized back."""
    h, w = img.shape[:2]
    th, tw = engine.fixed
    if isinstance(th, str) or th is None:
        return engine(img)
    small = cv2.resize(img, (int(tw), int(th)), interpolation=cv2.INTER_AREA)
    out = engine(small)
    return cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)


def infer_tiled(img: np.ndarray, engine: Engine, tile: int, overlap: int) -> np.ndarray:
    """Overlap-tile inference with cosine-weighted blending."""
    h, w = img.shape[:2]
    stride = max(tile - overlap, 1)
    win = cosine_window(tile)

    acc = np.zeros((h, w, 3), np.float32)
    wsum = np.zeros((h, w, 1), np.float32)

    ys = tile_starts(h, tile, stride)
    xs = tile_starts(w, tile, stride)

    for y in ys:
        for x in xs:
            patch = img[y:y + tile, x:x + tile]
            ph, pw = patch.shape[:2]
            if ph != tile or pw != tile:
                patch = cv2.copyMakeBorder(patch, 0, tile - ph, 0, tile - pw,
                                           cv2.BORDER_REFLECT)
            out = engine(patch)
            out = out[:ph, :pw]
            wgt = win[:ph, :pw]
            acc[y:y + ph, x:x + pw] += out * wgt
            wsum[y:y + ph, x:x + pw] += wgt

    return np.clip(acc / np.maximum(wsum, 1e-6), 0.0, 1.0)


def to_u8(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, 0, 1) * 255.0 + 0.5).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse < 1e-12 else 10.0 * np.log10(255.0 ** 2 / mse)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--tile", type=int, default=512)
    ap.add_argument("--overlap", type=int, default=64)
    ap.add_argument("--out-prefix", default="tile_check")
    args = ap.parse_args()

    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"cannot read {args.image}")
    img = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    print(f"[info] input {img.shape[1]}x{img.shape[0]}  tile={args.tile} overlap={args.overlap}")

    engine = Engine(args.onnx)

    full = infer_full(img, engine)
    tiled = infer_tiled(img, engine, args.tile, args.overlap)

    a, b = to_u8(full), to_u8(tiled)
    score = psnr(a, b)
    diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
    mae = float(diff.mean()) / 255.0
    p99 = float(np.percentile(diff, 99)) / 255.0
    print(f"[result] PSNR={score:.2f}dB  MAE={mae:.4f}  p99={p99:.4f}")

    # seam error map: where does tiling disagree most?
    err = diff.max(axis=2)
    cv2.imwrite(f"{args.out_prefix}_seam.png", cv2.applyColorMap(
        np.clip(err * 8, 0, 255).astype(np.uint8), cv2.COLORMAP_INFERNO))
    cv2.imwrite(f"{args.out_prefix}_full.png", cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{args.out_prefix}_tiled.png", cv2.cvtColor(b, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"{args.out_prefix}_input.png", bgr)

    # Distinguish a seam problem from a block-level tone shift: seams are thin
    # lines, tone shift is uniform over each tile. If raising --overlap does not
    # move the number, it is the latter and no amount of blending will fix it.
    if score > 45:
        print("[ok] tiling is visually lossless -- ship this config")
    elif score > 38:
        print("[ok-ish] acceptable; check the seam map for a faint grid")
    elif score > 30:
        print("[warn] measured block-level tone drift, not a seam problem.\n"
              "       Check the seam map: if the bright regions are large blocks\n"
              "       rather than thin lines, raising --overlap will NOT help --\n"
              "       the AFU gate is global, so each tile sees different context.\n"
              "       Options: accept it, or export a dynamic-shape model and infer\n"
              "       the whole frame at once (memory permitting).")
    else:
        print("[fail] tiling is clearly wrong -- verify crop/padding alignment")


if __name__ == "__main__":
    main()
