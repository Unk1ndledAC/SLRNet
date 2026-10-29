"""Export SLRNet (.pth) to ONNX for Android deployment, then verify numerics.

Usage (run inside the SLRNet repo so `models.slrnet` is importable):

    python export_onnx.py --repo /path/to/SLRNet --ckpt /path/to/weight.pth \
                          --out ./slrnet.onnx --fp16

Why this script exists:
  * SLRNet is fully convolutional, so a dynamic H/W axis is correct, but
    mobile EPs (NNAPI / GPU delegate) rebuild their graph on every new shape.
    We therefore export a fixed 512x512 graph as the primary artifact.
  * PyTorch -> ONNX -> TensorFlow -> TFLite loses the GELU `erf` op on some
    versions of onnx2tf. Going straight to ONNX keeps the graph exact, which
    is why the Android app consumes .onnx via ONNX Runtime instead.
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn


def load_model(repo: str, ckpt_path: str) -> nn.Module:
    sys.path.insert(0, os.path.abspath(repo))
    from models.slrnet import SLRNet  # noqa: E402

    model = SLRNet()
    ckpt = torch.load(ckpt_path, map_location="cpu")
    if isinstance(ckpt, dict):
        for key in ("state_dict", "model", "model_state_dict", "params"):
            if key in ckpt and isinstance(ckpt[key], dict):
                ckpt = ckpt[key]
                break
    state = {k.replace("module.", "", 1): v for k, v in ckpt.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected[:5]}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[info] parameters: {n_params / 1e3:.1f}K  ({n_params * 4 / 1e6:.2f} MB fp32)")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="path to the SLRNet repository root")
    ap.add_argument("--ckpt", required=True, help="path to weight.pth")
    ap.add_argument("--out", default="slrnet.onnx")
    ap.add_argument("--size", type=int, default=512, help="fixed square export size")
    ap.add_argument("--dynamic", action="store_true", help="also export a dynamic-shape variant")
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    model = load_model(args.repo, args.ckpt)

    dummy = torch.rand(1, 3, args.size, args.size)
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {"input": {2: "h", 3: "w"}, "output": {2: "h", 3: "w"}}

    with torch.no_grad():
        torch.onnx.export(
            model,
            dummy,
            args.out,
            input_names=["input"],
            output_names=["output"],
            opset_version=args.opset,
            do_constant_folding=True,
            dynamic_axes=dynamic_axes,
        )
    print(f"[ok] exported -> {args.out}")

    # ---- numerical parity check: PyTorch vs ONNX Runtime ----
    import onnxruntime as ort

    sess = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {in_name: dummy.numpy()})[0]
    diff = np.abs(ref - got)
    print(f"[check] max|dPyTorch-ONNX| = {diff.max():.3e}  mean = {diff.mean():.3e}")
    if diff.max() > 1e-4:
        print("[warn] numeric drift above 1e-4 -- inspect Clip/GELU fusion")
    else:
        print("[ok] numerics match")

    # ---- shape introspection for the Android side ----
    for i in sess.get_inputs():
        print(f"[io] input  {i.name}: {i.shape} {i.type}")
    for o in sess.get_outputs():
        print(f"[io] output {o.name}: {o.shape} {o.type}")

    print("\nNext: copy the .onnx into "
          "android/app/src/main/assets/slrnet.onnx, then run tile_infer.py "
          "to confirm tiling loss is acceptable before trusting on-device output.")


if __name__ == "__main__":
    main()
