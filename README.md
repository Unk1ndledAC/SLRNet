# SLRNet for Android — On-Device Real-Time Image Dehazing

A native Android implementation of [SLRNet](https://github.com/Unk1ndledAC/SLRNet) (ICIC 2026, 96.5K parameters), running fully on-device. No network, no server, no photo ever leaves the phone.

## Why this stack

For Android-only, cross-platform frameworks are all cost and no benefit:

| Decision | Choice | Rationale |
|---|---|---|
| Language / UI | Kotlin + ViewBinding | No plugin layer; CameraX and ONNX Runtime are first-class citizens |
| Inference runtime | **ONNX Runtime 1.19** | `torch.onnx.export` preserves GELU's `erf` exactly; `onnx2tf` degrades it in some versions, which shows up as a washed-out output |
| Acceleration backend | NNAPI with CPU fallback | Both backends are benchmarked at startup; the faster one wins — no assumptions |
| Camera | CameraX 1.4 | `STRATEGY_KEEP_ONLY_LATEST` drops frames instead of queueing them |
| Frame format | `ImageAnalysis` in `RGBA_8888` | Skips the YUV→RGB conversion |

## Performance reality (read before setting expectations)

The model is tiny, but dehazing is a dense per-pixel computation:

- ~**96.9K MAC per pixel** (3→32 projection 864 + AFU×2 ≈ 3000 + 5×ResBlock 92160 + output 864)
- 512×512 = 262K pixels → **≈ 51 GFLOP per frame**
- Mobile GPUs deliver ~300–500 effective GFLOPS → **100–400 ms per frame**

So the product shape is:

| Mode | Resolution | Latency | Implementation |
|---|---|---|---|
| Live compare | 512×512 (center crop) | 100–400 ms | `FrameAnalyzer`, throttled to 150 ms |
| Photo capture | long edge 2048, 512 tiles, overlap 64 | 3–8 s (~20 tiles) | `TiledDehazer` with cosine-window blending |

**Don't expect 30 fps live preview.** The realistic outcome on a phone is a 3–6 fps refresh — perfectly fine for a capture-oriented app.

## Repository layout

```
├── README.md
├── .gitignore
├── tools/
│   ├── export_onnx.py        # .pth → .onnx, with PyTorch/ORT numeric verification
│   └── tile_infer.py         # tiled vs. full-image PSNR comparison (pre-flight gate)
└── android/                  # the app; open this folder in Android Studio
    └── app/src/main/
        ├── assets/           # slrnet.onnx lives here
        ├── java/com/unk1ndled/slrnet/
        │   ├── MainActivity.kt    # camera wiring, dual-path scheduling
        │   ├── DehazeEngine.kt    # ORT session, backend benchmarking, pixel↔tensor
        │   ├── Tiling.kt          # overlapping tiles + cosine-window blending
        │   ├── FrameAnalyzer.kt   # frame intake, square crop, throttling
        │   ├── ImageProxyExt.kt   # JPEG / RGBA decode paths
        │   └── CompareView.kt     # draggable before/after slider
        └── res/
```

## Getting started

### 1. Export the model

In an environment with PyTorch:

```bash
python tools/export_onnx.py \
    --repo /path/to/SLRNet \
    --ckpt /path/to/weight.pth \
    --out slrnet.onnx --size 512
```

Continue only if you see `[ok] numerics match`. If `max|PyTorch−ONNX| > 1e-4`, the Clip/GELU fusion went wrong — check the opset (default 17).

A pre-exported `slrnet.onnx` is already included in `android/app/src/main/assets/`, so this step is optional for a quick start.

### 2. Verify the cost of tiling (don't skip)

```bash
python tools/tile_infer.py --onnx slrnet.onnx --image hazy.png --tile 512 --overlap 64
```

SLRNet's AFU uses global average pooling for gating, so every tile sees a different "global" context and tiled output differs from full-image inference. The script prints PSNR/MAE and writes an error heatmap `*_seam.png`.

**Judge by the shape of the bright regions, not just the number.** Thin grid lines → raising overlap helps. Large blocky regions → block-level color shift; overlap won't fix it (see below).

### 3. Build

Open `android/` in Android Studio, or:

```bash
cd android && ./gradlew assembleDebug
```

`minSdk 26`, `targetSdk 34`, `compileSdk 36`, AGP 8.5.2 + Gradle 8.7. AGP 8 requires JDK 17; if Gradle picks up an older JDK, set `JAVA_HOME` or `org.gradle.java.home` in `android/gradle.properties`.

> **Network-restricted environments:** if `services.gradle.org` or `maven.google.com` are unreachable, point the Gradle wrapper's `distributionUrl` at a mirror (e.g. `https://mirrors.cloud.tencent.com/gradle/gradle-8.7-bin.zip`) and add `maven.aliyun.com` repositories to `settings.gradle.kts` — the layout of both files makes this a two-line change.

### 4. Install and run

```bash
bash android/run_on_phone.sh   # detects device, installs, grants camera, launches, tails logcat
adb logcat -s DehazeEngine     # or check the backend choice
```

Expected log:

```
backend=nnapi tile=512px warmup=183.4ms
backend=cpu tile=512px warmup=612.0ms
selected backend=nnapi (183.4ms/pass)
```

If NNAPI is slower than CPU, graph-partitioning fell back to per-node CPU execution — a common quirk on some SoCs. The code picks the faster backend automatically, so nothing breaks.

## Tiled-inference findings (important)

Measured on a 1600×1200 synthetic hazy image, fixed 512 input, baseline = dynamic-shape full-image inference with the same weights:

| overlap | PSNR (tiled vs. full) |
|---|---|
| 64 | 31.35 dB |
| 128 | 32.49 dB |
| 192 | 32.41 dB |
| 256 | 32.45 dB |

**PSNR saturates after ~1 dB** — the error heatmap shows blocky regions, not grid lines.

Root cause: AFU's channel gate comes from GAP, so each tile's "global context" is really the tile itself, giving each tile a slightly different tone mapping. **This is block-level drift, not a boundary artifact — windowed blending cannot remove it.**

Options:

1. Accept it — the composite shows no visible grid, and the app ships this path.
2. Full-image dynamic-shape inference: **infeasible**, 32 channels × 1600×1200 × 4B ≈ 245 MB per feature map, and AFU keeps early/late features simultaneously.
3. Retrain with local-window pooling in AFU — the real fix, but it changes the published model definition.

Counter-intuitive fact: **tiled inference does far more compute than full-image inference.** At overlap 256 every pixel is computed ~4×. Tiling exists to bound memory, not to save compute.

## Known pitfalls

1. **`noCompress += "onnx"` must stay.** Otherwise the model is compressed inside the APK, ORT cannot mmap it, and startup loads it fully into memory.
2. **Fixed 512 input.** NNAPI rebuilds the graph for dynamic shapes, an order of magnitude slower. `PREVIEW_SIDE` must match the exported size.
3. **Preview and CompareView must crop identically** (both center-crop to fill), or the slider sides misalign.
4. **Capture is JPEG, preview is RGBA.** `ImageProxyExt.kt` branches on `format` — don't mix them.
5. **On low-end devices, lower `PREVIEW_SIDE` (512→384)** rather than raising `minIntervalMs`. Latency scales with pixel count.
6. **Capture-path memory is a hard constraint.** A 4000×3000 photo as float32 RGB is 144 MB — above most devices' Java heap, and it will throw. Therefore:
   - `decodeJpeg` uses `inSampleSize` to cap the long edge at **2048** (small at decode time, not scaled afterwards)
   - `TiledDehazer` never converts the whole image to float — it reads pixels tile by tile
   - `TiledDehazer.MAX_PIXELS = 3.5M` acts as a final clamp
   - the manifest enables `largeHeap`

   Peak usage is ~95 MB. Raising `MAX_DECODE_SIDE` / `MAX_PIXELS` trades OOM risk for resolution directly.

## Roadmap

- **INT8 quantization**: quantize convolutions only, keep AFU's sigmoid branch in fp16 (gating saturation loses detail). Expected 1.5–2× speedup at 0.1–0.3 dB.
- **Video pipeline**: `MediaCodec` → `SurfaceTexture` → GL sampling → inference → encode; 10–15 fps at 480p is plausible.
- **NPU backends**: QNN EP for Snapdragon 8 Gen series, NeuroPilot for MediaTek.
- **A/B export**: composite the original and dehazed frames into one side-by-side image for paper figures.

## Citation

```bibtex
@inproceedings{qu2026slrnet,
  title={SLRNet: Super Lightweight Residual Network for Real-Time Image Dehazing},
  author={Qu, Guanheng and Jiang, Fan and Liu, Jiangming},
  booktitle={Proceedings of International Conference on Intelligent Computing (ICIC)},
  month={July},
  year={2026},
  address={Toronto, Canada},
  pages={218--231},
  note={Poster Volume Ⅰ},
  doi = {10.65286/icic.v22i1.71074},
}
```
