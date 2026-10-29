把导出的 slrnet.onnx 放在这个目录下（与 README.txt 同级）。

生成方式见项目根 README：
  python tools/export_onnx.py --repo <SLRNet仓库> --ckpt <weight.pth> --out slrnet.onnx --size 512
  copy slrnet.onnx android/app/src/main/assets/slrnet.onnx

没有模型文件时 App 能启动，但会弹出“载入 slrnet.onnx 失败”的提示。
