#!/usr/bin/env bash
# 手机插上 USB 后运行本脚本：安装并启动 App，然后跟踪推理日志。
# 用法：bash run_on_phone.sh
set -uo pipefail

SDK="${ANDROID_HOME:-$LOCALAPPDATA/Android/Sdk}"
ADB="$SDK/platform-tools/adb.exe"
APK="$(cd "$(dirname "$0")" && pwd)/app/build/outputs/apk/debug/app-debug.apk"
PKG="com.unk1ndled.slrnet"

if [ ! -f "$APK" ]; then
    echo "找不到 APK：$APK"
    echo "先编译：cd $(dirname "$0") && ../../vendor/gradle-8.7/bin/gradle assembleDebug"
    exit 1
fi

if ! "$ADB" get-state >/dev/null 2>&1; then
    echo "未检测到设备。请依次确认："
    echo "  1. 手机『设置 → 关于手机』连点『版本号』7 次，开启开发者选项"
    echo "  2. 『开发者选项』里打开『USB 调试』"
    echo "  3. 数据线连电脑，手机弹出『允许 USB 调试吗』点『允许』"
    echo "  4. 若插上只充电不识别，下拉通知栏把 USB 用途改成『传输文件/MTP』"
    exit 1
fi

echo "== 设备 =="
"$ADB" devices -l

echo "== 安装（覆盖安装，保留数据）=="
"$ADB" install -r -d "$APK" || exit 1

echo "== 授权相机权限（免去手点）=="
"$ADB" shell pm grant "$PKG" android.permission.CAMERA 2>/dev/null || true

echo "== 启动 =="
"$ADB" shell am start -n "$PKG/.MainActivity" || exit 1

echo "== 清空日志并跟踪（Ctrl+C 退出）=="
sleep 2
"$ADB" logcat -c
"$ADB" logcat -s DehazeEngine:I FrameAnalyzer:W MainActivity:I AndroidRuntime:E
