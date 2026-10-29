package com.unk1ndled.slrnet

import android.graphics.Bitmap
import android.graphics.Matrix
import android.os.SystemClock
import android.util.Log
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import kotlin.math.min

/**
 * Pulls frames from CameraX, crops to a square and throttles.
 *
 * Throttling matters: one 512x512 pass costs 100-400ms on a phone GPU, so
 * analysing every frame would just build a queue. CameraX with
 * STRATEGY_KEEP_ONLY_LATEST drops the backlog for us; this interval keeps CPU
 * headroom for the preview surface.
 */
class FrameAnalyzer(
    private val size: Int,
    private val minIntervalMs: Long = 150L,
    private val onFrame: (Bitmap) -> Unit,
) : ImageAnalysis.Analyzer {

    private var lastProcessed = 0L

    override fun analyze(image: ImageProxy) {
        val now = SystemClock.elapsedRealtime()
        if (now - lastProcessed < minIntervalMs) {
            image.close()
            return
        }
        lastProcessed = now
        try {
            onFrame(centerSquare(image.toArgbBitmap(), size))
        } catch (t: Throwable) {
            Log.w(TAG, "frame skipped: ${t.message}")
        } finally {
            image.close()
        }
    }

    companion object {
        private const val TAG = "FrameAnalyzer"

        /** Center-crops to a square and scales to [size], matching the model's fixed input. */
        fun centerSquare(src: Bitmap, size: Int): Bitmap {
            val s = min(src.width, src.height)
            val x = (src.width - s) / 2
            val y = (src.height - s) / 2
            val scale = size.toFloat() / s
            val m = Matrix().apply { setScale(scale, scale) }
            return Bitmap.createBitmap(src, x, y, s, s, m, true)
        }
    }
}
