package com.unk1ndled.slrnet

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import ai.onnxruntime.TensorInfo
import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import java.io.File
import java.nio.FloatBuffer

/**
 * SLRNet inference wrapper.
 *
 * Backend selection is measured, not assumed: NNAPI sometimes silently falls
 * back to CPU per-node (graph partitioning), so we time a warmup on both
 * candidates and keep whichever is actually faster on this device.
 */
class DehazeEngine private constructor(
    private val env: OrtEnvironment,
    private val session: OrtSession,
    private val inputName: String,
    val backend: String,
    val tileSize: Int,
    val warmupMs: Double,
) : AutoCloseable {

    /**
     * Runs one forward pass. [rgb] is planar NCHW-ish layout produced by
     * [bitmapToRgb]: plane 0 = R, plane 1 = G, plane 2 = B, all in [0,1].
     * Returns the same layout.
     */
    fun forward(rgb: FloatArray, h: Int, w: Int): FloatArray {
        val tensor = OnnxTensor.createTensor(
            env, FloatBuffer.wrap(rgb), longArrayOf(1, 3, h.toLong(), w.toLong())
        )
        tensor.use { t ->
            session.run(mapOf(inputName to t)).use { result ->
                val out = result.get(0) as OnnxTensor
                val fb: FloatBuffer = out.floatBuffer
                val arr = FloatArray(fb.remaining())
                fb.get(arr)
                return arr
            }
        }
    }

    /** Convenience path for a bitmap that is already exactly tileSize square. */
    fun forwardBitmap(bmp: Bitmap): Bitmap {
        val rgb = bitmapToRgb(bmp)
        val out = forward(rgb, bmp.height, bmp.width)
        return rgbToBitmap(out, bmp.width, bmp.height)
    }

    override fun close() {
        session.close()
    }

    companion object {
        private const val TAG = "DehazeEngine"
        const val MODEL_ASSET = "slrnet.onnx"

        fun create(context: Context, assetName: String = MODEL_ASSET): DehazeEngine {
            val env = OrtEnvironment.getEnvironment()
            val modelFile = copyAsset(context, assetName)

            var best: OrtSession? = null
            var bestBackend = "cpu"
            var bestMs = Double.MAX_VALUE
            var bestTile = 512

            for (backend in listOf("nnapi", "cpu")) {
                val s = try {
                    buildSession(env, modelFile, backend)
                } catch (t: Throwable) {
                    Log.w(TAG, "backend $backend unavailable: ${t.message}")
                    continue
                }
                val name = s.inputNames.first()
                val tile = detectTile(s, name, 512)
                val ms = timeWarmup(env, s, name, tile)
                Log.i(TAG, "backend=$backend tile=${tile}px warmup=${"%.1f".format(ms)}ms")
                if (ms < bestMs) {
                    best?.close()
                    best = s
                    bestBackend = backend
                    bestMs = ms
                    bestTile = tile
                } else {
                    s.close()
                }
            }

            val chosen = best ?: throw IllegalStateException(
                "no usable ORT backend -- is slrnet.onnx valid?"
            )
            Log.i(TAG, "selected backend=$bestBackend (${"%.1f".format(bestMs)}ms/pass)")
            return DehazeEngine(env, chosen, chosen.inputNames.first(), bestBackend, bestTile, bestMs)
        }

        private fun buildSession(env: OrtEnvironment, file: File, backend: String): OrtSession {
            val opts = OrtSession.SessionOptions()
            opts.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            if (backend == "nnapi") {
                opts.addNnapi()
                opts.setIntraOpNumThreads(1)
            } else {
                opts.setIntraOpNumThreads(Runtime.getRuntime().availableProcessors().coerceIn(1, 4))
            }
            return env.createSession(file.absolutePath, opts)
        }

        private fun detectTile(session: OrtSession, name: String, fallback: Int): Int = try {
            val info = session.inputInfo[name]?.info as? TensorInfo
            val shape = info?.shape
            if (shape != null && shape.size == 4 && shape[2] > 0 && shape[3] > 0) {
                minOf(shape[2], shape[3]).toInt()
            } else {
                fallback
            }
        } catch (t: Throwable) {
            fallback
        }

        private fun timeWarmup(
            env: OrtEnvironment, session: OrtSession, name: String, tile: Int
        ): Double {
            val dummy = FloatArray(3 * tile * tile) { 0.5f }
            // first pass allocates arenas, second pass is the number we care about
            repeat(2) {
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(dummy), longArrayOf(1, 3, tile.toLong(), tile.toLong())
                ).use { t -> session.run(mapOf(name to t)).close() }
            }
            val t0 = System.nanoTime()
            repeat(3) {
                OnnxTensor.createTensor(
                    env, FloatBuffer.wrap(dummy), longArrayOf(1, 3, tile.toLong(), tile.toLong())
                ).use { t -> session.run(mapOf(name to t)).close() }
            }
            return (System.nanoTime() - t0) / 3.0 / 1_000_000.0
        }

        private fun copyAsset(context: Context, name: String): File {
            val dest = File(context.filesDir, name)
            context.assets.open(name).use { input ->
                if (!dest.exists() || dest.length() != input.available().toLong()) {
                    dest.outputStream().use { input.copyTo(it) }
                }
            }
            return dest
        }

        // ---------- pixel <-> planar float ----------

        fun bitmapToRgb(bmp: Bitmap): FloatArray {
            val w = bmp.width
            val h = bmp.height
            val px = IntArray(w * h)
            bmp.getPixels(px, 0, w, 0, 0, w, h)
            val plane = w * h
            val out = FloatArray(3 * plane)
            val k = 1f / 255f
            for (i in px.indices) {
                val p = px[i]
                out[i] = ((p shr 16) and 0xFF) * k
                out[plane + i] = ((p shr 8) and 0xFF) * k
                out[2 * plane + i] = (p and 0xFF) * k
            }
            return out
        }

        fun rgbToBitmap(rgb: FloatArray, w: Int, h: Int): Bitmap {
            val plane = w * h
            val px = IntArray(plane)
            for (i in 0 until plane) {
                val r = (rgb[i] * 255f + 0.5f).toInt().coerceIn(0, 255)
                val g = (rgb[plane + i] * 255f + 0.5f).toInt().coerceIn(0, 255)
                val b = (rgb[2 * plane + i] * 255f + 0.5f).toInt().coerceIn(0, 255)
                px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
            return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
        }
    }
}
