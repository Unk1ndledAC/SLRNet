package com.unk1ndled.slrnet

import android.graphics.Bitmap
import kotlin.math.PI
import kotlin.math.cos

/**
 * Overlap-tile inference for full-resolution photos.
 *
 * Memory is the hard constraint here, not compute. A 12MP photo needs 144MB for
 * a single float32 RGB copy (3 x 4000 x 3000 x 4B), which alone exceeds the Java
 * heap on most phones -- that is what killed the first release. So this class
 * never materialises the whole image as floats: it pulls one 512x512 tile
 * straight out of the source bitmap, runs it, and blends the result into a
 * full-image accumulator.
 *
 * The accumulator is still float32 (3 x w x h), which is why callers must hand
 * in a bitmap that has already been bounded by MAX_PIXELS.
 */
class TiledDehazer(private val engine: DehazeEngine) {

    fun process(
        source: Bitmap,
        overlap: Int = 64,
        onProgress: ((Float) -> Unit)? = null,
    ): Bitmap {
        val tile = engine.tileSize
        val src = fitWithinBudget(source)
        val w = src.width
        val h = src.height

        // Smaller than one tile: resize up, infer, resize back. Cheaper and
        // simpler than special-casing clamped sampling.
        if (w < tile || h < tile) {
            val up = Bitmap.createScaledBitmap(src, tile, tile, true)
            val res = engine.forwardBitmap(up)
            return Bitmap.createScaledBitmap(res, w, h, true)
        }

        val stride = (tile - overlap).coerceAtLeast(tile / 4)
        val xs = starts(w, tile, stride)
        val ys = starts(h, tile, stride)
        val window = cosineWindow(tile)

        val plane = w * h
        val acc = FloatArray(3 * plane)
        val wsum = FloatArray(plane)
        val tilePx = IntArray(tile * tile)
        val total = xs.size.toFloat() * ys.size
        var done = 0

        for (y0 in ys) {
            for (x0 in xs) {
                // starts() guarantees x0+tile <= w and y0+tile <= h, so this
                // read is always in bounds -- no clamped sampling needed.
                src.getPixels(tilePx, 0, tile, x0, y0, tile, tile)
                val patch = argbToPlanar(tilePx, tile)
                val res = engine.forward(patch, tile, tile)
                blend(acc, wsum, res, window, x0, y0, tile, w, plane)
                done++
                onProgress?.invoke(done / total)
            }
        }

        return planarToBitmap(acc, wsum, w, h)
    }

    /** Last-resort guard: keeps the float accumulator under ~50MB. */
    private fun fitWithinBudget(source: Bitmap): Bitmap {
        val pixels = source.width.toLong() * source.height
        if (pixels <= MAX_PIXELS) return source
        val scale = Math.sqrt(MAX_PIXELS.toDouble() / pixels)
        val nw = (source.width * scale).toInt().coerceAtLeast(engine.tileSize)
        val nh = (source.height * scale).toInt().coerceAtLeast(engine.tileSize)
        return Bitmap.createScaledBitmap(source, nw, nh, true)
    }

    private fun argbToPlanar(px: IntArray, tile: Int): FloatArray {
        val n = tile * tile
        val out = FloatArray(3 * n)
        val k = 1f / 255f
        for (i in 0 until n) {
            val p = px[i]
            out[i] = ((p shr 16) and 0xFF) * k
            out[n + i] = ((p shr 8) and 0xFF) * k
            out[2 * n + i] = (p and 0xFF) * k
        }
        return out
    }

    private fun blend(
        acc: FloatArray, wsum: FloatArray, res: FloatArray, win: FloatArray,
        x0: Int, y0: Int, tile: Int, w: Int, plane: Int,
    ) {
        val tilePlane = tile * tile
        for (yy in 0 until tile) {
            val rowBase = (y0 + yy) * w + x0
            val srcRow = yy * tile
            for (xx in 0 until tile) {
                val si = srcRow + xx
                val wt = win[si]
                val gi = rowBase + xx
                acc[gi] += res[si] * wt
                acc[plane + gi] += res[tilePlane + si] * wt
                acc[2 * plane + gi] += res[2 * tilePlane + si] * wt
                wsum[gi] += wt
            }
        }
    }

    private fun planarToBitmap(acc: FloatArray, wsum: FloatArray, w: Int, h: Int): Bitmap {
        val plane = w * h
        val px = IntArray(plane)
        for (i in 0 until plane) {
            val inv = 1f / wsum[i].coerceAtLeast(1e-6f)
            val r = (acc[i] * inv * 255f + 0.5f).toInt().coerceIn(0, 255)
            val g = (acc[plane + i] * inv * 255f + 0.5f).toInt().coerceIn(0, 255)
            val b = (acc[2 * plane + i] * inv * 255f + 0.5f).toInt().coerceIn(0, 255)
            px[i] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
        }
        return Bitmap.createBitmap(px, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun starts(length: Int, tile: Int, stride: Int): List<Int> {
        if (length <= tile) return listOf(0)
        val list = ArrayList<Int>()
        var s = 0
        while (s + tile < length) {
            list.add(s)
            s += stride
        }
        list.add(length - tile)
        return list
    }

    private fun cosineWindow(tile: Int): FloatArray {
        val w1 = FloatArray(tile) { i ->
            (0.5 - 0.5 * cos(2.0 * PI * i / (tile - 1))).toFloat()
        }
        val win = FloatArray(tile * tile)
        for (y in 0 until tile) {
            val wy = w1[y]
            val row = y * tile
            for (x in 0 until tile) {
                win[row + x] = (wy * w1[x]).coerceAtLeast(1e-3f)
            }
        }
        return win
    }

    companion object {
        /**
         * ~3.5MP, which is what a 2048px-longest-edge decode lands on, so the
         * guard does not trigger a redundant rescale for typical captures.
         * Accumulator is 3 x MAX_PIXELS x 4B = 42MB, plus 14MB for the weights.
         */
        const val MAX_PIXELS = 3_500_000L
    }
}
