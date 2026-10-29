package com.unk1ndled.slrnet

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.ImageFormat
import android.graphics.Matrix
import androidx.camera.core.ImageProxy
import java.nio.ByteBuffer

/**
 * Converts a CameraX [ImageProxy] to an upright bitmap.
 *
 * Deliberately hand-rolled instead of using ImageProxy.toBitmap(): that helper
 * is an extension function whose availability differs across CameraX versions,
 * and the still-capture path delivers JPEG while analysis delivers RGBA. Doing
 * it here keeps both paths explicit and release-orientation correct.
 */
fun ImageProxy.toArgbBitmap(): Bitmap {
    val decoded = if (format == ImageFormat.JPEG) decodeJpeg() else decodeRgba()
    val degrees = imageInfo.rotationDegrees
    if (degrees == 0) return decoded
    val m = Matrix().apply { postRotate(degrees.toFloat()) }
    val rotated = Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, m, true)
    // Free the unrotated copy immediately -- on a 3MP capture that is 12MB of
    // peak pressure we do not need while the tiling accumulator is being built.
    if (rotated !== decoded) decoded.recycle()
    return rotated
}

private fun ImageProxy.decodeJpeg(): Bitmap {
    val buf = planes[0].buffer
    val bytes = ByteArray(buf.remaining())
    buf.get(bytes)

    // Decode straight to a bounded size. Without this a 12MP capture allocates
    // a 48MB bitmap before any processing starts, and the float pipeline right
    // after it needs ~3x more -- instant OutOfMemoryError on real devices.
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    BitmapFactory.decodeByteArray(bytes, 0, bytes.size, bounds)
    val longest = maxOf(bounds.outWidth, bounds.outHeight)
    var sample = 1
    while (longest > 0 && longest / sample > MAX_DECODE_SIDE) sample *= 2

    val opts = BitmapFactory.Options().apply { inSampleSize = sample }
    return BitmapFactory.decodeByteArray(bytes, 0, bytes.size, opts)
        ?: error("JPEG decode failed")
}

/** Longest edge of a decoded still, in pixels. Keep in sync with TiledDehazer.MAX_PIXELS. */
private const val MAX_DECODE_SIDE = 2048

private fun ImageProxy.decodeRgba(): Bitmap {
    val plane = planes[0]
    val pixelStride = plane.pixelStride
    val rowStride = plane.rowStride
    val rowBytes = width * pixelStride
    val packed = ByteArray(rowBytes * height)

    val buf = plane.buffer
    for (row in 0 until height) {
        val offset = row * rowStride
        if (offset >= buf.limit()) break
        buf.position(offset)
        val n = minOf(rowBytes, buf.remaining())
        buf.get(packed, row * rowBytes, n)
    }

    val bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
    bmp.copyPixelsFromBuffer(ByteBuffer.wrap(packed))
    return bmp
}
