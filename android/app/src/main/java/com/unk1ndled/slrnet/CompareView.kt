package com.unk1ndled.slrnet

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.util.AttributeSet
import android.view.MotionEvent
import android.view.View
import kotlin.math.max

/**
 * Before/after wipe comparison. Left of the divider shows the original frame,
 * right shows the dehazed output. Drag anywhere to move the divider.
 */
class CompareView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
) : View(context, attrs) {

    private var original: Bitmap? = null
    private var processed: Bitmap? = null
    private var split = 0.5f

    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val linePaint = Paint().apply {
        color = Color.WHITE
        strokeWidth = 3f
        isAntiAlias = true
    }
    private val labelPaint = Paint().apply {
        color = Color.WHITE
        textSize = 32f
        isAntiAlias = true
        setShadowLayer(6f, 0f, 0f, Color.BLACK)
    }
    private val srcRect = Rect()

    fun setBitmaps(before: Bitmap, after: Bitmap) {
        original = before
        processed = after
        invalidate()
    }

    fun clear() {
        original = null
        processed = null
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        val a = original ?: return
        val b = processed ?: return
        if (width == 0 || height == 0) return

        coverSource(a, srcRect)
        val dst = Rect(0, 0, width, height)
        val divider = (width * split).toInt()

        canvas.save()
        canvas.clipRect(0, 0, divider, height)
        canvas.drawBitmap(a, srcRect, dst, paint)
        canvas.restore()

        canvas.save()
        canvas.clipRect(divider, 0, width, height)
        coverSource(b, srcRect)
        canvas.drawBitmap(b, srcRect, dst, paint)
        canvas.restore()

        canvas.drawLine(divider.toFloat(), 0f, divider.toFloat(), height.toFloat(), linePaint)
        canvas.drawText("原始", 28f, height - 40f, labelPaint)
        canvas.drawText("去雾", divider + 28f, height - 40f, labelPaint)
    }

    /** Source rect that center-crops [bmp] to fill this view without distortion. */
    private fun coverSource(bmp: Bitmap, out: Rect) {
        val vw = width.toFloat()
        val vh = height.toFloat()
        val scale = max(vw / bmp.width, vh / bmp.height)
        val cw = vw / scale
        val ch = vh / scale
        val left = (bmp.width - cw) / 2f
        val top = (bmp.height - ch) / 2f
        out.set(
            left.toInt().coerceAtLeast(0),
            top.toInt().coerceAtLeast(0),
            (left + cw).toInt().coerceIn(1, bmp.width),
            (top + ch).toInt().coerceIn(1, bmp.height),
        )
    }

    override fun onTouchEvent(event: MotionEvent): Boolean {
        when (event.actionMasked) {
            MotionEvent.ACTION_DOWN, MotionEvent.ACTION_MOVE -> {
                split = (event.x / width.coerceAtLeast(1)).coerceIn(0f, 1f)
                invalidate()
                return true
            }
        }
        return super.onTouchEvent(event)
    }
}
