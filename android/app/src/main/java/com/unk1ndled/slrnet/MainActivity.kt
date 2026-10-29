package com.unk1ndled.slrnet

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.util.Log
import android.util.Size
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.unk1ndled.slrnet.databinding.ActivityMainBinding
import java.io.File
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding

    // Single thread for both init and inference: ORT sessions are not cheap to
    // run concurrently with themselves, and serialising keeps memory predictable.
    private val worker = Executors.newSingleThreadExecutor()

    private var engine: DehazeEngine? = null
    private var tiled: TiledDehazer? = null
    private var imageCapture: ImageCapture? = null
    private var liveMode = true

    private val permissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startCamera()
        else {
            Toast.makeText(this, "需要相机权限", Toast.LENGTH_LONG).show()
            finish()
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        binding.btnToggle.setOnClickListener { toggleLive() }
        binding.btnShot.setOnClickListener { captureFull() }

        initEngine()

        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
            permissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun initEngine() {
        worker.execute {
            try {
                val e = DehazeEngine.create(this)
                engine = e
                tiled = TiledDehazer(e)
                runOnUiThread {
                    binding.stats.text = "就绪 · ${e.backend} · 单帧 ${"%.0f".format(e.warmupMs)}ms"
                }
                Log.i(TAG, "engine ready: ${e.backend}, tile=${e.tileSize}")
            } catch (t: Throwable) {
                Log.e(TAG, "engine init failed", t)
                runOnUiThread {
                    binding.stats.text = "模型未加载"
                    Toast.makeText(
                        this,
                        "载入 slrnet.onnx 失败，请确认已放入 app/src/main/assets/\n${t.message}",
                        Toast.LENGTH_LONG
                    ).show()
                }
            }
        }
    }

    private fun startCamera() {
        val future = ProcessCameraProvider.getInstance(this)
        future.addListener({
            val provider = future.get()

            val preview = Preview.Builder().build().also {
                it.setSurfaceProvider(binding.previewView.surfaceProvider)
            }

            val capture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            imageCapture = capture

            val analysis = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                // RGBA output skips the YUV->RGB conversion inside toBitmap()
                .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                // Analyse at ~720x540. The model only ever sees 512x512 after
                // centre-cropping, so decoding 1080p+ every frame just burns
                // memory and triggers GC churn.
                .setResolutionSelector(
                    ResolutionSelector.Builder()
                        .setResolutionStrategy(
                            ResolutionStrategy(
                                Size(720, 540),
                                ResolutionStrategy.FALLBACK_RULE_CLOSEST_HIGHER_THEN_LOWER
                            )
                        )
                        .build()
                )
                .build()

            analysis.setAnalyzer(worker, FrameAnalyzer(PREVIEW_SIDE) { bmp -> onPreviewFrame(bmp) })

            provider.unbindAll()
            provider.bindToLifecycle(
                this, CameraSelector.DEFAULT_BACK_CAMERA, preview, capture, analysis
            )
        }, ContextCompat.getMainExecutor(this))
    }

    private fun onPreviewFrame(square: Bitmap) {
        if (!liveMode) return
        val e = engine ?: return
        val t0 = SystemClock.elapsedRealtime()
        val out = try {
            e.forwardBitmap(square)
        } catch (t: Throwable) {
            Log.w(TAG, "inference failed", t)
            return
        }
        val ms = SystemClock.elapsedRealtime() - t0
        runOnUiThread {
            binding.compare.setBitmaps(square, out)
            binding.compare.alpha = 1f
            binding.stats.text = "${ms}ms · ${e.backend} · ${PREVIEW_SIDE}px"
        }
    }

    private fun toggleLive() {
        liveMode = !liveMode
        if (liveMode) {
            binding.compare.alpha = 1f
            binding.btnToggle.text = "暂停对比"
        } else {
            binding.compare.clear()
            binding.compare.alpha = 0f
            binding.btnToggle.text = "实时对比"
        }
    }

    /** Full-resolution still: tiled inference, quality first, latency second. */
    private fun captureFull() {
        val cap = imageCapture ?: return
        val t = tiled ?: run {
            Toast.makeText(this, "模型尚未就绪", Toast.LENGTH_SHORT).show()
            return
        }

        binding.btnShot.isEnabled = false
        binding.stats.text = "处理中…"

        cap.takePicture(worker, object : ImageCapture.OnImageCapturedCallback() {
            override fun onCaptureSuccess(image: ImageProxy) {
                try {
                    val source = image.toArgbBitmap()
                    val start = SystemClock.elapsedRealtime()
                    val result = t.process(source) { p ->
                        runOnUiThread { binding.stats.text = "分块处理 ${(p * 100).toInt()}%" }
                    }
                    val cost = SystemClock.elapsedRealtime() - start
                    val file = save(result)
                    runOnUiThread {
                        binding.stats.text = "${source.width}x${source.height} · ${cost}ms · 已保存"
                        binding.btnShot.isEnabled = true
                        Toast.makeText(
                            this@MainActivity,
                            "已保存 ${file.name}",
                            Toast.LENGTH_LONG
                        ).show()
                    }
                } catch (t2: Throwable) {
                    Log.e(TAG, "tiled inference failed", t2)
                    runOnUiThread {
                        binding.btnShot.isEnabled = true
                        binding.stats.text = "处理失败"
                        Toast.makeText(this@MainActivity, "${t2.message}", Toast.LENGTH_LONG).show()
                    }
                } finally {
                    image.close()
                }
            }

            override fun onError(exception: ImageCaptureException) {
                Log.e(TAG, "capture failed", exception)
                runOnUiThread {
                    binding.btnShot.isEnabled = true
                    binding.stats.text = "拍摄失败"
                }
            }
        })
    }

    private fun save(bmp: Bitmap): File {
        val dir = getExternalFilesDir("Pictures") ?: filesDir
        if (!dir.exists()) dir.mkdirs()
        val f = File(dir, "dehazed_${System.currentTimeMillis()}.jpg")
        f.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 95, it) }
        return f
    }

    override fun onDestroy() {
        super.onDestroy()
        worker.execute { engine?.close() }
        worker.shutdown()
    }

    companion object {
        private const val TAG = "MainActivity"

        /** Must equal the fixed square size used by tools/export_onnx.py. */
        private const val PREVIEW_SIDE = 512
    }
}
