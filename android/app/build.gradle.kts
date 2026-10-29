plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.unk1ndled.slrnet"
    compileSdk = 36
    // 本机只装了 build-tools 36.x；AGP 8.5.2 默认想要 34.0.0，而 dl.google.com 在本机不可达
    buildToolsVersion = "36.0.0"

    defaultConfig {
        applicationId = "com.unk1ndled.slrnet"
        minSdk = 26          // NNAPI 1.0 needs API 27 for the common path, 26 keeps XNNPACK fallback
        targetSdk = 34
        versionCode = 1
        versionName = "0.1"

        // ONNX Runtime ships ~15MB of native libs per ABI. Every phone from the
        // last several years is arm64, so shipping four ABIs quadruples the APK
        // for nothing. Add "x86_64" back if you need an emulator.
        ndk {
            abiFilters += listOf("arm64-v8a")
        }
    }

    buildFeatures {
        viewBinding = true
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
        }
    }

    // Critical: without this the .onnx lands in the APK compressed and ORT
    // can no longer mmap it, forcing a full copy into RAM at startup.
    androidResources {
        noCompress += "onnx"
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("com.google.android.material:material:1.12.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")

    val camerax = "1.4.0"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.19.2")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.4")
}
