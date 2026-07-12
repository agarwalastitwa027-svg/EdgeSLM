plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
}

android {
    namespace = "com.edgeslm.app"
    compileSdk = 35
    ndkVersion = "27.2.12479018"

    defaultConfig {
        applicationId = "com.edgeslm.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "1.0"

        ndk {
            // Snapdragon phones are arm64 only - skip other ABIs to cut build time and APK size.
            abiFilters += "arm64-v8a"
        }

        externalNativeBuild {
            cmake {
                arguments += listOf(
                    "-DANDROID_STL=c++_shared",
                    "-DANDROID_LD=lld",
                    "-DGGML_OPENCL=ON",
                    "-DGGML_OPENCL_USE_ADRENO_KERNELS=ON",
                    "-DGGML_OPENCL_EMBED_KERNELS=ON",
                    "-DGGML_LLAMAFILE=ON",
                    "-DGGML_NATIVE=OFF",
                    "-DLLAMA_CURL=OFF",
                    "-DLLAMA_BUILD_TESTS=OFF",
                    "-DLLAMA_BUILD_EXAMPLES=OFF",
                    "-DLLAMA_BUILD_SERVER=OFF",
                    "-DLLAMA_BUILD_TOOLS=OFF",
                    "-DCMAKE_BUILD_TYPE=Release"
                )
                cppFlags += listOf("-O3", "-fvisibility=hidden")
                abiFilters += "arm64-v8a"
            }
        }
    }

    externalNativeBuild {
        cmake {
            path = file("src/main/cpp/CMakeLists.txt")
            version = "3.22.1"
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            externalNativeBuild {
                cmake {
                    arguments += "-DCMAKE_BUILD_TYPE=Release"
                }
            }
        }
        debug {
            // Keep native optimizations on in debug too, otherwise on-device inference is unusably slow.
            externalNativeBuild {
                cmake {
                    arguments += "-DCMAKE_BUILD_TYPE=RelWithDebInfo"
                }
            }
        }
    }

    packaging {
        jniLibs {
            useLegacyPackaging = false
            pickFirsts += listOf("**/libc++_shared.so", "**/libOpenCL.so")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
        freeCompilerArgs += listOf("-opt-in=androidx.compose.material3.ExperimentalMaterial3Api")
    }

    buildFeatures {
        compose = true
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.8.7")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation(platform("androidx.compose:compose-bom:2024.10.01"))
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.ui:ui-graphics")
    implementation("androidx.compose.ui:ui-tooling-preview")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.documentfile:documentfile:1.0.1")
}
