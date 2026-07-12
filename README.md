# EdgeSLM

An Android app that runs small language models (SLMs) fully on-device, offloaded to the
Qualcomm Adreno GPU via llama.cpp's OpenCL backend, through a JNI bridge.

## What it does

- Runs GGUF-format models locally with GPU acceleration (llama.cpp OpenCL backend +
  Adreno-optimized kernels: `GGML_OPENCL_USE_ADRENO_KERNELS`).
- **Never downloads a model on its own.** You bring your own `.gguf` file - via the in-app
  file picker, or by dropping it in your device's Downloads folder, where it's auto-detected.
- Once imported, models are auto-detected on every subsequent launch - no re-picking needed.
- Streaming chat UI built with Jetpack Compose, with a "Think" toggle for hybrid-reasoning
  models (e.g. Qwen3) that support a `/no_think` directive to skip extended reasoning.

## Architecture

- **UI**: Kotlin + Jetpack Compose (`app/src/main/java/com/edgeslm/app/`)
  - `MainActivity.kt` - screens (model picker, loading, chat) and chat UI
  - `LlamaBridge.kt` - Kotlin facade over the native JNI bridge
  - `ModelManager.kt` - local/Downloads model detection, SAF import
- **Native**: C++ JNI bridge over llama.cpp (`app/src/main/cpp/`)
  - `llama_jni.cpp` - model load, chat templating, tokenization, batched decode, streaming
    generation, sampling (repetition penalty + DRY + reasoning-budget cap)
  - `CMakeLists.txt` - builds llama.cpp/ggml statically into a single `libedgeslm_jni.so`,
    with the OpenCL/Adreno backend enabled
  - `llama.cpp/` - vendored llama.cpp source (GPU inference engine)
  - `opencl/` - vendored OpenCL headers + a built ICD loader `libOpenCL.so` for Android,
    used to link against and to dlopen the real vendor driver at runtime

## Build

Requires Android Studio / NDK r27+, CMake 3.22+, and a JDK 17 toolchain (Gradle doesn't
support newer JDKs yet). Native libs target `arm64-v8a` only (all real Snapdragon phones).

```
./gradlew assembleDebug
```

Output: `app/build/outputs/apk/debug/app-debug.apk`

## Notes

- `n_gpu_layers = -1` offloads every model layer to the GPU by default.
- Context window and generation limits are configured in `llama_jni.cpp` /
  `LlamaBridge.kt` - see comments there for the current defaults.
- This is a personal/experimental project, not a production release.
