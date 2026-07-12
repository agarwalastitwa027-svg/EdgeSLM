package com.edgeslm.app

/**
 * Thin Kotlin façade over the native JNI bridge (app/src/main/cpp/llama_jni.cpp).
 * Loading edgeslm_jni pulls in libllama.so / libggml*.so / libOpenCL.so automatically -
 * they all live in the same jniLibs/arm64-v8a directory and are resolved via DT_NEEDED.
 */
object LlamaBridge {

    init {
        System.loadLibrary("edgeslm_jni")
    }

    @Volatile
    var isModelLoaded: Boolean = false
        private set

    fun init(nativeLibDir: String) = nativeInit(nativeLibDir)

    fun systemInfo(): String = nativeSystemInfo()

    fun activeBackend(): String = nativeActiveBackend()

    /** Runs on the calling thread - callers must invoke this off the main thread. */
    fun loadModel(modelPath: String, contextSize: Int = 16384): Boolean {
        unloadModel()
        val result = nativeLoadModel(modelPath, contextSize)
        isModelLoaded = (result == 0)
        return isModelLoaded
    }

    fun unloadModel() {
        if (isModelLoaded) {
            nativeUnloadModel()
            isModelLoaded = false
        }
    }

    fun setSystemPrompt(prompt: String): Boolean = nativeProcessSystemPrompt(prompt) == 0

    /**
     * Feeds the user prompt through the model; call [nextToken] in a loop afterwards.
     * [maxNewTokens] <= 0 means "no artificial cap" - like other chat apps, the reply runs
     * until the model itself decides to stop, bounded only by the context window.
     */
    fun startGeneration(userPrompt: String, maxNewTokens: Int = -1): Boolean =
        nativeProcessUserPrompt(userPrompt, maxNewTokens) == 0

    /** Returns the next decoded text chunk, or null when generation has finished. */
    fun nextToken(): String? = nativeGenerateNextToken()

    fun shutdown() {
        unloadModel()
        nativeShutdown()
    }

    private external fun nativeInit(nativeLibDir: String)
    private external fun nativeSystemInfo(): String
    private external fun nativeActiveBackend(): String
    private external fun nativeLoadModel(modelPath: String, nCtx: Int): Int
    private external fun nativeProcessSystemPrompt(systemPrompt: String): Int
    private external fun nativeProcessUserPrompt(userPrompt: String, nPredict: Int): Int
    private external fun nativeGenerateNextToken(): String?
    private external fun nativeUnloadModel()
    private external fun nativeShutdown()
}
