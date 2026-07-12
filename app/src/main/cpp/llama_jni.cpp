#include <android/log.h>
#include <jni.h>
#include <iomanip>
#include <cmath>
#include <string>
#include <unistd.h>
#include <sampling.h>

#include "logging.h"
#include "chat.h"
#include "common.h"
#include "llama.h"

// JNI bridge between the Kotlin/Compose UI (com.edgeslm.app) and llama.cpp.
// Inference runs on the Adreno GPU via the OpenCL backend (see app/src/main/cpp/CMakeLists.txt);
// llama_model_default_params() offloads all layers to GPU (n_gpu_layers = -1) whenever a GPU
// backend is registered, so no CPU fallback path is taken unless the device lacks OpenCL.

constexpr int   N_THREADS_MIN        = 2;
constexpr int   N_THREADS_MAX        = 4;
constexpr int   N_THREADS_HEADROOM   = 2;
constexpr int   DEFAULT_CONTEXT_SIZE = 16384; // reasoning models (e.g. Qwen3's <think> traces) need more room than a plain chat model
constexpr int   OVERFLOW_HEADROOM    = 4;
constexpr int   BATCH_SIZE           = 512;
// Sentinel for "let the model finish on its own", like a normal chat app: bounded only by the
// context window (shift_context() below evicts older turns to keep making room), not by an
// arbitrary reply-length cap. This ceiling only guards against a model that never emits EOG.
constexpr int   UNLIMITED_SAFETY_CAP = 32768;

static llama_model             * g_model    = nullptr;
static llama_context            * g_context  = nullptr;
static llama_batch                g_batch;
static common_chat_templates_ptr  g_chat_templates;
static common_sampler            * g_sampler  = nullptr;

static std::vector<common_chat_msg> chat_msgs;
static llama_pos system_prompt_position = 0;
static llama_pos current_position = 0;
static llama_pos stop_generation_position = 0;
static std::string cached_token_chars;
static std::ostringstream assistant_ss;

constexpr const char *ROLE_SYSTEM    = "system";
constexpr const char *ROLE_USER      = "user";
constexpr const char *ROLE_ASSISTANT = "assistant";

extern "C" JNIEXPORT void JNICALL
Java_com_edgeslm_app_LlamaBridge_nativeInit(JNIEnv *env, jobject, jstring nativeLibDir) {
    llama_log_set(edgeslm_android_log_callback, nullptr);
    const auto *path = env->GetStringUTFChars(nativeLibDir, nullptr);
    ggml_backend_load_all_from_path(path);
    env->ReleaseStringUTFChars(nativeLibDir, path);
    llama_backend_init();
    LOGi("backend initialized");
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_edgeslm_app_LlamaBridge_nativeSystemInfo(JNIEnv *env, jobject) {
    return env->NewStringUTF(llama_print_system_info());
}

static std::string active_backend_name() {
    std::vector<std::string> backends;
    for (size_t i = 0; i < ggml_backend_reg_count(); i++) {
        auto *reg = ggml_backend_reg_get(i);
        std::string name = ggml_backend_reg_name(reg);
        if (name != "CPU") backends.push_back(name);
    }
    return backends.empty() ? "CPU" : backends.front();
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_edgeslm_app_LlamaBridge_nativeActiveBackend(JNIEnv *env, jobject) {
    return env->NewStringUTF(active_backend_name().c_str());
}

extern "C" JNIEXPORT jint JNICALL
Java_com_edgeslm_app_LlamaBridge_nativeLoadModel(JNIEnv *env, jobject, jstring jmodelPath, jint nCtx) {
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = -1; // offload every layer to the GPU backend

    const auto *model_path = env->GetStringUTFChars(jmodelPath, nullptr);
    LOGi("loading model: %s", model_path);
    auto *model = llama_model_load_from_file(model_path, model_params);
    env->ReleaseStringUTFChars(jmodelPath, model_path);
    if (!model) {
        LOGe("failed to load model");
        return 1;
    }
    g_model = model;

    const int n_threads = std::max(N_THREADS_MIN, std::min(N_THREADS_MAX,
            (int) sysconf(_SC_NPROCESSORS_ONLN) - N_THREADS_HEADROOM));

    llama_context_params ctx_params = llama_context_default_params();
    const int ctx_size = nCtx > 0 ? nCtx : DEFAULT_CONTEXT_SIZE;
    ctx_params.n_ctx = ctx_size;
    ctx_params.n_batch = BATCH_SIZE;
    ctx_params.n_ubatch = BATCH_SIZE;
    ctx_params.n_threads = n_threads;
    ctx_params.n_threads_batch = n_threads;

    auto *context = llama_init_from_model(g_model, ctx_params);
    if (!context) {
        LOGe("failed to create context");
        llama_model_free(g_model);
        g_model = nullptr;
        return 2;
    }
    g_context = context;
    g_batch = llama_batch_init(BATCH_SIZE, 0, 1);
    g_chat_templates = common_chat_templates_init(g_model, "");

    common_params_sampling sparams;
    sparams.temp = 0.7f;
    sparams.top_p = 0.9f;
    sparams.top_k = 40;
    // Without a repetition penalty, small quantized models can loop through near-identical
    // reasoning phrases forever instead of concluding. penalty_repeat defaults to 1.0 (a no-op).
    sparams.penalty_repeat = 1.1f;
    sparams.penalty_last_n = 256;
    // DRY catches verbatim repeated *sequences* (whole repeated sentences/paragraphs - the
    // "circling" symptom), which a plain per-token repeat penalty above is too weak to stop.
    sparams.dry_multiplier = 0.8f;
    sparams.dry_base = 1.75f;
    sparams.dry_allowed_length = 2;
    sparams.dry_penalty_last_n = -1; // scan the full context

    // Cap how long the model is allowed to stay inside a <think>...</think> block: once the
    // budget runs out mid-thought, the reasoning-budget sampler force-emits the closing tag so
    // the model is pushed into producing the actual answer instead of thinking indefinitely.
    // This is a no-op passthrough for models that never emit a "<think>" sequence at all.
    const auto think_start = common_tokenize(g_context, "<think>", false, true);
    const auto think_end   = common_tokenize(g_context, "</think>", false, true);
    if (!think_start.empty() && !think_end.empty()) {
        sparams.reasoning_budget_start = think_start;
        sparams.reasoning_budget_end = think_end;
        sparams.reasoning_budget_forced = think_end;
        sparams.reasoning_budget_tokens = 2048;
    }

    g_sampler = common_sampler_init(g_model, sparams);

    chat_msgs.clear();
    system_prompt_position = current_position = 0;

    LOGi("model ready, ctx=%d threads=%d backend=%s", ctx_size, n_threads, active_backend_name().c_str());
    return 0;
}

static void shift_context() {
    const int n_discard = (current_position - system_prompt_position) / 2;
    llama_memory_seq_rm(llama_get_memory(g_context), 0, system_prompt_position, system_prompt_position + n_discard);
    llama_memory_seq_add(llama_get_memory(g_context), 0, system_prompt_position + n_discard, current_position, -n_discard);
    current_position -= n_discard;
}

static std::string chat_add_and_format(const std::string &role, const std::string &content) {
    common_chat_msg new_msg;
    new_msg.role = role;
    new_msg.content = content;
    auto formatted = common_chat_format_single(
            g_chat_templates.get(), chat_msgs, new_msg, role == ROLE_USER, /* use_jinja */ false);
    chat_msgs.push_back(new_msg);
    return formatted;
}

static int decode_tokens_in_batches(const llama_tokens &tokens, llama_pos start_pos, bool compute_last_logit) {
    const int max_ctx = (int) llama_n_ctx(g_context) - OVERFLOW_HEADROOM;
    for (int i = 0; i < (int) tokens.size(); i += BATCH_SIZE) {
        const int cur_batch_size = std::min((int) tokens.size() - i, BATCH_SIZE);
        common_batch_clear(g_batch);
        if (start_pos + i + cur_batch_size >= max_ctx) {
            shift_context();
        }
        for (int j = 0; j < cur_batch_size; j++) {
            const bool want_logit = compute_last_logit && (i + j == (int) tokens.size() - 1);
            common_batch_add(g_batch, tokens[i + j], start_pos + i + j, {0}, want_logit);
        }
        if (llama_decode(g_context, g_batch) != 0) {
            LOGe("llama_decode failed during prompt processing");
            return 1;
        }
    }
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_edgeslm_app_LlamaBridge_nativeProcessSystemPrompt(JNIEnv *env, jobject, jstring jsystemPrompt) {
    chat_msgs.clear();
    system_prompt_position = current_position = 0;
    cached_token_chars.clear();
    assistant_ss.str("");
    llama_memory_clear(llama_get_memory(g_context), false);

    const auto *raw = env->GetStringUTFChars(jsystemPrompt, nullptr);
    std::string prompt(raw);
    env->ReleaseStringUTFChars(jsystemPrompt, raw);
    if (prompt.empty()) {
        return 0;
    }

    const bool has_template = common_chat_templates_was_explicit(g_chat_templates.get());
    std::string formatted = has_template ? chat_add_and_format(ROLE_SYSTEM, prompt) : prompt;
    auto tokens = common_tokenize(g_context, formatted, has_template, has_template);

    if (decode_tokens_in_batches(tokens, current_position, false)) {
        return 2;
    }
    system_prompt_position = current_position = (int) tokens.size();
    return 0;
}

extern "C" JNIEXPORT jint JNICALL
Java_com_edgeslm_app_LlamaBridge_nativeProcessUserPrompt(JNIEnv *env, jobject, jstring juserPrompt, jint nPredict) {
    cached_token_chars.clear();
    assistant_ss.str("");

    const auto *raw = env->GetStringUTFChars(juserPrompt, nullptr);
    std::string prompt(raw);
    env->ReleaseStringUTFChars(juserPrompt, raw);

    const bool has_template = common_chat_templates_was_explicit(g_chat_templates.get());
    std::string formatted = has_template ? chat_add_and_format(ROLE_USER, prompt) : prompt;
    auto tokens = common_tokenize(g_context, formatted, has_template, has_template);

    const int max_ctx = (int) llama_n_ctx(g_context) - OVERFLOW_HEADROOM;
    if ((int) tokens.size() > max_ctx) {
        tokens.resize(max_ctx);
    }

    if (decode_tokens_in_batches(tokens, current_position, true)) {
        return 2;
    }
    current_position += (int) tokens.size();
    stop_generation_position = current_position + (nPredict > 0 ? nPredict : UNLIMITED_SAFETY_CAP);
    return 0;
}

static bool is_valid_utf8(const std::string &s) {
    const auto *bytes = (const unsigned char *) s.c_str();
    while (*bytes) {
        int num;
        if ((*bytes & 0x80) == 0x00) num = 1;
        else if ((*bytes & 0xE0) == 0xC0) num = 2;
        else if ((*bytes & 0xF0) == 0xE0) num = 3;
        else if ((*bytes & 0xF8) == 0xF0) num = 4;
        else return false;
        bytes++;
        for (int i = 1; i < num; i++) {
            if ((*bytes & 0xC0) != 0x80) return false;
            bytes++;
        }
    }
    return true;
}

extern "C" JNIEXPORT jstring JNICALL
Java_com_edgeslm_app_LlamaBridge_nativeGenerateNextToken(JNIEnv *env, jobject) {
    const int max_ctx = (int) llama_n_ctx(g_context) - OVERFLOW_HEADROOM;
    if (current_position >= max_ctx) {
        shift_context();
    }
    if (current_position >= stop_generation_position) {
        return nullptr;
    }

    const auto new_token = common_sampler_sample(g_sampler, g_context, -1);
    common_sampler_accept(g_sampler, new_token, true);

    common_batch_clear(g_batch);
    common_batch_add(g_batch, new_token, current_position, {0}, true);
    if (llama_decode(g_context, g_batch) != 0) {
        LOGe("llama_decode failed during generation");
        return nullptr;
    }
    current_position++;

    if (llama_vocab_is_eog(llama_model_get_vocab(g_model), new_token)) {
        chat_add_and_format(ROLE_ASSISTANT, assistant_ss.str());
        return nullptr;
    }

    cached_token_chars += common_token_to_piece(g_context, new_token);
    if (is_valid_utf8(cached_token_chars)) {
        assistant_ss << cached_token_chars;
        jstring result = env->NewStringUTF(cached_token_chars.c_str());
        cached_token_chars.clear();
        return result;
    }
    return env->NewStringUTF("");
}

extern "C" JNIEXPORT void JNICALL
Java_com_edgeslm_app_LlamaBridge_nativeUnloadModel(JNIEnv *, jobject) {
    if (g_sampler) { common_sampler_free(g_sampler); g_sampler = nullptr; }
    g_chat_templates.reset();
    if (g_batch.token) { llama_batch_free(g_batch); g_batch = {}; }
    if (g_context) { llama_free(g_context); g_context = nullptr; }
    if (g_model) { llama_model_free(g_model); g_model = nullptr; }
    chat_msgs.clear();
    system_prompt_position = current_position = stop_generation_position = 0;
}

extern "C" JNIEXPORT void JNICALL
Java_com_edgeslm_app_LlamaBridge_nativeShutdown(JNIEnv *, jobject) {
    llama_backend_free();
}
