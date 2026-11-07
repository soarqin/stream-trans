// Included in the fixed b11960 server.cpp; shares its private route helpers.
#include "streamtrans_llama_embedded.h"

#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>

#if defined(_WIN32)
#define STREAMTRANS_EXPORT extern "C" __declspec(dllexport)
#else
#define STREAMTRANS_EXPORT extern "C" __attribute__((visibility("default")))
#endif

// Private bridge in llama.cpp: reports real layer placements, not requested flags.
extern "C" LLAMA_API int32_t streamtrans_llama_gpu_layers(
    const llama_model *, ggml_backend_dev_t, uint32_t *);

struct streamtrans_server {
    std::string model_path;
    std::string api_key;
    uint32_t gpu;
    uint32_t port;
    uint32_t context_size;
    std::atomic<bool> stop{false};
    std::atomic<bool> ran{false};
    std::mutex mutex;
    server_http_context * http = nullptr;
    server_context * context = nullptr;
    streamtrans_server_receipt receipt{};
};

// Upstream stream-session GC and common logger are process-wide singletons.
static std::mutex streamtrans_active_server;

struct streamtrans_failure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

static void streamtrans_copy_message(char * out, size_t capacity, const char * message) {
    if (out && capacity) {
        const size_t count = std::min(capacity - 1, std::strlen(message));
        std::memcpy(out, message, count);
        out[count] = '\0';
    }
}

static bool streamtrans_load_progress(float, void * opaque) {
    return !static_cast<streamtrans_server *>(opaque)->stop.load();
}

static bool streamtrans_abort(void * opaque) {
    return static_cast<streamtrans_server *>(opaque)->stop.load();
}

static void streamtrans_load_local_backends() {
    // GGML's default search uses the executable directory, not this DLL's
    // directory. Standalone LLM callers have no ASR to preload the backends.
    if (ggml_backend_reg_count()) { return; }
#if defined(_WIN32)
    static const char module_anchor = 0;
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&module_anchor), &module)) {
        throw streamtrans_failure("cannot locate embedded server module");
    }
    std::vector<wchar_t> path(32768);
    const DWORD length = GetModuleFileNameW(module, path.data(), DWORD(path.size()));
    if (!length || length >= path.size()) {
        throw streamtrans_failure("cannot locate embedded server directory");
    }
    const std::wstring full_path(path.data(), length);
    const size_t separator = full_path.find_last_of(L"\\/");
    if (separator == std::wstring::npos) {
        throw streamtrans_failure("invalid embedded server directory");
    }
    const int size = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, full_path.data(), int(separator),
        nullptr, 0, nullptr, nullptr);
    if (size <= 0) { throw streamtrans_failure("invalid embedded server directory encoding"); }
    std::string directory(size, '\0');
    if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, full_path.data(), int(separator),
            &directory[0], size, nullptr, nullptr) != size) {
        throw streamtrans_failure("cannot encode embedded server directory");
    }
    ggml_backend_load_all_from_path(directory.c_str());
#endif
}

STREAMTRANS_EXPORT uint32_t streamtrans_server_abi_version(void) {
    return 1;
}

STREAMTRANS_EXPORT streamtrans_server * streamtrans_server_create(
        const streamtrans_server_config * config, char * error, size_t error_capacity) {
    try {
        if (!config || config->abi_version != 1 || config->struct_size != sizeof(*config)) {
            throw std::invalid_argument("unsupported embedded server config ABI");
        }
        if (!config->model_path || !*config->model_path || !config->api_key || !*config->api_key ||
                !config->port || config->port > 65535 || config->context_size < 512 ||
                config->context_size > uint32_t(std::numeric_limits<int32_t>::max())) {
            throw std::invalid_argument("invalid embedded model, key, port or context size");
        }
        auto server = std::make_unique<streamtrans_server>();
        server->model_path = config->model_path;
        server->api_key = config->api_key;
        server->gpu = config->gpu;
        server->port = config->port;
        server->context_size = config->context_size;
        server->receipt.abi_version = 1;
        server->receipt.struct_size = sizeof(server->receipt);
        server->receipt.result = -1;
        server->receipt.build_number = llama_build_number();
        streamtrans_copy_message(error, error_capacity, "");
        return server.release();
    } catch (const std::exception &) {
        // Never echo caller strings (the API key may be among them).
        streamtrans_copy_message(error, error_capacity, "cannot create embedded server: invalid config or allocation failure");
    } catch (...) {
        streamtrans_copy_message(error, error_capacity, "cannot create embedded server");
    }
    return nullptr;
}

STREAMTRANS_EXPORT void streamtrans_server_request_stop(streamtrans_server * server) {
    if (!server) { return; }
    server->stop.store(true);
    std::lock_guard<std::mutex> lock(server->mutex);
    if (server->context) { server->context->terminate(); }
    if (server->http) { server->http->stop(); }
}

STREAMTRANS_EXPORT int32_t streamtrans_server_status(streamtrans_server * server, streamtrans_server_receipt * out) {
    if (!server || !out || out->abi_version != 1 || out->struct_size != sizeof(*out)) { return -1; }
    std::lock_guard<std::mutex> lock(server->mutex);
    *out = server->receipt;
    return 0;
}

static int streamtrans_run_instance(streamtrans_server & server) {
    // Typed parameters deliberately bypass CLI/env/config parsing and process-wide
    // locale, console codepage, signal handlers, standard streams and cwd changes.
    common_params params;
    params.embedded_host = true;
    params.model.path = server.model_path;
    params.model_alias = {"Hy-MT2-1.8B"};
    params.hostnames = {"127.0.0.1"};
    params.port = server.port;
    params.api_keys = {server.api_key};
    params.n_ctx = server.context_size;
    params.n_parallel = 1;
    params.n_gpu_layers = -2;
    params.split_mode = LLAMA_SPLIT_MODE_NONE;
    params.fit_params = false;
    params.warmup = true;
    params.ctx_shift = false;
    params.cache_prompt = false;
    params.cache_idle_slots = false;
    params.cache_ram_mib = 0;
    params.cache_ram_explicit = true;
    params.cache_type_k = GGML_TYPE_F16;
    params.cache_type_v = GGML_TYPE_F16;
    params.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
    params.n_threads_http = 2;
    params.sleep_idle_seconds = -1;
    params.ui = false;
    params.load_progress_callback = streamtrans_load_progress;
    params.load_progress_callback_user_data = &server;
    params.abort_callback = streamtrans_abort;
    params.abort_callback_data = &server;
    postprocess_cpu_params(params.cpuparams);
    postprocess_cpu_params(params.cpuparams_batch, &params.cpuparams);

    if (server.stop.load()) { return 0; }
    streamtrans_load_local_backends();
    llama_backend_init();
    const std::string device_name = "CUDA" + std::to_string(server.gpu);
    auto * device = ggml_backend_dev_by_name(device_name.c_str());
    if (!device || ggml_backend_dev_type(device) != GGML_BACKEND_DEVICE_TYPE_GPU) {
        throw streamtrans_failure("requested CUDA device is unavailable");
    }
    params.devices = {device, nullptr};

    server_mcp mcp;
    server_context context;
    server_routes routes(params, context);
    server_tools tools;
    server_http_context http;

    // Detach stack pointers before any object is destroyed, including exceptions.
    struct cleanup {
        streamtrans_server & server;
        server_http_context & http;
        server_context & context;
        server_mcp & mcp;
        ~cleanup() {
            {
                std::lock_guard<std::mutex> lock(server.mutex);
                server.http = nullptr;
                server.context = nullptr;
            }
            server_stream_session_manager_stop();
            http.stop();
            context.terminate();
            mcp.shutdown();
            http.join();
            // Backend registry and common logger intentionally remain process-owned.
        }
    } guard{server, http, context, mcp};

    if (server.stop.load()) { return 0; }
    if (!http.init(params)) { throw streamtrans_failure("cannot initialize embedded HTTP listener"); }
    server_register_routes(http, routes);
    server_register_stream_routes(http, server_stream_make_get_handler(),
        server_stream_make_lookup_handler(), server_stream_make_delete_handler());
    if (!server_setup_features(http, params, mcp, tools, SERVER_ROLE_INSTANCE)) {
        throw streamtrans_failure("cannot initialize embedded server routes");
    }
    server_stream_session_manager_start();
    {
        std::lock_guard<std::mutex> lock(server.mutex);
        server.http = &http;
        server.context = &context;
        if (server.stop.load()) { return 0; }
        if (!http.start()) { throw streamtrans_failure("cannot start embedded HTTP listener"); }
    }
    if (!context.load_model(params)) {
        if (server.stop.load()) { return 0; }
        throw streamtrans_failure("model load or GPU context initialization failed");
    }
    if (server.stop.load()) { return 0; }

    uint32_t total_layers = 0;
    const int32_t gpu_layers = streamtrans_llama_gpu_layers(
        llama_get_model(context.get_llama_context()), device, &total_layers);
    if (gpu_layers < 0 || !total_layers || uint32_t(gpu_layers) != total_layers) {
        throw streamtrans_failure("model was not fully offloaded to the requested CUDA device");
    }
    routes.update_meta(context);
    {
        std::lock_guard<std::mutex> lock(server.mutex);
        server.receipt.gpu_layers = gpu_layers;
        server.receipt.total_layers = total_layers;
        streamtrans_copy_message(server.receipt.device, sizeof(server.receipt.device), device_name.c_str());
        server.receipt.state = 2;
        http.is_ready.store(true);
        if (server.stop.load()) { context.terminate(); }
    }
    context.start_loop(); // terminate() is sticky even before start_loop().
    return 0;
}

STREAMTRANS_EXPORT int32_t streamtrans_server_run(streamtrans_server * server) {
    if (!server || server->ran.exchange(true)) { return -1; }
    std::unique_lock<std::mutex> active(streamtrans_active_server, std::try_to_lock);
    int32_t result = 1;
    std::string failure = "embedded server failed";
    try {
        if (!active.owns_lock()) { throw streamtrans_failure("an embedded server is already running"); }
        {
            std::lock_guard<std::mutex> lock(server->mutex);
            server->receipt.state = 1;
        }
        result = streamtrans_run_instance(*server);
    } catch (const streamtrans_failure & e) {
        failure = e.what();
    } catch (const std::exception & e) {
        // Messages originate only from fixed bridge errors; never forward arbitrary
        // upstream exception strings, which may include prompts or file contents.
        (void)e;
    } catch (...) {}
    {
        std::lock_guard<std::mutex> lock(server->mutex);
        server->receipt.result = result;
        server->receipt.state = result == 0 ? 3 : 4;
        streamtrans_copy_message(server->receipt.message, sizeof(server->receipt.message), result == 0 ? "" : failure.c_str());
    }
    return result;
}

STREAMTRANS_EXPORT void streamtrans_server_destroy(streamtrans_server * server) {
    delete server;
}
