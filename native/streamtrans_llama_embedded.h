#ifndef STREAMTRANS_LLAMA_EMBEDDED_H
#define STREAMTRANS_LLAMA_EMBEDDED_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(STREAMTRANS_EMBEDDED_SERVER)
#define STREAMTRANS_SERVER_API __declspec(dllexport)
#else
#define STREAMTRANS_SERVER_API __declspec(dllimport)
#endif
#else
#define STREAMTRANS_SERVER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ABI 1, natural C alignment. UTF-8 strings are copied by create(). */
typedef struct streamtrans_server_config {
    uint32_t abi_version;
    uint32_t struct_size;
    const char * model_path;
    const char * api_key;
    uint32_t gpu;
    uint32_t port;
    uint32_t context_size;
} streamtrans_server_config;

typedef struct streamtrans_server_receipt {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t state; /* 0 created, 1 loading, 2 ready, 3 stopped, 4 failed */
    int32_t result; /* -1 until run() returns */
    int32_t build_number;
    uint32_t gpu_layers;
    uint32_t total_layers;
    char device[64];
    char message[512]; /* NUL-terminated UTF-8; no prompts or request keys */
} streamtrans_server_receipt;

typedef struct streamtrans_server streamtrans_server;

/* Each handle runs once; only one active embedded server is supported. */
STREAMTRANS_SERVER_API uint32_t streamtrans_server_abi_version(void);
STREAMTRANS_SERVER_API streamtrans_server * streamtrans_server_create(const streamtrans_server_config *, char * error, size_t error_capacity);
/* Blocking: call on a dedicated thread. All C++ exceptions are caught. */
STREAMTRANS_SERVER_API int32_t streamtrans_server_run(streamtrans_server *);
/* Thread-safe and idempotent, including while loading. CUDA calls finish cooperatively. */
STREAMTRANS_SERVER_API void streamtrans_server_request_stop(streamtrans_server *);
/* Initialize receipt.abi_version and receipt.struct_size before calling. */
STREAMTRANS_SERVER_API int32_t streamtrans_server_status(streamtrans_server *, streamtrans_server_receipt *);
/* Only after run returned, with no concurrent callers. Pin the DLL until process exit. */
STREAMTRANS_SERVER_API void streamtrans_server_destroy(streamtrans_server *);

#ifdef __cplusplus
}
#endif
#endif
