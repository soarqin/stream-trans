# transcribe.cpp keeps its public ABI while using the exact ggml built for llama.cpp.
foreach(required STREAMTRANS_GGML_SOURCE STREAMTRANS_GGML_BUILD)
    if(NOT DEFINED ${required})
        message(FATAL_ERROR "Missing ${required}; run scripts/build-asr.ps1")
    endif()
endforeach()
add_library(ggml INTERFACE)
target_include_directories(ggml INTERFACE
    "${STREAMTRANS_GGML_SOURCE}/include"
    "${STREAMTRANS_GGML_SOURCE}/src")
target_compile_definitions(ggml INTERFACE GGML_SHARED GGML_BACKEND_SHARED)
target_link_libraries(ggml INTERFACE
    "${STREAMTRANS_GGML_BUILD}/ggml.lib"
    "${STREAMTRANS_GGML_BUILD}/ggml-base.lib")
