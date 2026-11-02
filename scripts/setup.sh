#!/usr/bin/env bash
set -Eeuo pipefail
trap 'printf "安装失败，请修复上方错误后重新运行 scripts/setup.sh。\n" >&2' ERR

if [[ $(uname -s) != Linux || $(uname -m) != x86_64 ]]; then
    printf '此脚本仅支持 Linux x86_64 NVIDIA CUDA。Windows 请运行 scripts/setup.ps1。\n' >&2
    exit 1
fi
if [[ $# -ne 0 ]]; then
    printf '用法：bash scripts/setup.sh\n' >&2
    exit 1
fi
for command in cargo c++ ffmpeg nvidia-smi curl sha256sum tar; do
    if ! command -v "$command" >/dev/null; then
        printf '找不到 %s。请先安装 Rust、C++ 编译器、NVIDIA 驱动、FFmpeg、curl 和 tar。\n' "$command" >&2
        exit 1
    fi
done

project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
native_root="$project_root/.tools/transcribe"
llama_root="$project_root/.tools/llama"
downloads="$project_root/.tools/downloads"
mkdir -p "$native_root" "$llama_root" "$downloads"

download_and_extract() {
    local asset="$1" sha256="$2" base_url="$3" destination="$4"
    local archive="$downloads/$asset"
    if [[ ! -f "$archive" ]] || ! printf '%s  %s\n' "$sha256" "$archive" | sha256sum --check --status; then
        printf '下载 %s\n' "$asset"
        curl --fail --location --retry 3 "$base_url/$asset" --output "$archive.partial"
        printf '%s  %s\n' "$sha256" "$archive.partial" | sha256sum --check --status
        mv -- "$archive.partial" "$archive"
    fi
    tar -xzf "$archive" -C "$destination" --strip-components=1
}

download_and_extract \
    transcribe-native-0.4.1-linux-x86_64-cuda.tar.gz \
    c4bdf9c4798bac1868ecdcaa04f27eb7712fb43125268313ce14d508c70d247d \
    https://github.com/handy-computer/transcribe.cpp/releases/download/v0.4.1 \
    "$native_root"
download_and_extract \
    llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz \
    64d57beb9abb637c177c6afb313760a0c0782224cac71e74869eed5b8f9a1ea4 \
    https://github.com/ggml-org/llama.cpp/releases/download/b11960 \
    "$llama_root"
download_and_extract \
    cudart-llama-b11960-bin-ubuntu-cuda-12.8-x64.tar.gz \
    7c4d5974bd39649df649649e5d71950688a70e37aaae00bf11afbb656ea2cc56 \
    https://github.com/ggml-org/llama.cpp/releases/download/b11960 \
    "$llama_root"
if [[ ! -f "$native_root/libtranscribe.so" ]]; then
    printf '解压后找不到 %s/libtranscribe.so，请检查官方 v0.4.1 CUDA 压缩包。\n' "$native_root" >&2
    exit 1
fi
if [[ ! -x "$llama_root/llama-server" ]]; then
    printf '解压后找不到可执行的 %s/llama-server，请检查官方 b11960 Ubuntu CUDA 压缩包。\n' "$llama_root" >&2
    exit 1
fi
for library in libcudart.so.12 libcublas.so.12 libcublasLt.so.12; do
    if [[ ! -f "$llama_root/$library" ]]; then
        printf 'CUDA 运行库缺失：%s/%s。请重新运行安装脚本。\n' "$llama_root" "$library" >&2
        exit 1
    fi
done
cd -- "$project_root"
cargo build --release --locked
printf '构建完成，不需要 Python、NVCC 或系统 CUDA Toolkit。Linux 流程尚未实机验证；官方 llama.cpp 包面向 Ubuntu 24.04 构建。\n'
printf '从项目根目录启动：\nLD_LIBRARY_PATH=%q ./target/release/streamtrans --asr-library %q --llama-server %q\n' \
    "$native_root:$llama_root${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
    "$native_root/libtranscribe.so" "$llama_root/llama-server"
