#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'USAGE'
用法: run_capture_and_parse.sh [选项]

选项:
  --targets <path>   指定 targets.tsv（默认 scripts/bash_metric/targets.tsv）
  --topn <N>         Top-N kernel 数量（默认读取环境变量 TOPN 或 10）
  --nvtx             捕获阶段启用 NVTX
  --out-base <path>  自定义输出根目录（默认 scripts/bash_metric/out_nsys_ncu_all）
  -h, --help         显示此帮助

该脚本顺序调用 capture_once.py 与 analyze_topk.py。
USAGE
}

SCRIPT_DIR=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)
CAPTURE_PY="${SCRIPT_DIR}/capture_once.py"
ANALYZE_PY="${SCRIPT_DIR}/analyze_topk.py"

PYTHON_BIN=${PYTHON_BIN:-python3}
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    if command -v python3 >/dev/null 2>&1; then
        PYTHON_BIN=python3
    elif command -v python >/dev/null 2>&1; then
        PYTHON_BIN=python
    else
        echo "[ERROR] 未找到可用的 python 解释器" >&2
        exit 1
    fi
fi

TARGETS=""
TOPN=""
OUT_BASE=""
NVTX=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --targets)
            [[ $# -ge 2 ]] || { echo "--targets 需要参数" >&2; exit 1; }
            TARGETS="$2"
            shift 2
            ;;
        --topn)
            [[ $# -ge 2 ]] || { echo "--topn 需要参数" >&2; exit 1; }
            TOPN="$2"
            shift 2
            ;;
        --out-base)
            [[ $# -ge 2 ]] || { echo "--out-base 需要参数" >&2; exit 1; }
            OUT_BASE="$2"
            shift 2
            ;;
        --nvtx)
            NVTX=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "未知参数: $1" >&2
            usage
            exit 1
            ;;
    esac
done

capture_args=()
analyze_args=()

if [[ -n "${TARGETS}" ]]; then
    capture_args+=("--targets" "${TARGETS}")
    analyze_args+=("--targets" "${TARGETS}")
fi

if [[ -n "${TOPN}" ]]; then
    capture_args+=("--topn" "${TOPN}")
    analyze_args+=("--topn" "${TOPN}")
fi

if [[ -n "${OUT_BASE}" ]]; then
    capture_args+=("--out-base" "${OUT_BASE}")
    analyze_args+=("--out-base" "${OUT_BASE}")
fi

if [[ ${NVTX} -eq 1 ]]; then
    capture_args+=("--nvtx")
fi

echo "[INFO] 运行捕获阶段: ${PYTHON_BIN} ${CAPTURE_PY} ${capture_args[*]}"
"${PYTHON_BIN}" "${CAPTURE_PY}" "${capture_args[@]}"

echo "[INFO] 运行解析阶段: ${PYTHON_BIN} ${ANALYZE_PY} ${analyze_args[*]}"
"${PYTHON_BIN}" "${ANALYZE_PY}" "${analyze_args[@]}"

echo "[INFO] 流水线执行完成"
