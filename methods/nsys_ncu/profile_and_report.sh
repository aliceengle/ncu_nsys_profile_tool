#!/usr/bin/env bash
# profile_and_report.sh (nsys+ncu method)
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGETS_TSV="${TARGETS_TSV:-${SCRIPT_DIR}/../../targets.tsv}"
OUT_DIR="${OUT_DIR:-${SCRIPT_DIR}/../../out}"
PY="${PYTHON_BIN:-python3}"
BASH_BIN="${BASH_BIN:-/bin/bash}"

NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx}"
NSYS_DURATION="${NSYS_DURATION:-}"
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-none}"
NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END:-stop}"
NSYS_QUIET="${NSYS_QUIET:-false}"

NCU_TOPN="${NCU_TOPN:-5}"
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"
NCU_FAST="${NCU_FAST:-0}"
if [[ "${NCU_FAST}" == "1" ]]; then NCU_SET="${NCU_SET:-speed-of-light}"; else NCU_SET="${NCU_SET:-full}"; fi
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"
NCU_MULTI_RUN="${NCU_MULTI_RUN:-1}"  # 1: 按 kernel 逐个运行 ncu；0: 单次运行匹配多个

have_cmd () { command -v "$1" >/dev/null 2>&1; }
ts() { date +"%Y-%m-%d %H:%M:%S"; }
die() { echo "[FATAL $(ts)] $*" >&2; exit 1; }

[[ -f "${TARGETS_TSV}" ]] || die "Targets TSV not found: ${TARGETS_TSV}"
have_cmd nsys || die "nsys not found in PATH"
have_cmd ncu  || die "ncu not found in PATH"
have_cmd "${PY}" || die "python3 not found"
{ [[ -x "${BASH_BIN}" ]] || have_cmd bash; } || die "bash not found"

GEN_TOPK_PY="${SCRIPT_DIR}/gen_topk_regex.py"
[[ -f "${GEN_TOPK_PY}" ]] || die "Missing helper Python script: ${GEN_TOPK_PY}"

mkdir -p "${OUT_DIR}"
GLOBAL_LOG="${GLOBAL_LOG:-${OUT_DIR}/profile_and_report.log}"
: > "${GLOBAL_LOG}"
echo "[START $(ts)] profile_and_report.sh run" | tee -a "${GLOBAL_LOG}"
print_run_config "${GLOBAL_LOG}"

run_and_log() {
  local _log="$1"; shift
  { echo "[$(ts)] CMD: $*"; "$@" 2>&1; local rc=$?; echo "[$(ts)] RC: $rc"; return $rc; } | tee -a "${_log}"
}
run_and_capture_stdout() {
  local _log="$1"; shift; local _outfile="$1"; shift
  { echo "[$(ts)] CMD: $*"; "$@" 1>"${_outfile}"; local rc=$?; echo "[$(ts)] RC: $rc"; return $rc; } 2>&1 | tee -a "${_log}"
}
check_csv_basic() {
  local _f="$1"; local _log="$2"; local _tag="$3"; if [[ ! -s "${_f}" ]]; then echo "[CHECK] ${_tag}: 文件不存在或为空 -> ${_f}" | tee -a "${_log}"; return 1; fi
  local header; header=$(head -n 1 "${_f}" 2>/dev/null || true)
  if [[ -z "${header}" ]]; then echo "[CHECK] ${_tag}: 首行为空 -> ${_f}" | tee -a "${_log}"; return 1; fi
  if [[ "${header}" != *","* && "${header}" != *";"* ]]; then echo "[CHECK] ${_tag}: 首行未检测到逗号/分号分隔 -> ${_f}" | tee -a "${_log}"; return 1; fi
  if [[ "${header}" == *"CMD:"* || "${header}" == *"RC:"* || "${header}" == *"STEP"* ]]; then echo "[CHECK] ${_tag}: 首行疑似被日志污染 -> ${_f}" | tee -a "${_log}"; return 1; fi
  echo "[CHECK] ${_tag}: CSV 头部看起来正常 -> ${_f}" | tee -a "${_log}"
}

print_run_config() {
  # $1 = logfile
  local _log="$1"
  echo "----- Run Config -----" | tee -a "${_log}"
  echo "CWD                : $(pwd)" | tee -a "${_log}"
  echo "SCRIPT_DIR         : ${SCRIPT_DIR}" | tee -a "${_log}"
  echo "TARGETS_TSV        : ${TARGETS_TSV}" | tee -a "${_log}"
  echo "OUT_DIR            : ${OUT_DIR}" | tee -a "${_log}"
  echo "PY (bin)           : ${PY} ($(command -v "${PY}" 2>/dev/null || true))" | tee -a "${_log}"
  echo "BASH_BIN           : ${BASH_BIN}" | tee -a "${_log}"
  echo "nsys (bin)         : $(command -v nsys 2>/dev/null || echo not-found)" | tee -a "${_log}"
  echo "ncu  (bin)         : $(command -v ncu 2>/dev/null || echo not-found)" | tee -a "${_log}"
  echo "NSYS_TRACE         : ${NSYS_TRACE}" | tee -a "${_log}"
  echo "NSYS_DURATION      : ${NSYS_DURATION}" | tee -a "${_log}"
  echo "NSYS_CAPTURE_RANGE : ${NSYS_CAPTURE_RANGE}" | tee -a "${_log}"
  echo "NCU_SET            : ${NCU_SET}" | tee -a "${_log}"
  echo "NCU_TOPN           : ${NCU_TOPN}" | tee -a "${_log}"
  echo "NCU_LAUNCH_COUNT   : ${NCU_LAUNCH_COUNT}" | tee -a "${_log}"
  echo "NCU_MULTI_RUN      : ${NCU_MULTI_RUN}" | tee -a "${_log}"
  echo "CONTINUE_ON_ERROR  : ${CONTINUE_ON_ERROR}" | tee -a "${_log}"
  echo "Helper(gen_topk)   : ${GEN_TOPK_PY}" | tee -a "${_log}"
  # Count targets
  local _tcount
  _tcount=$(awk -F '\t' 'NF>=2 { if ($1 ~ /^#/){next}; l=tolower($1); if (l=="id" || l=="name"){next}; c++ } END{print c+0}' "${TARGETS_TSV}" 2>/dev/null || echo 0)
  echo "Targets to process : ${_tcount}" | tee -a "${_log}"
  echo "---------------------" | tee -a "${_log}"
}

init_stages() { STAGE_NAMES=("准备目录与上下文" "nsys 采集" "nsys 导出 CSV" "生成 TopN kernel 正则" "ncu 采集" "ncu 导出 raw CSV"); STAGE_TOTAL=${#STAGE_NAMES[@]}; STAGE_IDX=0; }
stage_start() { local _log="$1"; local _name="$2"; STAGE_IDX=$((STAGE_IDX+1)); echo "[TARGET ${CURRENT_TARGET}] [STEP ${STAGE_IDX}/${STAGE_TOTAL}] 开始：${_name}" | tee -a "${_log}"; }
stage_done()  { local _log="$1"; local _name="$2"; local _remain=$((STAGE_TOTAL - STAGE_IDX)); echo "[TARGET ${CURRENT_TARGET}] [STEP ${STAGE_IDX}/${STAGE_TOTAL}] 完成：${_name}；剩余 ${_remain} 步" | tee -a "${_log}"; if (( _remain>0 )); then local _next_name="${STAGE_NAMES[${STAGE_IDX}]}"; local _next_idx=$((STAGE_IDX+1)); echo "[TARGET ${CURRENT_TARGET}] [NEXT ${_next_idx}/${STAGE_TOTAL}] 即将执行：${_next_name}" | tee -a "${_log}"; fi; }
stage_fail()  { local _log="$1"; local _name="$2"; local _remain=$((STAGE_TOTAL - STAGE_IDX)); echo "[TARGET ${CURRENT_TARGET}] [STEP ${STAGE_IDX}/${STAGE_TOTAL}] 失败：${_name}；剩余 ${_remain} 步（CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR}）" | tee -a "${_log}"; }

while IFS=$'\t' read -r NAME CMD WORKDIR ENVS; do
  [[ -z "${NAME:-}" ]] && continue
  [[ "${NAME}" =~ ^# ]] && continue
  case "${NAME,,}" in id|name) continue ;; esac
  if [[ -z "${CMD:-}" ]]; then echo "[WARN] Malformed row in ${TARGETS_TSV} for target '${NAME}'. Ensure TAB-separated columns: NAME<TAB>CMD<TAB>WORKDIR<TAB>ENVS" | tee -a "${GLOBAL_LOG}"; continue; fi

  echo | tee -a "${GLOBAL_LOG}"; echo "================== [$(ts)] Target: ${NAME} ==================" | tee -a "${GLOBAL_LOG}"
  TDIR="${OUT_DIR}/${NAME}"; mkdir -p "${TDIR}"; NSYS_LOG="${GLOBAL_LOG}"; NCU_LOG="${GLOBAL_LOG}"
  NSYS_DIR="${TDIR}/nsys"; mkdir -p "${NSYS_DIR}"; NSYS_BASENAME="${NSYS_DIR}/${NAME}"; NSYS_REP="${NSYS_BASENAME}.nsys-rep"; NSYS_SQLITE="${NSYS_BASENAME}.sqlite"
  NCU_DIR="${TDIR}/ncu"; mkdir -p "${NCU_DIR}"; NCU_BASENAME="${NCU_DIR}/${NAME}"; NCU_REP="${NCU_BASENAME}.ncu-rep"; NCU_RAW_CSV="${NCU_BASENAME}_raw.csv"; TOPK_REGEX_TXT="${NCU_DIR}/${NAME}_topk_regex.txt"

  init_stages; CURRENT_TARGET="${NAME}"; stage_start "${NSYS_LOG}" "${STAGE_NAMES[0]}"
  WD="."; if [[ -n "${WORKDIR:-}" && "${WORKDIR}" != "-" ]]; then WD="${WORKDIR}"; [[ -d "${WD}" ]] || { echo "[WARN] WORKDIR not found: ${WD}" | tee -a "${NSYS_LOG}"; }; fi
  ENV_KV=(); if [[ -n "${ENVS//[[:space:]]/}" && "${ENVS//[[:space:]]/}" != "-" ]]; then IFS=',' read -r -a ENV_KV <<< "${ENVS}"; fi
  stage_done "${NSYS_LOG}" "${STAGE_NAMES[0]}"

  stage_start "${NSYS_LOG}" "${STAGE_NAMES[1]}"; echo "[*] ${NAME} :: nsys profile" | tee -a "${NSYS_LOG}"
  NSYS_CMD=(env); [[ ${#ENV_KV[@]} -gt 0 ]] && NSYS_CMD+=("${ENV_KV[@]}")
  NSYS_CMD+=( nsys profile "--trace=${NSYS_TRACE}" "--force-overwrite=true" )
  [[ -n "${NSYS_DURATION}" ]] && NSYS_CMD+=("--duration=${NSYS_DURATION}")
  if [[ "${NSYS_CAPTURE_RANGE}" != "none" ]]; then NSYS_CMD+=("--capture-range=${NSYS_CAPTURE_RANGE}" "--capture-range-end=${NSYS_CAPTURE_RANGE_END}"); fi
  [[ "${NSYS_QUIET}" == "true" ]] && NSYS_CMD+=("--quiet")
  NSYS_CMD+=( "-o" "${NSYS_BASENAME}" "--" "${BASH_BIN}" -lc "${CMD}" )
  if ( cd "${WD}" && run_and_log "${NSYS_LOG}" "${NSYS_CMD[@]}" ); then stage_done "${NSYS_LOG}" "${STAGE_NAMES[1]}"; else echo "[ERROR] nsys profile failed for ${NAME}" | tee -a "${NSYS_LOG}"; stage_fail "${NSYS_LOG}" "${STAGE_NAMES[1]}"; [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1; fi

  stage_start "${NSYS_LOG}" "${STAGE_NAMES[2]}"; echo "[*] ${NAME} :: nsys stats -> CSV" | tee -a "${NSYS_LOG}"
  if run_and_log "${NSYS_LOG}" nsys stats --force-export=true --force-overwrite=true --sqlite "${NSYS_SQLITE}" --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum --format csv --output "${NSYS_BASENAME}" "${NSYS_REP}"; then stage_done "${NSYS_LOG}" "${STAGE_NAMES[2]}"; else echo "[ERROR] nsys stats failed for ${NAME}" | tee -a "${NSYS_LOG}"; stage_fail "${NSYS_LOG}" "${STAGE_NAMES[2]}"; [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1; fi

  KERN_CSV="${NSYS_BASENAME}_cuda_gpu_kern_sum.csv"; stage_start "${NSYS_LOG}" "${STAGE_NAMES[3]}"
  if [[ ! -f "${KERN_CSV}" ]]; then echo "[WARN] Top kernel CSV not found: ${KERN_CSV}" | tee -a "${NSYS_LOG}"; else echo "[*] ${NAME} :: pick Top-${NCU_TOPN} kernels from ${KERN_CSV}" | tee -a "${NSYS_LOG}"; "${PY}" "${GEN_TOPK_PY}" "${KERN_CSV}" "${NCU_TOPN}" "${TOPK_REGEX_TXT}" | tee -a "${NSYS_LOG}" || true; check_csv_basic "${KERN_CSV}" "${NSYS_LOG}" "nsys kern_sum"; fi
  stage_done "${NSYS_LOG}" "${STAGE_NAMES[3]}"

  stage_start "${NCU_LOG}" "${STAGE_NAMES[4]}"; echo "[*] ${NAME} :: ncu on Top${NCU_TOPN} kernels" | tee -a "${NCU_LOG}"
  KREGEX=""; [[ -s "${TOPK_REGEX_TXT}" ]] && KREGEX="$(<"${TOPK_REGEX_TXT}")" || echo "[WARN] No kernel regex generated; will let ncu match first launches only." | tee -a "${NCU_LOG}"
  if [[ "${NCU_MULTI_RUN}" == "1" && -n "${KREGEX}" ]]; then
    IFS='|' read -r -a KLIST <<< "${KREGEX}"
    for pat in "${KLIST[@]}"; do
      [[ -z "${pat}" ]] && continue
      local_base="${NCU_DIR}/${NAME}__$(echo "${pat}" | sed 's/[^A-Za-z0-9._-]/_/g' | cut -c1-64)"
      NCU_CMD=(env); [[ ${#ENV_KV[@]} -gt 0 ]] && NCU_CMD+=("${ENV_KV[@]}")
      NCU_CMD+=( ncu -f "--target-processes" all "--kernel-name-base" demangled "--launch-count" "${NCU_LAUNCH_COUNT}" "--set" "${NCU_SET}" "-o" "${local_base}.ncu-rep" "-k" "regex:${pat}" )
      NCU_CMD+=( "${BASH_BIN}" -lc "${CMD}" )
      if ( cd "${WD}" && run_and_log "${NCU_LOG}" "${NCU_CMD[@]}" ); then :; else echo "[ERROR] ncu profiling failed for ${NAME} (pattern=${pat})" | tee -a "${NCU_LOG}"; [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1; fi
      # export raw CSV for this kernel
      if run_and_capture_stdout "${NCU_LOG}" "${local_base}_raw.csv" ncu --import "${local_base}.ncu-rep" --page raw --csv; then check_csv_basic "${local_base}_raw.csv" "${NCU_LOG}" "ncu raw (${pat})"; else echo "[ERROR] ncu export raw CSV failed for ${NAME} (pattern=${pat})" | tee -a "${NCU_LOG}"; [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1; fi
    done
    stage_done "${NCU_LOG}" "${STAGE_NAMES[4]}"
  else
    NCU_CMD=(env); [[ ${#ENV_KV[@]} -gt 0 ]] && NCU_CMD+=("${ENV_KV[@]}")
    NCU_CMD+=( ncu -f "--target-processes" all "--kernel-name-base" demangled "--launch-count" "${NCU_LAUNCH_COUNT}" "--set" "${NCU_SET}" "-o" "${NCU_REP}" )
    [[ -n "${KREGEX}" ]] && NCU_CMD+=("-k" "regex:${KREGEX}")
    NCU_CMD+=( "${BASH_BIN}" -lc "${CMD}" )
    if ( cd "${WD}" && run_and_log "${NCU_LOG}" "${NCU_CMD[@]}" ); then stage_done "${NCU_LOG}" "${STAGE_NAMES[4]}"; else echo "[ERROR] ncu profiling failed for ${NAME}" | tee -a "${NCU_LOG}"; stage_fail "${NCU_LOG}" "${STAGE_NAMES[4]}"; [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1; fi
  fi

  # 当单次运行模式时导出统一 raw CSV；多次运行模式已在循环中导出各自 raw CSV
  if [[ "${NCU_MULTI_RUN}" != "1" ]]; then
    stage_start "${NCU_LOG}" "${STAGE_NAMES[5]}"; echo "[*] ${NAME} :: ncu export raw CSV" | tee -a "${NCU_LOG}"
    if run_and_capture_stdout "${NCU_LOG}" "${NCU_RAW_CSV}" ncu --import "${NCU_REP}" --page raw --csv; then check_csv_basic "${NCU_RAW_CSV}" "${NCU_LOG}" "ncu raw"; stage_done "${NCU_LOG}" "${STAGE_NAMES[5]}"; else echo "[ERROR] ncu export raw CSV failed for ${NAME}" | tee -a "${NCU_LOG}"; stage_fail "${NCU_LOG}" "${STAGE_NAMES[5]}"; [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1; fi
  fi

  echo "[OK] ${NAME} outputs under: ${TDIR}" | tee -a "${GLOBAL_LOG}"; echo "     - NSYS: ${NSYS_REP}, ${NSYS_SQLITE}, ${KERN_CSV}" | tee -a "${GLOBAL_LOG}"; echo "     - NCU : ${NCU_REP}, ${NCU_RAW_CSV}" | tee -a "${GLOBAL_LOG}"
done < "${TARGETS_TSV}"

echo | tee -a "${GLOBAL_LOG}"; echo "[DONE $(ts)] All targets processed. Outputs at: ${OUT_DIR}" | tee -a "${GLOBAL_LOG}"
