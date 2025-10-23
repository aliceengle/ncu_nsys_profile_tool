#!/usr/bin/env bash
# profile_and_report.sh
# Robust, faster-by-default pipeline for Nsight Systems (nsys) + Nsight Compute (ncu)
# - Reads a TSV (targets.tsv): columns = NAME  CMD  WORKDIR  ENVS
# - Profiles each target with nsys, extracts Top-N kernels, then profiles those with ncu
# - Outputs .nsys-rep/.sqlite and nsys stats CSVs + ncu .ncu-rep + raw CSV
#
# Requirements:
#   - nsys 2025+
#   - ncu  2025+
#   - python3 (for small CSV/regex helper)
#
# Notes:
#   * We do NOT use `--set full` in ncu by default to keep runtime reasonable.
#   * When CAPTURE_RANGE=none, we DO NOT add --capture-range-end (avoids the exact error reported).
#   * We force-export and force-overwrite in nsys stats for re-runs.
#   * We avoid `eval` and pass commands via `bash -lc "<CMD>"` to preserve quoting/pipes.

set -Eeuo pipefail

#######################
# Config via env vars #
#######################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGETS_TSV="${TARGETS_TSV:-${SCRIPT_DIR}/targets.tsv}"     # 输入任务表（TSV，默认取脚本同目录）
OUT_DIR="${OUT_DIR:-./out}"                      # 输出根目录
LOG_DIR="${LOG_DIR:-logs}"                        # 日志目录名（在每个任务子目录内）
PY="${PYTHON_BIN:-python3}"

# nsys 配置
NSYS_TRACE="${NSYS_TRACE:-cuda,nvtx}"           # 仅采集 cuda,nvtx 避免额外开销
NSYS_DURATION="${NSYS_DURATION:-}"               # 可选，例如 0（到进程结束）或秒数
NSYS_CAPTURE_RANGE="${NSYS_CAPTURE_RANGE:-none}" # none|nvtx|cudaProfilerApi|hotkey
NSYS_CAPTURE_RANGE_END="${NSYS_CAPTURE_RANGE_END:-stop}" # 仅在 capture-range!=none 时生效
NSYS_QUIET="${NSYS_QUIET:-false}"                # true/false

# ncu 配置
NCU_TOPN="${NCU_TOPN:-5}"                        # 仅对 Top-N kernels 做 ncu
NCU_LAUNCH_COUNT="${NCU_LAUNCH_COUNT:-1}"        # 每个匹配 kernel 采样次数
NCU_FAST="${NCU_FAST:-0}"                         # 1: 跳过 MemoryWorkloadAnalysis 以更快
# 默认 section：LaunchStats/SpeedOfLight/WarpStateStats（快且覆盖率高）
# 如需更细的缓存/带宽分析，再包含 MemoryWorkloadAnalysis
# 为了兼容不同 ncu 版本，优先使用 --set 预置集合，避免各版本 section 名称差异
# NCU_FAST=1 -> 使用 speed-of-light（较快，覆盖核心瓶颈）
# NCU_FAST!=1 -> 使用 full（更全面，但较慢）
if [[ "${NCU_FAST}" == "1" ]]; then
  NCU_SET="${NCU_SET:-speed-of-light}"
else
  NCU_SET="${NCU_SET:-full}"
fi

# 失败时是否继续后续目标
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-1}"      # 1: 继续；0: 立即退出

#################################
# Preflight checks and helpers  #
#################################

have_cmd () { command -v "$1" >/dev/null 2>&1; }
ts() { date +"%Y-%m-%d %H:%M:%S"; }

die() { echo "[FATAL $(ts)] $*" >&2; exit 1; }

[[ -f "${TARGETS_TSV}" ]] || die "Targets TSV not found: ${TARGETS_TSV}"
have_cmd nsys || die "nsys not found in PATH"
have_cmd ncu  || die "ncu not found in PATH"
have_cmd "${PY}" || die "python3 not found"

mkdir -p "${OUT_DIR}"

# 单一全局日志文件（每次运行覆盖重写）
GLOBAL_LOG="${GLOBAL_LOG:-${OUT_DIR}/profile_and_report.log}"
: > "${GLOBAL_LOG}"
echo "[START $(ts)] profile_and_report.sh run" | tee -a "${GLOBAL_LOG}"

run_and_log() {
  # $1 = logfile ; rest = command...
  local _log="$1"; shift
  {
    echo "[$(ts)] CMD: $*"
    "$@" 2>&1
    local rc=$?
    echo "[$(ts)] RC: $rc"
    return $rc
  } | tee -a "${_log}"
}

# 仅捕获命令的标准输出到文件，同时将命令的标准错误与状态记录到日志
# 用于避免 CSV 被日志信息污染（例如 ncu --page raw --csv）
run_and_capture_stdout() {
  # $1 = logfile ; $2 = outfile ; rest = command...
  local _log="$1"; shift
  local _outfile="$1"; shift
  {
    echo "[$(ts)] CMD: $*"
    "$@" 1>"${_outfile}"
    local rc=$?
    echo "[$(ts)] RC: $rc"
    return $rc
  } 2>&1 | tee -a "${_log}"
}

# 简单 CSV 健康度检查：检查首行是否包含分隔符且不含日志噪声
check_csv_basic() {
  # $1 = path ; $2 = logfile ; $3 = tag
  local _f="$1"; local _log="$2"; local _tag="$3"
  if [[ ! -s "${_f}" ]]; then
    echo "[CHECK] ${_tag}: 文件不存在或为空 -> ${_f}" | tee -a "${_log}"
    return 1
  fi
  local header
  header=$(head -n 1 "${_f}" 2>/dev/null || true)
  if [[ -z "${header}" ]]; then
    echo "[CHECK] ${_tag}: 首行为空 -> ${_f}" | tee -a "${_log}"
    return 1
  fi
  # 常见 CSV 逗号分隔；ncu 也可能是逗号或分号，先检查逗号
  if [[ "${header}" != *","* && "${header}" != *";"* ]]; then
    echo "[CHECK] ${_tag}: 首行未检测到逗号/分号分隔，可能不是有效 CSV -> ${_f}" | tee -a "${_log}"
    return 1
  fi
  # 不应包含我们流水线的日志标签
  if [[ "${header}" == *"CMD:"* || "${header}" == *"RC:"* || "${header}" == *"STEP"* ]]; then
    echo "[CHECK] ${_tag}: 首行疑似被日志污染 -> ${_f}" | tee -a "${_log}"
    return 1
  fi
  echo "[CHECK] ${_tag}: CSV 头部看起来正常 -> ${_f}" | tee -a "${_log}"
}

# 阶段管理：为每个目标提供清晰的执行进度输出
init_stages() {
  # 定义本目标的阶段列表（按执行顺序）
  STAGE_NAMES=(
    "准备目录与上下文"
    "nsys 采集"
    "nsys 导出 CSV"
    "生成 TopN kernel 正则"
    "ncu 采集"
    "ncu 导出 raw CSV"
  )
  STAGE_TOTAL=${#STAGE_NAMES[@]}
  STAGE_IDX=0
}

stage_start() {
  # $1 = log_file, $2 = 阶段名称（可与 STAGE_NAMES 对应）
  local _log="$1"; local _name="$2"
  STAGE_IDX=$((STAGE_IDX+1))
  local _cur=$STAGE_IDX
  local _tot=$STAGE_TOTAL
  echo "[TARGET ${CURRENT_TARGET}] [STEP ${_cur}/${_tot}] 开始：${_name}" | tee -a "${_log}"
}

stage_done() {
  # $1 = log_file, $2 = 阶段名称
  local _log="$1"; local _name="$2"
  local _cur=$STAGE_IDX
  local _tot=$STAGE_TOTAL
  local _remain=$((_tot - _cur))
  echo "[TARGET ${CURRENT_TARGET}] [STEP ${_cur}/${_tot}] 完成：${_name}；剩余 ${_remain} 步" | tee -a "${_log}"
  if (( _remain > 0 )); then
    local _next_name="${STAGE_NAMES[${_cur}]}"
    local _next_idx=$((_cur + 1))
    echo "[TARGET ${CURRENT_TARGET}] [NEXT ${_next_idx}/${_tot}] 即将执行：${_next_name}" | tee -a "${_log}"
  fi
}

stage_fail() {
  # $1 = log_file, $2 = 阶段名称
  local _log="$1"; local _name="$2"
  local _cur=$STAGE_IDX
  local _tot=$STAGE_TOTAL
  local _remain=$((_tot - _cur))
  echo "[TARGET ${CURRENT_TARGET}] [STEP ${_cur}/${_tot}] 失败：${_name}；剩余 ${_remain} 步（依据 CONTINUE_ON_ERROR=${CONTINUE_ON_ERROR} 决定是否继续）" | tee -a "${_log}"
}

# 解析 ENVS="A=1,B=2" => env 调用用的 "A=1" "B=2"
parse_envs_to_array() {
  local envs="$1"
  ENV_KV=()
  # 将空白去掉后若为空或为 '-'，表示不设置任何环境变量（避免传入 'env -' 清空 PATH）
  local compact="${envs//[[:space:]]/}"
  if [[ -z "${compact}" || "${compact}" == "-" ]]; then
    return
  fi
  IFS=',' read -r -a pairs <<< "${envs}"
  for kv in "${pairs[@]}"; do
    # trim kv 两端空白
    local _t="${kv}"
    _t="${_t#${_t%%[![:space:]]*}}"   # ltrim
    _t="${_t%${_t##*[![:space:]]}}"   # rtrim
    [[ -z "${_t}" || "${_t}" == "-" ]] && continue
    ENV_KV+=("${_t}")
  done
}

# 用 python 从 cuda_gpu_kern_sum.csv 选 TopN 并输出安全的 regex 片段
gen_kernel_regex_from_nsys_csv() {
  local csv="$1"
  local topn="$2"
  local out_regex_file="$3"
  "${PY}" - "${csv}" "${topn}" "${out_regex_file}" <<'PY'
import csv, sys, re, math
csv_path, topn_s, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
topn = int(topn_s)
rows = []
with open(csv_path, newline='', encoding='utf-8') as f:
    # nsys stats CSV, first line is header
    rdr = csv.DictReader(f)
    # Try to find time columns (nsys 有不同报告列名，尽量兼容)
    # Prefer columns with 'Total Time' or 'Time (ns)' in name
    # Fallback to first numeric-like column.
    time_cols = [c for c in rdr.fieldnames if 'Total Time' in c or 'Time (ns)' in c or c.lower().startswith('time')]
    for r in rdr:
        name = r.get('Name') or r.get('Kernel Name') or r.get('Kernel')
        if not name:
            # try any column containing 'Name'
            for k in r:
                if 'Name' in k:
                    name = r[k]; break
        if not name:
            continue
        # 选一个时间列
        tval = None
        for tc in time_cols:
            v = r.get(tc)
            if v is None: continue
            # 去掉逗号与空格
            v2 = v.replace(',', '').strip()
            try:
                tval = float(v2)
                break
            except:
                pass
        if tval is None:
            # 尝试所有数值列
            for k,v in r.items():
                if k == 'Name': continue
                if v is None: continue
                v2 = v.replace(',', '').strip()
                try:
                    tval = float(v2); break
                except:
                    pass
        if tval is None or math.isnan(tval):
            tval = 0.0
        rows.append((tval, name))

rows.sort(key=lambda x: x[0], reverse=True)
top = [r[1] for r in rows[:topn]]
# 生成 demangled 安全 regex
def escape_for_regex(s:str)->str:
    # ncu 的 --kernel-name-base demangled，Name 里包含 ()<> 等，统一转义
    return re.escape(s)

parts = [escape_for_regex(x) for x in top if x]
regex = "|".join(parts) if parts else ""
with open(out_path, "w", encoding="utf-8") as f:
    f.write(regex)
print(regex)
PY
}

##########################################
# Main loop over targets.tsv (tab sep)   #
##########################################

# 跳过空行与以#开头的注释
# 列顺序：NAME \t CMD \t WORKDIR \t ENVS（与 README 一致）
while IFS=$'\t' read -r NAME CMD WORKDIR ENVS; do
  [[ -z "${NAME:-}" ]] && continue
  [[ "${NAME}" =~ ^# ]] && continue
  # 跳过可能的表头（如：id / name）
  case "${NAME,,}" in
    id|name) continue ;;
  esac

  # 基本字段校验：CMD 不能为空；若为空，多半是使用了空格而非制表符分隔
  if [[ -z "${CMD:-}" ]]; then
    echo "[WARN] Malformed row in ${TARGETS_TSV} for target '${NAME}'. Ensure TAB-separated columns: NAME<TAB>CMD<TAB>WORKDIR<TAB>ENVS" >&2
    continue
  fi

  echo | tee -a "${GLOBAL_LOG}"
  echo "================== [$(ts)] Target: ${NAME} ==================" | tee -a "${GLOBAL_LOG}"

  TDIR="${OUT_DIR}/${NAME}"
  mkdir -p "${TDIR}"

  # 统一单日志：所有输出写入同一个全局日志文件
  mkdir -p "${TDIR}/${LOG_DIR}"  # 兼容旧结构（本运行不再写入子日志）
  NSYS_LOG="${GLOBAL_LOG}"
  NCU_LOG="${GLOBAL_LOG}"

  # NSYS OUTPUT
  NSYS_DIR="${TDIR}/nsys"
  mkdir -p "${NSYS_DIR}"
  NSYS_BASENAME="${NSYS_DIR}/${NAME}"
  NSYS_REP="${NSYS_BASENAME}.nsys-rep"
  NSYS_SQLITE="${NSYS_BASENAME}.sqlite"

  # NCU OUTPUT
  NCU_DIR="${TDIR}/ncu"
  mkdir -p "${NCU_DIR}"
  NCU_BASENAME="${NCU_DIR}/${NAME}"
  NCU_REP="${NCU_BASENAME}.ncu-rep"
  NCU_RAW_CSV="${NCU_BASENAME}_raw.csv"
  TOPK_REGEX_TXT="${NCU_DIR}/${NAME}_topk_regex.txt"

  # 初始化阶段并标记第 1 步（准备目录与上下文）
  init_stages
  CURRENT_TARGET="${NAME}"
  stage_start "${NSYS_LOG}" "${STAGE_NAMES[0]}"

  # 进入工作目录（WORKDIR 可留空或用 '-' 表示当前目录）
  WD="."
  if [[ -n "${WORKDIR:-}" && "${WORKDIR}" != "-" ]]; then
    WD="${WORKDIR}"
    [[ -d "${WD}" ]] || { echo "[WARN] WORKDIR not found: ${WD}" | tee -a "${NSYS_LOG}"; }
  fi

  # 解析 ENVS
  parse_envs_to_array "${ENVS:-}"

  ################################
  # 1) nsys profile + stats CSVs #
  ################################
  # 完成第 1 步
  stage_done "${NSYS_LOG}" "${STAGE_NAMES[0]}"

  # 第 2 步：nsys 采集
  stage_start "${NSYS_LOG}" "${STAGE_NAMES[1]}"
  echo "[*] ${NAME} :: nsys profile" | tee -a "${NSYS_LOG}"

  # 组装 nsys profile 选项
  NSYS_CMD=(env)
  if [[ ${#ENV_KV[@]} -gt 0 ]]; then
    NSYS_CMD+=("${ENV_KV[@]}")
  fi
  NSYS_CMD+=(
    nsys profile
    "--trace=${NSYS_TRACE}"
    "--force-overwrite=true"               # 覆盖旧输出（官方支持）
  )
  [[ -n "${NSYS_DURATION}" ]] && NSYS_CMD+=("--duration=${NSYS_DURATION}")
  # capture-range
  if [[ "${NSYS_CAPTURE_RANGE}" != "none" ]]; then
    NSYS_CMD+=("--capture-range=${NSYS_CAPTURE_RANGE}")
    # 只有在 capture-range!=none 的情况下才允许 capture-range-end（修复用户错误）
    NSYS_CMD+=("--capture-range-end=${NSYS_CAPTURE_RANGE_END}")
  fi
  [[ "${NSYS_QUIET}" == "true" ]] && NSYS_CMD+=("--quiet")
  NSYS_CMD+=(
    "-o" "${NSYS_BASENAME}"
    "--"                        # 官方建议在此分隔 nsys 选项与应用命令
    bash -lc "${CMD}"
  )

  # 执行（在 WORKDIR 中）
  ( cd "${WD}" && run_and_log "${NSYS_LOG}" "${NSYS_CMD[@]}" ) || {
    echo "[ERROR] nsys profile failed for ${NAME}" | tee -a "${NSYS_LOG}"
    stage_fail "${NSYS_LOG}" "${STAGE_NAMES[1]}"
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1
  }
  stage_done "${NSYS_LOG}" "${STAGE_NAMES[1]}"

  # nsys stats：生成 sqlite + CSV 报告（强制导出/覆盖）
  # 第 3 步：nsys 导出 CSV
  stage_start "${NSYS_LOG}" "${STAGE_NAMES[2]}"
  echo "[*] ${NAME} :: nsys stats -> CSV" | tee -a "${NSYS_LOG}"
  run_and_log "${NSYS_LOG}" nsys stats \
    --force-export=true --force-overwrite=true \
    --sqlite "${NSYS_SQLITE}" \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
    --format csv \
    --output "${NSYS_BASENAME}" \
    "${NSYS_REP}" || {
      echo "[ERROR] nsys stats failed for ${NAME}" | tee -a "${NSYS_LOG}"
      stage_fail "${NSYS_LOG}" "${STAGE_NAMES[2]}"
      [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1
    }
  stage_done "${NSYS_LOG}" "${STAGE_NAMES[2]}"

  KERN_CSV="${NSYS_BASENAME}_cuda_gpu_kern_sum.csv"
  # 第 4 步：生成 TopN kernel 正则
  stage_start "${NSYS_LOG}" "${STAGE_NAMES[3]}"
  if [[ ! -f "${KERN_CSV}" ]]; then
    echo "[WARN] Top kernel CSV not found: ${KERN_CSV}" | tee -a "${NSYS_LOG}"
  else
    echo "[*] ${NAME} :: pick Top-${NCU_TOPN} kernels from ${KERN_CSV}" | tee -a "${NSYS_LOG}"
    gen_kernel_regex_from_nsys_csv "${KERN_CSV}" "${NCU_TOPN}" "${TOPK_REGEX_TXT}" | tee -a "${NSYS_LOG}" || true
    check_csv_basic "${KERN_CSV}" "${NSYS_LOG}" "nsys kern_sum"
  fi
  stage_done "${NSYS_LOG}" "${STAGE_NAMES[3]}"

  #############################
  # 2) ncu on Top-N kernels   #
  #############################
  # 第 5 步：ncu 采集
  stage_start "${NCU_LOG}" "${STAGE_NAMES[4]}"
  echo "[*] ${NAME} :: ncu on Top${NCU_TOPN} kernels" | tee -a "${NCU_LOG}"

  KREGEX=""

  if [[ -s "${TOPK_REGEX_TXT}" ]]; then
    KREGEX="$(<"${TOPK_REGEX_TXT}")"
  else
    echo "[WARN] No kernel regex generated; will let ncu match first launches only." | tee -a "${NCU_LOG}"
  fi

  # 组装 ncu 命令（注意：不使用 '--' 分隔；直接把应用作为命令传给 ncu）
  NCU_CMD=(env)
  if [[ ${#ENV_KV[@]} -gt 0 ]]; then
    NCU_CMD+=("${ENV_KV[@]}")
  fi
  NCU_CMD+=(
    ncu
    -f                               # 覆盖输出（force-overwrite）
    "--target-processes" "all"       # 追踪子进程
    "--kernel-name-base" "demangled" # 与我们从 nsys 取到的 Name 一致
    "--launch-count" "${NCU_LAUNCH_COUNT}"
    "--set" "${NCU_SET}"
    "-o" "${NCU_REP}"
  )
  if [[ -n "${KREGEX}" ]]; then
    NCU_CMD+=("-k" "regex:${KREGEX}")
  else
    # 没有 regex 时：默认只采样前若干个 launch（由 --launch-count 控制）
    :
  fi
  # 把真正的应用命令作为 bash -lc "<CMD>"
  NCU_CMD+=( bash -lc "${CMD}" )

  ( cd "${WD}" && run_and_log "${NCU_LOG}" "${NCU_CMD[@]}" ) || {
    echo "[ERROR] ncu profiling failed for ${NAME}" | tee -a "${NCU_LOG}"
    stage_fail "${NCU_LOG}" "${STAGE_NAMES[4]}"
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1
  }
  stage_done "${NCU_LOG}" "${STAGE_NAMES[4]}"

  # 导出 ncu raw 页为 CSV（更易解析/拼表）
  # --import 与 --page raw --csv 配合：打印所有收集到的指标（含 section/metrics）
  # 第 6 步：ncu 导出 raw CSV
  stage_start "${NCU_LOG}" "${STAGE_NAMES[5]}"
  echo "[*] ${NAME} :: ncu export raw CSV" | tee -a "${NCU_LOG}"
  run_and_capture_stdout "${NCU_LOG}" "${NCU_RAW_CSV}" \
    ncu --import "${NCU_REP}" --page raw --csv || {
      echo "[ERROR] ncu export raw CSV failed for ${NAME}" | tee -a "${NCU_LOG}"
      stage_fail "${NCU_LOG}" "${STAGE_NAMES[5]}"
      [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1
    }
  check_csv_basic "${NCU_RAW_CSV}" "${NCU_LOG}" "ncu raw"
  stage_done "${NCU_LOG}" "${STAGE_NAMES[5]}"

  echo "[OK] ${NAME} outputs under: ${TDIR}"
  echo "     - NSYS: ${NSYS_REP}, ${NSYS_SQLITE}, ${KERN_CSV}"
  echo "     - NCU : ${NCU_REP}, ${NCU_RAW_CSV}"
done < "${TARGETS_TSV}"

echo | tee -a "${GLOBAL_LOG}"
echo "[DONE $(ts)] All targets processed. Outputs at: ${OUT_DIR}" | tee -a "${GLOBAL_LOG}"
