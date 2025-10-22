#!/usr/bin/env bash
# profile_and_report.sh
# Robust, faster-by-default pipeline for Nsight Systems (nsys) + Nsight Compute (ncu)
# - Reads a TSV (targets.tsv): columns = NAME  WORKDIR  ENVS  CMD
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

TARGETS_TSV="${TARGETS_TSV:-./targets.tsv}"     # 输入任务表（TSV）
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
if [[ "${NCU_FAST}" == "1" ]]; then
  NCU_SECTIONS="${NCU_SECTIONS:-LaunchStats,SpeedOfLight,WarpStateStats}"
else
  NCU_SECTIONS="${NCU_SECTIONS:-LaunchStats,SpeedOfLight,WarpStateStats,MemoryWorkloadAnalysis}"
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

# 解析 ENVS="A=1,B=2" => env 调用用的 "A=1" "B=2"
parse_envs_to_array() {
  local envs="$1"
  ENV_KV=()
  if [[ -n "${envs}" ]]; then
    # 以逗号分割，不 trim 空格（交给 bash -lc）
    IFS=',' read -r -a pairs <<< "${envs}"
    for kv in "${pairs[@]}"; do
      [[ -z "${kv}" ]] && continue
      ENV_KV+=("${kv}")
    done
  fi
}

# 用 python 从 cuda_gpu_kern_sum.csv 选 TopN 并输出安全的 regex 片段
gen_kernel_regex_from_nsys_csv() {
  local csv="$1"
  local topn="$2"
  local out_regex_file="$3"
  "${PY}" - <<'PY' -- "${csv}" "${topn}" "${out_regex_file}"
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
# 列顺序：NAME \t WORKDIR \t ENVS \t CMD
while IFS=$'\t' read -r NAME WORKDIR ENVS CMD; do
  [[ -z "${NAME:-}" ]] && continue
  [[ "${NAME}" =~ ^# ]] && continue

  echo
  echo "================== [$(ts)] Target: ${NAME} =================="

  TDIR="${OUT_DIR}/${NAME}"
  mkdir -p "${TDIR}"

  # per-target logs
  mkdir -p "${TDIR}/${LOG_DIR}"
  NSYS_LOG="${TDIR}/${LOG_DIR}/${NAME}_nsys.log"
  NCU_LOG="${TDIR}/${LOG_DIR}/${NAME}_ncu.log"

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

  # 进入工作目录
  if [[ -n "${WORKDIR}" ]]; then
    [[ -d "${WORKDIR}" ]] || { echo "[WARN] WORKDIR not found: ${WORKDIR}" | tee -a "${NSYS_LOG}"; }
  fi
  WD="${WORKDIR:-.}"

  # 解析 ENVS
  parse_envs_to_array "${ENVS:-}"

  ################################
  # 1) nsys profile + stats CSVs #
  ################################
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
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1
  }

  # nsys stats：生成 sqlite + CSV 报告（强制导出/覆盖）
  echo "[*] ${NAME} :: nsys stats -> CSV" | tee -a "${NSYS_LOG}"
  run_and_log "${NSYS_LOG}" nsys stats \
    --force-export=true --force-overwrite=true \
    --sqlite "${NSYS_SQLITE}" \
    --report cuda_gpu_kern_sum,cuda_gpu_mem_size_sum,cuda_api_sum \
    --format csv \
    --output "${NSYS_BASENAME}" \
    "${NSYS_REP}" || {
      echo "[ERROR] nsys stats failed for ${NAME}" | tee -a "${NSYS_LOG}"
      [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1
    }

  KERN_CSV="${NSYS_BASENAME}_cuda_gpu_kern_sum.csv"
  if [[ ! -f "${KERN_CSV}" ]]; then
    echo "[WARN] Top kernel CSV not found: ${KERN_CSV}" | tee -a "${NSYS_LOG}"
  else
    echo "[*] ${NAME} :: pick Top-${NCU_TOPN} kernels from ${KERN_CSV}" | tee -a "${NSYS_LOG}"
    gen_kernel_regex_from_nsys_csv "${KERN_CSV}" "${NCU_TOPN}" "${TOPK_REGEX_TXT}" | tee -a "${NSYS_LOG}" || true
  fi

  #############################
  # 2) ncu on Top-N kernels   #
  #############################
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
    "--section" "${NCU_SECTIONS}"
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
    [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1
  }

  # 导出 ncu raw 页为 CSV（更易解析/拼表）
  # --import 与 --page raw --csv 配合：打印所有收集到的指标（含 section/metrics）
  echo "[*] ${NAME} :: ncu export raw CSV" | tee -a "${NCU_LOG}"
  run_and_log "${NCU_LOG}" ncu \
    --import "${NCU_REP}" \
    --page raw --csv \
    > "${NCU_RAW_CSV}" || {
      echo "[ERROR] ncu export raw CSV failed for ${NAME}" | tee -a "${NCU_LOG}"
      [[ "${CONTINUE_ON_ERROR}" == "1" ]] || exit 1
    }

  echo "[OK] ${NAME} outputs under: ${TDIR}"
  echo "     - NSYS: ${NSYS_REP}, ${NSYS_SQLITE}, ${KERN_CSV}"
  echo "     - NCU : ${NCU_REP}, ${NCU_RAW_CSV}"
done < "${TARGETS_TSV}"

echo
echo "[DONE $(ts)] All targets processed. Outputs at: ${OUT_DIR}"
