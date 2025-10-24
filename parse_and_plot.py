#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, pathlib, re, logging, numpy as np, pandas as pd, matplotlib.pyplot as plt
from utils import read_csv_safe, pick_value, to_metric_unit_value, extract_rule_value

# Try to configure a font that supports Chinese labels to avoid glyph warnings.
try:
    import matplotlib
    matplotlib.rcParams['axes.unicode_minus'] = False
    # Try to register any local fonts shipped in repo (scripts/bash_metric/fonts/*.ttf|*.otf)
    try:
        from matplotlib import font_manager as _fm
        _here = pathlib.Path(__file__).resolve().parent
        _fonts_dir = _here / "fonts"
        _registered = []
        if _fonts_dir.exists():
            for p in sorted(_fonts_dir.glob("*.ttf")) + sorted(_fonts_dir.glob("*.otf")):
                try:
                    _fm.fontManager.addfont(str(p))
                    # Use the real family name if we can resolve it
                    try:
                        fam = _fm.FontProperties(fname=str(p)).get_name()
                    except Exception:
                        fam = None
                    if fam:
                        _registered.append(fam)
                except Exception:
                    pass
        # If we successfully registered any family names, prefer them explicitly
        if _registered:
            # de-duplicate while preserving order
            seen = set()
            ordered = []
            for f in _registered + [
                'Noto Sans SC', 'Noto Sans CJK SC', 'Source Han Sans SC',
                'WenQuanYi Micro Hei', 'SimHei', 'Microsoft YaHei', 'PingFang SC', 'DejaVu Sans']:
                if f and f not in seen:
                    seen.add(f); ordered.append(f)
            # Prefer the first registered family directly as default, but
            # if LXGW WenKai is present, use it for better CJK coverage.
            preferred = None
            for candidate in (['LXGW WenKai'] + ordered):
                if candidate in ordered:
                    preferred = candidate
                    break
            matplotlib.rcParams['font.family'] = [preferred or ordered[0]]
            matplotlib.rcParams['font.sans-serif'] = ordered
            _HAS_LOCAL_CJK = True
        else:
            _HAS_LOCAL_CJK = False
    except Exception:
        pass
    # Prefer using mplfonts if available (it downloads and registers CJK fonts)
    try:
        if not globals().get('_HAS_LOCAL_CJK', False):
            import mplfonts
            # Try to activate a bundled CJK font if available
            try:
                mplfonts.use_font('Noto Sans CJK SC')
            except Exception:
                try:
                    mplfonts.use_font('Source Han Sans SC')
                except Exception:
                    pass
    except Exception:
        # Fallback to a list of common CJK-capable families if present
        for fam in [
            'Noto Sans SC', 'Noto Sans CJK SC', 'Source Han Sans SC', 'WenQuanYi Micro Hei',
            'SimHei', 'Microsoft YaHei', 'PingFang SC', 'DejaVu Sans'
        ]:
            try:
                matplotlib.rcParams['font.family'] = ['sans-serif']
                matplotlib.rcParams['font.sans-serif'] = [fam]
                break
            except Exception:
                continue
except Exception:
    pass

# === Heuristic thresholds ===
HEURISTIC = dict(
    SM_PEAK_HI = 60.0,
    DRAM_PEAK_HI = 60.0,
    SM_PEAK_LO = 40.0,
    DRAM_PEAK_LO = 40.0,
)

METRICS_CAND = {
  # SM % peak: include elapsed/active variants and smsp scope
  "sm_pct_peak": [
      r"sm__throughput\.avg\.pct_of_peak_sustained_elapsed",
      r"sm__throughput\.avg\.pct_of_peak_sustained_active",
      r"smsp__throughput\.avg\.pct_of_peak_sustained_active",
  ],
  # DRAM % peak: legacy gpu__dram_* and dram__throughput variants
  "dram_pct_peak": [
      r"gpu__dram_throughput\.avg\.pct_of_peak_sustained_elapsed",
      r"dram__throughput\.avg\.pct_of_peak_sustained_elapsed",
      r"dram__throughput\.avg\.pct_of_peak_sustained_active",
  ],
  # Tensor Core activity: legacy pipe cycles + newer HMMA pipe utilisation
  "tensor_active_pct": [
      r"sm__pipe_tensor_cycles_active\.avg\.pct_of_peak_sustained_active",
      r"sm__inst_executed_pipe_tensor_op_hmma\.avg\.pct_of_peak_sustained_active",
      r"smsp__inst_executed_pipe_tensor_op_hmma\.avg\.pct_of_peak_sustained_active",
  ],
  # FP32 efficiency/activity: legacy derived name + newer fp32 pipe utilisation
  "fp32_eff_pct": [
      r"flop_sp_efficiency",
      r"sm__inst_executed_pipe_fp32\.avg\.pct_of_peak_sustained_active",
      r"smsp__inst_executed_pipe_fp32\.avg\.pct_of_peak_sustained_active",
  ],
  # Cache hit rate: include sector and avg hit rate variants
  "l1_hit_pct": [
      r"l1tex__t_sector_hit_rate\.pct",
      r"l1tex__avg_hit_rate\.pct",
  ],
  "l2_hit_pct": [
      r"lts__t_sector_hit_rate\.pct",
      r"lts__avg_hit_rate\.pct",
  ],
  # Warp stall (old and new metric names)
  "stall_barrier": [
      r"smsp__warp_issue_stalled_barrier_per_warp_active\.avg",
      r"smsp__average_warps_issue_stalled_barrier_per_issue_active\.ratio",
  ],
  "stall_short_sb": [
      r"smsp__warp_issue_stalled_short_scoreboard_per_warp_active\.avg",
      r"smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active\.ratio",
  ],
  "stall_long_sb": [
      r"smsp__warp_issue_stalled_long_scoreboard_per_warp_active\.avg",
      r"smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active\.ratio",
  ],
}

def ge(x, t):  # x >= t
    import numpy as np
    return (not np.isnan(x)) and (x >= t)

def lt(x, t):  # x < t
    import numpy as np
    return (not np.isnan(x)) and (x < t)

def classify_bottleneck(sm_pct, dram_pct, l1_pct, l2_pct):
    if ge(sm_pct, HEURISTIC["SM_PEAK_HI"]) and lt(dram_pct, HEURISTIC["DRAM_PEAK_LO"]):
        return "Compute Bound"
    if ge(dram_pct, HEURISTIC["DRAM_PEAK_HI"]) and lt(sm_pct, HEURISTIC["SM_PEAK_LO"]):
        return "Memory Bound"
    if (lt(l1_pct, 50) or lt(l2_pct, 60)):
        return "Memory Bound / Latency"
    if (np.isnan(sm_pct) and np.isnan(dram_pct)):
        return "Unknown"
    return "Mixed"


def parse_nsys_kern_sum(path):
    df = read_csv_safe(path)
    if df.empty: return df
    cols = {c.lower(): c for c in df.columns}
    name = next((cols[k] for k in cols if 'name' in k), None)
    time_pct = next((cols[k] for k in cols if 'time' in k and '%' in k), None)
    time_total = next((cols[k] for k in cols if 'total' in k and 'time' in k and 'ns' in k), None)
    if not name: return pd.DataFrame()
    out = pd.DataFrame({
        "kernel": df[name].astype(str),
        "kernel_time_pct": pd.to_numeric(df[time_pct], errors='coerce') if time_pct else np.nan,
        "kernel_total_time_ns": pd.to_numeric(df[time_total], errors='coerce') if time_total else np.nan,
    })
    return out

def parse_ncu_csv_pair(raw_csv, id_csv):
    df_id = to_metric_unit_value(read_csv_safe(id_csv))
    df_raw = to_metric_unit_value(read_csv_safe(raw_csv))
    df_raw_full = read_csv_safe(raw_csv)
    vals={}
    # Human-readable label fallbacks for newer Nsight versions
    LABEL_FALLBACKS = {
        "sm_pct_peak": [
            "SM % of Peak Sustained Elapsed",  # older
            "Compute (SM) Throughput",         # newer human-readable
            "Compute Throughput",
        ],
        "dram_pct_peak": [
            "DRAM % of Peak Sustained Elapsed",  # older
            "DRAM Throughput",                    # newer human-readable
        ],
        "tensor_active_pct": [
            "Tensor Pipe Active",
            "Tensor Core Active",
            "Tensor Pipe Active (%)",
        ],
        "fp32_eff_pct": [
            "FP32 (SP) FLOP Efficiency",
            "FLOP_SP_EFFICIENCY",
            "FP32 Pipe Active",
            "FP32 Pipe Active (%)",
        ],
        "l1_hit_pct": [
            "L1/TEX Hit Rate",
            "L1 Hit Rate",
        ],
        "l2_hit_pct": [
            "L2 Hit Rate",
        ],
        "stall_barrier": [
            r"Warp.*Stall.*Barrier.*Warp.*Active",
            r"Barrier.*Warp.*Stall",
            "Warp Issue Stalled Barrier / Warp Active",
        ],
        "stall_short_sb": [
            r"Warp.*Stall.*Short.*Scoreboard.*Warp.*Active",
            r"Short.*Scoreboard.*Warp.*Stall",
            "Warp Issue Stalled Short Scoreboard / Warp Active",
        ],
        "stall_long_sb": [
            r"Warp.*Stall.*Long.*Scoreboard.*Warp.*Active",
            r"Long.*Scoreboard.*Warp.*Stall",
            "Warp Issue Stalled Long Scoreboard / Warp Active",
        ],
    }
    def find_by_labels(df, labels):
        if df is None or df.empty or not labels:
            return None
        # Prefer literal matching to avoid regex pitfalls with special chars like parentheses
        metrics = df['metric'].astype(str)
        for lb in labels:
            try:
                # 1) exact, case-insensitive match
                sub = df[metrics.str.strip().str.casefold() == str(lb).strip().casefold()]
                if not sub.empty:
                    v = pd.to_numeric(str(sub.iloc[0]['value']).replace('%',''), errors='coerce')
                    return v
                # 2) substring contains, case-insensitive, non-regex
                sub = df[metrics.str.contains(str(lb), regex=False, case=False)]
                if not sub.empty:
                    v = pd.to_numeric(str(sub.iloc[0]['value']).replace('%',''), errors='coerce')
                    return v
                # 3) as last resort, treat label as regex (for patterns like Warp.*Stall...)
                sub = df[metrics.str.contains(str(lb), regex=True)]
                if not sub.empty:
                    v = pd.to_numeric(str(sub.iloc[0]['value']).replace('%',''), errors='coerce')
                    return v
            except Exception:
                continue
        return None
    for k, cands in METRICS_CAND.items():
        # First try metrics-id CSV
        v,u,m = pick_value(df_id, cands)
        if not (isinstance(v,float) and np.isnan(v)):
            vals[k]=v
            continue
        # Fall back to raw CSV where newer Nsight versions may expose renamed metrics
        v_raw,u_raw,m_raw = pick_value(df_raw, cands)
        if not (isinstance(v_raw,float) and np.isnan(v_raw)):
            vals[k]=v_raw
            continue
        # Try label fallbacks: prefer id_csv (metrics-id), then raw
        labels = LABEL_FALLBACKS.get(k, [])
        v2 = find_by_labels(df_id, labels)
        if v2 is None:
            v2 = find_by_labels(df_raw, labels)
        if v2 is not None:
            vals[k] = v2
            continue
        # As an additional fallback for newer Nsight raw page (wide columns), try matching column headers
        if df_raw_full is not None and not df_raw_full.empty:
            try:
                import re
                for cand in cands:
                    # find first column whose name matches the candidate regex
                    cols = [c for c in df_raw_full.columns if re.search(cand, str(c))]
                    if cols:
                        col = cols[0]
                        ser = pd.to_numeric(df_raw_full[col], errors='coerce')
                        ser = ser.dropna()
                        if not ser.empty:
                            vals[k] = float(ser.iloc[0])
                            break
                if k in vals:
                    continue
            except Exception:
                pass
        # Generic wide-CSV melting heuristics for common suffix patterns
        if df_raw_full is not None and not df_raw_full.empty:
            try:
                # pick any column likely expressing peak percentage or hit rate
                patt = r"(pct_of_peak_sustained_(elapsed|active)|_hit_rate\.pct)"
                cols = [c for c in df_raw_full.columns if re.search(patt, str(c))]
                # if still empty, keep going to next key
                if cols:
                    for c in cols:
                        ser = pd.to_numeric(df_raw_full[c], errors='coerce').dropna()
                        if not ser.empty:
                            vals.setdefault(k, float(ser.iloc[0]))
                            break
                if k in vals:
                    continue
            except Exception:
                pass
        # As last resort, try extracting rule-based values from metrics-id (e.g., CPIStall % under WarpStateStats)
        if k == 'stall_barrier':
            v3 = extract_rule_value(df_id, 'BarrierStall')
            if not (isinstance(v3, float) and np.isnan(v3)):
                vals[k] = v3; continue
        if k == 'stall_short_sb':
            v3 = extract_rule_value(df_id, 'ShortScoreboardStall')
            if not (isinstance(v3, float) and np.isnan(v3)):
                vals[k] = v3; continue
        if k == 'stall_long_sb':
            v3 = extract_rule_value(df_id, 'LongScoreboardStall')
            if not (isinstance(v3, float) and np.isnan(v3)):
                vals[k] = v3; continue
        if k not in vals:
            vals[k]=np.nan
    vals["bottleneck"] = classify_bottleneck(vals["sm_pct_peak"], vals["dram_pct_peak"], vals["l1_hit_pct"], vals["l2_hit_pct"])
    return vals

def _setup_id_logger(id_dir: pathlib.Path, base_log: pathlib.Path|None = None):
    """Create a per-id logger that writes to out/<id>/log/parse_and_plot.log.
    If base_log is provided, also attach a file handler appending to that path,
    and a stream handler to mirror logs to stdout.
    """
    log_dir = id_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "parse_and_plot.log"
    logger = logging.getLogger(f"parse_and_plot.{id_dir.name}")
    logger.setLevel(logging.INFO)
    # Avoid duplicating handlers on repeated runs in the same interpreter
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(name)s - %(message)s')
    # default per-id file
    fh_id = logging.FileHandler(str(log_path), mode='w', encoding='utf-8')
    fh_id.setFormatter(fmt)
    logger.addHandler(fh_id)
    # optional base log appender (append mode)
    if base_log is not None:
        fh_base = logging.FileHandler(str(base_log), mode='a', encoding='utf-8')
        fh_base.setFormatter(fmt)
        logger.addHandler(fh_base)
    # also mirror to stdout
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.propagate = False
    return logger

def main(out_root, base_log: str|None = None):
    out_root = pathlib.Path(out_root)
    rows=[]
    for id_dir in sorted(out_root.glob("*")):
        if not (id_dir.is_dir() and (id_dir/"nsys").exists() and (id_dir/"ncu").exists()):
            continue
        logger = _setup_id_logger(id_dir, pathlib.Path(base_log) if base_log else None)
        logger.info("Begin summarize for id=%s", id_dir.name)
        nsys_csv = next((id_dir/"nsys").glob("*_cuda_gpu_kern_sum.csv"), None)
        if nsys_csv:
            logger.info("Found nsys kernel summary: %s", nsys_csv)
        else:
            logger.warning("nsys kernel summary CSV not found under %s", id_dir/"nsys")
        kern_df = parse_nsys_kern_sum(nsys_csv) if nsys_csv else pd.DataFrame()
        # Build a mapping from sanitized kernel name (as used in NCU filenames) to timing fields
        def _sanitize_base(s: str) -> str:
            return re.sub(r'[^A-Za-z0-9._-]+', '_', str(s))[:120]
        ktime_map = {}
        if not kern_df.empty:
            for _, row in kern_df.iterrows():
                k = str(row.get('kernel',''))
                if not k:
                    continue
                ktime_map[_sanitize_base(k)] = (
                    k,
                    float(row.get('kernel_time_pct')) if 'kernel_time_pct' in kern_df.columns else float('nan'),
                    float(row.get('kernel_total_time_ns')) if 'kernel_total_time_ns' in kern_df.columns else float('nan'),
                )
        # 兼容两种导出命名：*_ncu_raw.csv（Python 路线）与 *_raw.csv（Bash 方法脚本）
        raw_list = list((id_dir/"ncu").glob("*_ncu_raw.csv")) + list((id_dir/"ncu").glob("*_raw.csv"))
        seen = set()
        logger.info("Found %d ncu raw CSV files", len(raw_list))
        for ncu_raw in sorted(raw_list):
            if ncu_raw in seen:
                continue
            seen.add(ncu_raw)
            name = ncu_raw.name
            if name.endswith("_ncu_raw.csv"):
                id_csv = ncu_raw.with_name(name.replace("_ncu_raw.csv", "_metrics_id.csv"))
            elif name.endswith("_raw.csv"):
                id_csv = ncu_raw.with_name(name.replace("_raw.csv", "_metrics_id.csv"))
            else:
                id_csv = ncu_raw.with_suffix(".metrics_id.csv")
            file_base = name.replace("_ncu_raw.csv","").replace("_raw.csv","")
            # Skip top-level id capture like '<id>_raw.csv' which is not a specific kernel
            if file_base == id_dir.name:
                logger.info("Skip non-kernel raw CSV: %s", ncu_raw)
                continue
            # Determine display kernel name via mapping; fallback to file_base heuristic
            mapped = ktime_map.get(file_base)
            if mapped:
                disp_kernel, kernel_time_pct, kernel_total_time_ns = mapped
            else:
                disp_kernel = file_base.replace("____"," ")
                kernel_time_pct = float('nan')
                kernel_total_time_ns = float('nan')
            logger.info("Process ncu raw CSV: %s (derived id CSV: %s, kernel name: %s)", ncu_raw, id_csv, disp_kernel)
            vals = parse_ncu_csv_pair(ncu_raw, id_csv)
            logger.info("Parsed metrics: sm%%=%.2f dram%%=%.2f l1%%=%.2f l2%%=%.2f time%%=%.2f total_ns=%s",
                        vals.get('sm_pct_peak', float('nan')) if vals else float('nan'),
                        vals.get('dram_pct_peak', float('nan')) if vals else float('nan'),
                        vals.get('l1_hit_pct', float('nan')) if vals else float('nan'),
                        vals.get('l2_hit_pct', float('nan')) if vals else float('nan'),
                        kernel_time_pct if isinstance(kernel_time_pct, (int,float)) else float('nan'),
                        str(kernel_total_time_ns))
            # Drop rows with all-Nan core metrics AND no time info to reduce noise
            core_vals = [vals.get('sm_pct_peak'), vals.get('dram_pct_peak'), vals.get('l1_hit_pct'), vals.get('l2_hit_pct')]
            if all((v is None) or (isinstance(v,float) and np.isnan(v)) for v in core_vals) and (np.isnan(kernel_time_pct) and np.isnan(kernel_total_time_ns)):
                logger.info("Skip empty metrics row for kernel=%s", disp_kernel)
                continue
            rows.append(dict(
                id=id_dir.name, kernel=disp_kernel, **vals,
                kernel_time_pct=kernel_time_pct, kernel_total_time_ns=kernel_total_time_ns
            ))
        logger.info("Finish summarize for id=%s; appended %d kernel rows", id_dir.name, len([r for r in rows if r['id']==id_dir.name]))
    if not rows:
        print("No rows to summarize."); return
    df = pd.DataFrame(rows)

    # Long-form table
    long_rows=[]
    def add_row(idv,kernel,cate,detail,metric_key,value,unit,cmd_metric):
        long_rows.append(dict(类别=cate, 分析详情=detail, 具体指标=metric_key, 值=value, 单位=unit, 命令或计数器=cmd_metric, id=idv, kernel=kernel))

    for _, r in df.iterrows():
        idv, kernel = r['id'], r['kernel']
        add_row(idv,kernel,"算力单元利用率","SM 利用率(峰值%) / Compute Throughput","sm__throughput.avg.pct_of_peak_sustained_elapsed", r['sm_pct_peak'],"%", "ncu metrics")
        add_row(idv,kernel,"算力单元利用率","Tensor Core 活跃(%) / Tensor Pipe Active","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", r['tensor_active_pct'],"%", "ncu metrics")
        add_row(idv,kernel,"算力单元利用率","FP32 效率(%) / FP32 FLOP Efficiency","flop_sp_efficiency", r['fp32_eff_pct'],"%", "ncu metrics")
        add_row(idv,kernel,"Cache 命中率","L1/TEX Sector 命中率(%) / L1/TEX Hit Rate","l1tex__t_sector_hit_rate.pct", r['l1_hit_pct'],"%", "ncu metrics")
        add_row(idv,kernel,"Cache 命中率","L2 Sector 命中率(%) / L2 Hit Rate","lts__t_sector_hit_rate.pct", r['l2_hit_pct'],"%", "ncu metrics")
        add_row(idv,kernel,"内存带宽","DRAM 吞吐(峰值%) / DRAM Throughput","gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", r['dram_pct_peak'],"%", "ncu metrics")
        add_row(idv,kernel,"Warp 效率","Barrier Stall / Warp Active / 栅栏停顿(每活跃Warp)","smsp__warp_issue_stalled_barrier_per_warp_active.avg", r['stall_barrier'],"", "ncu metrics")
        add_row(idv,kernel,"Warp 效率","Short Scoreboard Stall / Warp Active / 短记分牌停顿(每活跃Warp)","smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg", r['stall_short_sb'],"", "ncu metrics")
        add_row(idv,kernel,"Warp 效率","Long Scoreboard Stall / Warp Active / 长记分牌停顿(每活跃Warp)","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.avg", r['stall_long_sb'],"", "ncu metrics")
        add_row(idv,kernel,"Kernel 耗时","时间占比(%) / Time(%)","Time(%)", r['kernel_time_pct'],"%", "nsys stats: cuda_gpu_kern_sum")
        add_row(idv,kernel,"Kernel 耗时","总耗时(ns) / Total Time (ns)","Total Time (ns)", r['kernel_total_time_ns'],"ns", "nsys stats: cuda_gpu_kern_sum")
        add_row(idv,kernel,"整体推理瓶颈","启发式(基于SM/DRAM/Cache)","bottleneck", r['bottleneck'],"", "heuristic")
    long_df = pd.DataFrame(long_rows)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    chart_dir = out_root / "figs"
    chart_dir.mkdir(exist_ok=True, parents=True)
    csv_path = chart_dir / f"summary_{ts}.csv"
    md_path  = chart_dir / f"summary_{ts}.md"
    long_df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 汇总表（关键指标）\n\n")
        for (idv, kernel), g in long_df.groupby(["id","kernel"]):
            f.write(f"## {idv} :: {kernel}\n\n")
            # 展示所有类别，保持与 CSV 一致；并进行稳定排序以便对齐
            sub = g[["类别","分析详情","具体指标","值","单位","命令或计数器"]]
            sub = sub.sort_values(["类别","分析详情","具体指标"], kind="stable")
            f.write(sub.fillna("nan").to_markdown(index=False))
            f.write("\n\n")

    figs_dir = out_root / "figs"
    figs_dir.mkdir(exist_ok=True, parents=True)

    # Top10 kernel by Time(%)
    def _auto_figsize_and_margins(labels, orient='y'):
        labels = [str(x) for x in (labels or [])]
        max_len = max((len(s) for s in labels), default=0)
        n = len(labels)
        if orient == 'y':
            width = max(12, min(30, 0.12 * max_len + 8))
            height = max(4, min(20, 0.5 * n + 2))
            margins = dict(left=min(0.6, max(0.18, 0.009 * max_len)), right=0.98, top=0.90, bottom=0.12)
        else:
            width = max(14, min(30, 0.12 * max_len + 8))
            height = max(5, min(22, 0.55 * n + 3))
            margins = dict(left=0.12, right=0.98, top=0.90, bottom=min(0.6, max(0.18, 0.010 * max_len)))
        return width, height, margins
    for idv, g in df.groupby("id"):
        g2 = g.sort_values("kernel_time_pct", ascending=False).head(10)
        if not g2.empty:
            labels = list(g2["kernel"].astype(str))
            fw, fh, mg = _auto_figsize_and_margins(labels, orient='y')
            fig, ax = plt.subplots(figsize=(fw, fh))
            ax.set_title(f"{idv} Top10 Kernel Time(%) / 按耗时Top10")
            ax.barh(labels, g2["kernel_time_pct"], color="#4C78A8")
            ax.invert_yaxis()
            ax.set_xlabel("Time (%) / 时间占比(%)")
            ax.tick_params(axis='y', labelsize=8)
            fig.subplots_adjust(**mg)
            plt.savefig(figs_dir / f"{idv}_topk_time.png")
            plt.close()

            labels = list(g2["kernel"].astype(str))
            fw, fh, mg = _auto_figsize_and_margins(labels, orient='x')
            fig, ax = plt.subplots(figsize=(fw, fh))
            ax.set_title(f"{idv} SM/DRAM % of Peak (Top10 by Time) / SM/DRAM 峰值占比（按耗时Top10）")
            a = g2[["kernel","sm_pct_peak","dram_pct_peak"]].set_index("kernel")
            # 设置更友好的图例（中英对照）
            a.columns = [
                "SM 利用率(峰值%) / Compute Throughput",
                "DRAM 吞吐(峰值%) / DRAM Throughput",
            ]
            a.plot(kind="bar", ax=ax, color=["#72B7B2", "#E45756"])
            ax.set_ylabel("%")
            ax.tick_params(axis='x', labelrotation=35, labelsize=8)
            fig.subplots_adjust(**mg)
            plt.savefig(figs_dir / f"{idv}_sm_dram_peak.png")
            plt.close()

    # 写入日志到每个 id 的日志目录
    for idv, g in df.groupby("id"):
        logger = _setup_id_logger(out_root / idv)
        logger.info("Wrote summary CSV: %s", csv_path)
        logger.info("Wrote summary MD : %s", md_path)
        logger.info("Figures dir      : %s", figs_dir)
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Figures: {figs_dir}")
if __name__=="__main__":
    # CLI: parse_and_plot.py <out_root> [--log <path>]
    out_root = None
    base_log = None
    args = sys.argv[1:]
    if not args:
        out_root = "./out"
    else:
        out_root = args[0]
        if len(args) > 2 and args[1] == "--log":
            base_log = args[2]
    main(out_root, base_log)
