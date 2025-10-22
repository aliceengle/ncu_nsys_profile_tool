#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, pathlib, re, numpy as np, pandas as pd, matplotlib.pyplot as plt
from utils import read_csv_safe, pick_value

# === Heuristic thresholds ===
HEURISTIC = dict(
    SM_PEAK_HI = 60.0,
    DRAM_PEAK_HI = 60.0,
    SM_PEAK_LO = 40.0,
    DRAM_PEAK_LO = 40.0,
)

METRICS_CAND = {
  "sm_pct_peak": [r"sm__throughput\.avg\.pct_of_peak_sustained_elapsed"],
  "dram_pct_peak": [r"gpu__dram_throughput\.avg\.pct_of_peak_sustained_elapsed"],
  "tensor_active_pct": [r"sm__pipe_tensor_cycles_active\.avg\.pct_of_peak_sustained_active"],
  "fp32_eff_pct": [r"flop_sp_efficiency"],
  "l1_hit_pct": [r"l1tex__t_sector_hit_rate\.pct"],
  "l2_hit_pct": [r"lts__t_sector_hit_rate\.pct"],
  "stall_barrier": [r"smsp__warp_issue_stalled_barrier_per_warp_active\.avg"],
  "stall_short_sb": [r"smsp__warp_issue_stalled_short_scoreboard_per_warp_active\.avg"],
  "stall_long_sb": [r"smsp__warp_issue_stalled_long_scoreboard_per_warp_active\.avg"],
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
    df_id = read_csv_safe(id_csv)
    if not df_id.empty:
        cols = [c.strip().lower() for c in df_id.columns]
        df_id.columns = ['metric','unit','value'] + list(range(len(cols)-3))
    df_raw = read_csv_safe(raw_csv)
    if not df_raw.empty:
        cols = [c.strip().lower() for c in df_raw.columns]
        df_raw.columns = ['metric','unit','value'] + list(range(len(cols)-3))
    vals={}
    for k, cands in METRICS_CAND.items():
        v,u,m = pick_value(df_id, cands)
        if not (isinstance(v,float) and np.isnan(v)):
            vals[k]=v
            continue
        # Try Raw labels fallback
        labels = {
            "sm_pct_peak": ["SM % of Peak Sustained Elapsed"],
            "dram_pct_peak": ["DRAM % of Peak Sustained Elapsed"],
            "tensor_active_pct": ["Tensor Pipe Active"],
            "fp32_eff_pct": ["FP32 (SP) FLOP Efficiency","FLOP_SP_EFFICIENCY"],
            "l1_hit_pct": ["L1/TEX Hit Rate","L1 Hit Rate"],
            "l2_hit_pct": ["L2 Hit Rate"],
            "stall_barrier": ["Warp Issue Stalled Barrier / Warp Active"],
            "stall_short_sb": ["Warp Issue Stalled Short Scoreboard / Warp Active"],
            "stall_long_sb": ["Warp Issue Stalled Long Scoreboard / Warp Active"],
        }.get(k, [])
        if not df_raw.empty and labels:
            for lb in labels:
                sub = df_raw[df_raw['metric'].astype(str).str.contains(lb, regex=False)]
                if not sub.empty:
                    v = pd.to_numeric(str(sub.iloc[0]['value']).replace('%',''), errors='coerce')
                    vals[k]=v; break
        if k not in vals:
            vals[k]=np.nan
    vals["bottleneck"] = classify_bottleneck(vals["sm_pct_peak"], vals["dram_pct_peak"], vals["l1_hit_pct"], vals["l2_hit_pct"])
    return vals

def main(out_root):
    out_root = pathlib.Path(out_root)
    rows=[]
    for id_dir in sorted(out_root.glob("*")):
        if not (id_dir.is_dir() and (id_dir/"nsys").exists() and (id_dir/"ncu").exists()):
            continue
        nsys_csv = next((id_dir/"nsys").glob("*_cuda_gpu_kern_sum.csv"), None)
        kern_df = parse_nsys_kern_sum(nsys_csv) if nsys_csv else pd.DataFrame()
        for ncu_raw in sorted((id_dir/"ncu").glob("*_ncu_raw.csv")):
            id_csv = pathlib.Path(str(ncu_raw).replace("_ncu_raw.csv","_metrics_id.csv"))
            kname = ncu_raw.name.replace("_ncu_raw.csv","").replace("____"," ")
            vals = parse_ncu_csv_pair(ncu_raw, id_csv)
            kernel_time_pct = kernel_total_ns = np.nan
            if not kern_df.empty:
                hit = kern_df[kern_df['kernel'].astype(str).str.contains(re.escape(kname), regex=True)]
                if hit.empty:
                    hit = kern_df[kern_df['kernel'].astype(str).str.contains(re.escape(kname).split('(')[0])]
                if not hit.empty:
                    kernel_time_pct = float(hit.iloc[0]['kernel_time_pct']) if 'kernel_time_pct' in hit else np.nan
                    kernel_total_time_ns = float(hit.iloc[0]['kernel_total_time_ns']) if 'kernel_total_time_ns' in hit else np.nan
            rows.append(dict(
                id=id_dir.name, kernel=kname, **vals,
                kernel_time_pct=kernel_time_pct, kernel_total_time_ns=kernel_total_time_ns
            ))
    if not rows:
        print("No rows to summarize."); return
    df = pd.DataFrame(rows)

    # Long-form table
    long_rows=[]
    def add_row(idv,kernel,cate,detail,metric_key,value,unit,cmd_metric):
        long_rows.append(dict(类别=cate, 分析详情=detail, 具体指标=metric_key, 值=value, 单位=unit, 命令或计数器=cmd_metric, id=idv, kernel=kernel))

    for _, r in df.iterrows():
        idv, kernel = r['id'], r['kernel']
        add_row(idv,kernel,"算力单元利用率","SM 利用率(峰值%)","sm__throughput.avg.pct_of_peak_sustained_elapsed", r['sm_pct_peak'],"%", "ncu metrics")
        add_row(idv,kernel,"算力单元利用率","Tensor Core 活跃(%)","sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", r['tensor_active_pct'],"%", "ncu metrics")
        add_row(idv,kernel,"算力单元利用率","FP32 效率(%)","flop_sp_efficiency", r['fp32_eff_pct'],"%", "ncu metrics")
        add_row(idv,kernel,"Cache 命中率","L1/TEX Sector 命中率(%)","l1tex__t_sector_hit_rate.pct", r['l1_hit_pct'],"%", "ncu metrics")
        add_row(idv,kernel,"Cache 命中率","L2 Sector 命中率(%)","lts__t_sector_hit_rate.pct", r['l2_hit_pct'],"%", "ncu metrics")
        add_row(idv,kernel,"内存带宽","DRAM 吞吐(峰值%)","gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed", r['dram_pct_peak'],"%", "ncu metrics")
        add_row(idv,kernel,"Warp 效率","Barrier Stall / Warp Active","smsp__warp_issue_stalled_barrier_per_warp_active.avg", r['stall_barrier'],"", "ncu metrics")
        add_row(idv,kernel,"Warp 效率","Short Scoreboard Stall / Warp Active","smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg", r['stall_short_sb'],"", "ncu metrics")
        add_row(idv,kernel,"Warp 效率","Long Scoreboard Stall / Warp Active","smsp__warp_issue_stalled_long_scoreboard_per_warp_active.avg", r['stall_long_sb'],"", "ncu metrics")
        add_row(idv,kernel,"Kernel 耗时","时间占比(%)","Time(%)", r['kernel_time_pct'],"%", "nsys stats: cuda_gpu_kern_sum")
        add_row(idv,kernel,"Kernel 耗时","总耗时(ns)","Total Time (ns)", r['kernel_total_time_ns'],"ns", "nsys stats: cuda_gpu_kern_sum")
        add_row(idv,kernel,"整体推理瓶颈","启发式(基于SM/DRAM/Cache)","bottleneck", r['bottleneck'],"", "heuristic")
    long_df = pd.DataFrame(long_rows)

    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_root / f"summary_{ts}.csv"
    md_path  = out_root / f"summary_{ts}.md"
    long_df.to_csv(csv_path, index=False)

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# 汇总表（关键指标）\n\n")
        for (idv, kernel), g in long_df.groupby(["id","kernel"]):
            f.write(f"## {idv} :: {kernel}\n\n")
            sub = g[g["类别"].isin(["Kernel 耗时","整体推理瓶颈","算力单元利用率","内存带宽","Cache 命中率"])][["类别","分析详情","具体指标","值","单位","命令或计数器"]]
            f.write(sub.fillna("nan").to_markdown(index=False))
            f.write("\n\n")

    figs_dir = out_root / "figs"
    figs_dir.mkdir(exist_ok=True, parents=True)

    # Top10 kernel by Time(%)
    for idv, g in df.groupby("id"):
        g2 = g.sort_values("kernel_time_pct", ascending=False).head(10)
        if not g2.empty:
            plt.figure()
            plt.title(f"{idv} Top10 Kernel Time(%)")
            plt.barh(g2["kernel"], g2["kernel_time_pct"])
            plt.gca().invert_yaxis()
            plt.xlabel("Time (%)")
            plt.tight_layout()
            plt.savefig(figs_dir / f"{idv}_topk_time.png")
            plt.close()

            plt.figure()
            plt.title(f"{idv} SM/DRAM % of Peak (Top10 by Time)")
            a = g2[["kernel","sm_pct_peak","dram_pct_peak"]].set_index("kernel")
            a.plot(kind="bar")
            plt.ylabel("%")
            plt.tight_layout()
            plt.savefig(figs_dir / f"{idv}_sm_dram_peak.png")
            plt.close()

    print(f"Wrote: {csv_path}")
    print(f"Wrote: {md_path}")
    print(f"Figures: {figs_dir}")
if __name__=="__main__":
    out_root = sys.argv[1] if len(sys.argv)>1 else "./out"
    main(out_root)
