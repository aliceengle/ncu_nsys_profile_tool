#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, csv, shlex, subprocess, pathlib, re, pandas as pd, numpy as np

ROOT = pathlib.Path(__file__).resolve().parent
OUT  = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

DEF_TOPN = int(os.environ.get("TOPN", "5"))
NCU_METRICS = os.environ.get("NCU_METRICS",
    "sm__throughput.avg.pct_of_peak_sustained_elapsed,"
    "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,"
    "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active,"
    "flop_sp_efficiency,"
    "l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,"
    "smsp__warp_issue_stalled_barrier_per_warp_active.avg,"
    "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg,"
    "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.avg"
)

# NVTX optional capture-range for nsys
CAPTURE_RANGE = os.environ.get("CAPTURE_RANGE", "none")  # none / nvtx / cudaProfilerApi / hotkey
NVTX_NAME = os.environ.get("NVTX_NAME", "")              # e.g. "range@domain"
NVTX_DOMAIN = os.environ.get("NVTX_DOMAIN", "")

def run(cmd, cwd=None, env=None):
    print(f"[RUN] {cmd}")
    res = subprocess.run(cmd, cwd=cwd, env=env, shell=True, executable="/bin/bash")
    if res.returncode != 0:
        raise SystemExit(f"Command failed: {cmd}")

def read_targets(tsv):
    rows=[]
    with open(tsv, newline='') as f:
        r=csv.DictReader(f, delimiter='\t')
        for row in r:
            if not row['id'] or row['id'].startswith('#'): continue
            rows.append(row)
    return rows

def pick_topk_kernels(csv_path, topn):
    df = pd.read_csv(csv_path)
    cols = df.columns.str.lower()
    name = df.columns[np.where(cols.str.contains('name'))[0][0]]
    m = np.where(cols.str.contains('time.*%'))[0]
    if len(m): key = df.columns[m[0]]
    else:
        m2 = np.where(cols.str.contains('total.*time.*ns'))[0]
        key = df.columns[m2[0]] if len(m2) else name
    s = df.sort_values(key, ascending=False)[name].head(topn)
    out = pathlib.Path(csv_path).with_suffix('.topk.txt')
    out.write_text('\n'.join(map(str,s)))
    return [str(x) for x in s]

def safe_regex_contains(s):
    return "regex:.*" + re.escape(s) + ".*"

def do_one(id, cmd, workdir, envs, topn):
    outdir = OUT / id
    (outdir/"nsys").mkdir(parents=True, exist_ok=True)
    (outdir/"ncu").mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if envs and envs != "-":
        for pair in envs.split(","):
            if not pair.strip(): continue
            if "=" in pair:
                k,v = pair.split("=",1)
                env[k]=v

    # nsys profile
    nvtxcap = f'--nvtx-capture={NVTX_NAME}{("@" + NVTX_DOMAIN) if NVTX_DOMAIN else ""}' if (CAPTURE_RANGE=="nvtx" and NVTX_NAME) else ""
    run(
        f'nsys profile --force-overwrite=true --trace=cuda,osrt,nvtx '
        f'--capture-range={CAPTURE_RANGE} '
        f'{"--capture-range-end=stop" if CAPTURE_RANGE!="none" else ""} '
        f'{nvtxcap} '
        f'--sample=none --stats=true --output "{outdir}/nsys/{id}" {cmd}',
        cwd=(workdir if workdir and workdir!="- " else None),
        env=env
    )

    # nsys stats → kernel summary CSV (force re-export & overwrite for repeated runs)
    run(
        f'nsys stats --report cuda_gpu_kern_sum --format csv '
        f'--force-export=true --force-overwrite=true '
        f'--output "{outdir}/nsys/{id}" "{outdir}/nsys/{id}.nsys-rep"'
    )

    # TopN kernels
    topk = pick_topk_kernels(f"{outdir}/nsys/{id}_cuda_gpu_kern_sum.csv", topn)

    # ncu for each kernel
    for kname in topk:
        if not kname: continue
        pattern = safe_regex_contains(kname)
        base = outdir / "ncu" / re.sub(r'[\/ :\\]+', '____', kname)[:120]
        run(
            f'ncu -o "{base}" --kernel-name-base demangled --kernel-name "{pattern}" '
            f'--set full --cache-control all --clock-control base '
            f'--profile-from-start yes --target-processes all',
            cwd=(workdir if workdir and workdir!="- " else None),
            env=env
        )
        rep=f"{base}.ncu-rep"
        run(f'ncu --import "{rep}" --page raw --csv > "{base}_ncu_raw.csv"')
        run(f'ncu --import "{rep}" --csv --metrics {NCU_METRICS} > "{base}_metrics_id.csv"')

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(ROOT/"targets.tsv"))
    ap.add_argument("--topn", type=int, default=DEF_TOPN)
    args = ap.parse_args()

    rows = read_targets(args.targets)
    for r in rows:
        do_one(r['id'], r['cmd'], r.get('workdir') or None, r.get('env') or None, args.topn)

    # Summarize + plots
    run(f'python "{ROOT}/parse_and_plot.py" "{OUT}"')

if __name__=="__main__":
    main()
