#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys, csv, shlex, subprocess, pathlib, re, shutil, pandas as pd, numpy as np
import logging

ROOT = pathlib.Path(__file__).resolve().parent
OUT  = ROOT / "out"
OUT.mkdir(parents=True, exist_ok=True)

DEF_TOPN = int(os.environ.get("TOPN", "5"))
# Ensure metrics default is a single comma-separated string
NCU_METRICS = os.environ.get(
    "NCU_METRICS",
    ",".join([
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "flop_sp_efficiency",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.avg",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.avg",
    ])
)

# Nsight Compute capture profile breadth
# Default to a broader preset to collect key metrics, while still allowing override
NCU_SET = os.environ.get("NCU_SET", "detailed")
# Include commonly useful sections to improve coverage of stalls and cache metrics
NCU_SECTIONS = os.environ.get(
    "NCU_SECTIONS",
    ",".join([
        "LaunchStats",
        "SpeedOfLight",
        "SpeedOfLight_RooflineChart",
        "SpeedOfLight_HierarchicalSingleRooflineChart",
        "SpeedOfLight_HierarchicalTensorRooflineChart",
        "SchedulerStats",
        "WarpStateStats",
        "MemoryWorkloadAnalysis",
        "Occupancy",
        "InstructionStats",
        "ComputeWorkloadAnalysis",
        "SourceCounters",
    ])
)

# NVTX optional capture-range for nsys
CAPTURE_RANGE = os.environ.get("CAPTURE_RANGE", "none")  # none / nvtx / cudaProfilerApi / hotkey
NVTX_NAME = os.environ.get("NVTX_NAME", "")              # e.g. "range@domain"
NVTX_DOMAIN = os.environ.get("NVTX_DOMAIN", "")

def _env_diff_str(base: dict, new: dict):
    if not new:
        return "{}"
    parts = []
    for k, v in new.items():
        bv = base.get(k)
        if bv != v:
            # redact very long values
            sval = str(v)
            if len(sval) > 200:
                sval = sval[:200] + "..."
            parts.append(f"{k}={sval}")
    return "{" + ", ".join(parts) + "}"

def run(cmd, cwd=None, env=None):
    base_env = os.environ.copy()
    env_show = _env_diff_str(base_env, env or {})
    logging.info("[RUN] CWD=%s ENV_DIFF=%s", cwd or os.getcwd(), env_show)
    logging.info("[RUN] CMD=%s", cmd)
    res = subprocess.run(
        cmd, cwd=cwd, env=env, shell=True, executable="/bin/bash",
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    if res.stdout:
        for line in res.stdout.splitlines():
            logging.info("[OUT] %s", line)
    logging.info("[RUN] RC=%s", res.returncode)
    if res.returncode != 0:
        logging.error("Command failed (rc=%s): %s", res.returncode, cmd)
        raise SystemExit(f"Command failed (rc={res.returncode}): {cmd}")

def run_quiet(cmd, cwd=None, env=None):
    """Run a command and return (rc, stdout). Does not raise."""
    res = subprocess.run(cmd, cwd=cwd, env=env, shell=True, executable="/bin/bash",
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    return res.returncode, (res.stdout or "")

def ncu_get_sets_sections():
    """Return (sets, sections) as lists of usable tokens reported by ncu.
    Filters out separators and table headers, strips trailing commas.
    """
    tok_ok = lambda s: bool(re.match(r'^[A-Za-z0-9][A-Za-z0-9_-]*$', s))
    sets = []
    sections = []
    rc, out = run_quiet('ncu --list-sets')
    if rc == 0 and out:
        for line in out.splitlines():
            s = line.strip().strip(',')
            if not s or 'available' in s.lower() or set(s) == {'-'}:
                continue
            head = s.split()[0]
            if tok_ok(head):
                sets.append(head)
    rc, out = run_quiet('ncu --list-sections')
    if rc == 0 and out:
        for line in out.splitlines():
            s = line.strip().strip(',')
            if not s or 'available' in s.lower() or set(s) == {'-'}:
                continue
            head = s.split()[0]
            if tok_ok(head):
                sections.append(head)
    logging.info("[NCU] available sets=%s", sets)
    logging.info("[NCU] available sections(sample)=%s", sections[:20])
    return sets, sections

def read_targets(tsv):
    """Read targets.tsv supporting two header styles:
    - id, cmd, workdir, env (legacy)
    - NAME, CMD, WORKDIR, ENVS (new; case-insensitive)
    Also supports headerless 4-column TSV in order: NAME, CMD, WORKDIR, ENVS.
    """
    rows = []
    with open(tsv, newline='') as f:
        # Peek first line to decide header presence
        first = f.readline()
        f.seek(0)
        has_header = False
        header_lower = []
        if first:
            header_lower = [x.strip().lower() for x in first.rstrip('\n').split('\t')]
            # consider it a header if contains any known column name
            known = {"id", "name", "cmd", "workdir", "env", "envs"}
            if any(h in known for h in header_lower):
                has_header = True

        if has_header:
            r = csv.DictReader(f, delimiter='\t')
            for row in r:
                # robust key lookup: accept keys containing substrings (case-insensitive), strip leading '#'
                def get_by_sub(key_subs):
                    for k, v in row.items():
                        kl = k.strip().lstrip('#').strip().lower()
                        for sub in key_subs:
                            if sub in kl:
                                return (v or '').strip()
                    return ''
                name = (row.get('id') or row.get('name') or row.get('NAME') or get_by_sub(['name'])).strip()
                if not name or name.startswith('#'):
                    continue
                cmd = (row.get('cmd') or row.get('CMD') or get_by_sub(['cmd','command'])).strip()
                workdir = (row.get('workdir') or row.get('WORKDIR') or get_by_sub(['workdir','work','wd'])).strip()
                envs = (row.get('env') or row.get('ENVS') or row.get('envs') or get_by_sub(['env'])).strip()
                rows.append({'id': name, 'cmd': cmd, 'workdir': workdir, 'env': envs})
        else:
            r = csv.reader(f, delimiter='\t')
            for cols in r:
                if not cols:
                    continue
                # skip comment line
                if cols[0].strip().startswith('#'):
                    continue
                # pad to 4 columns
                cols = (cols + ['','','',''])[:4]
                name, cmd, workdir, envs = [c.strip() for c in cols]
                if not name:
                    continue
                rows.append({
                    'id': name,
                    'cmd': cmd,
                    'workdir': workdir,
                    'env': envs,
                })
    logging.info("[TARGETS] Parsed %d rows from %s", len(rows), tsv)
    for i, r in enumerate(rows, 1):
        logging.info("[TARGETS] #%d id=%s cmd=%s workdir=%s env=%s", i, r['id'], r['cmd'], r['workdir'], r['env'])
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
    logging.info("[TOPK] Picked top-%d kernels from %s -> %s", topn, csv_path, list(map(str, s)))
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

    # preflight for tools
    nsys_path = shutil.which('nsys')
    ncu_path  = shutil.which('ncu')
    logging.info("[TOOLS] nsys=%s ncu=%s", nsys_path or 'NOT FOUND', ncu_path or 'NOT FOUND')
    if not nsys_path:
        logging.error("nsys not found in PATH; skip id=%s", id)
        raise SystemExit("nsys not found in PATH")
    if not ncu_path:
        logging.error("ncu not found in PATH; subsequent ncu steps will fail for id=%s", id)

    # Discover NCU sets/sections for compatibility
    avail_sets, avail_sections = ncu_get_sets_sections()
    # Choose set: prefer requested; else try common presets; else omit
    use_set = None
    if avail_sets:
        # map case-insensitively
        lower_sets = {s.lower(): s for s in avail_sets}
        if NCU_SET and NCU_SET.lower() in lower_sets:
            use_set = lower_sets[NCU_SET.lower()]
        else:
            for pref in ['speed-of-light','basic','detailed','full','roofline']:
                if pref in lower_sets:
                    use_set = lower_sets[pref]; break
        if use_set is None:
            logging.warning("[NCU] no preferred set available; will omit --set")
    # Validate requested sections (split by comma); fall back to a small useful subset
    use_sections = []
    if avail_sections:
        lower_secs = {s.lower(): s for s in avail_sections}
        req_secs = [s.strip() for s in (NCU_SECTIONS.split(',') if NCU_SECTIONS else []) if s.strip()]
        if req_secs:
            for sec in req_secs:
                key = sec.lower()
                if key in lower_secs:
                    use_sections.append(lower_secs[key])
                else:
                    logging.warning("[NCU] requested section '%s' not available; skipping", sec)
        else:
            # default preferred subset
            for pref in ['LaunchStats','SchedulerStats','MemoryWorkloadAnalysis']:
                key = pref.lower()
                if key in lower_secs:
                    use_sections.append(lower_secs[key])

    # nsys profile
    nvtxcap = f'--nvtx-capture={NVTX_NAME}{("@" + NVTX_DOMAIN) if NVTX_DOMAIN else ""}' if (CAPTURE_RANGE=="nvtx" and NVTX_NAME) else ""
    logging.info("[NSYS] id=%s outdir=%s workdir=%s", id, outdir, workdir or os.getcwd())
    run(
        f'nsys profile --force-overwrite=true --trace=cuda,osrt,nvtx '
        f'--capture-range={CAPTURE_RANGE} '
        f'{"--capture-range-end=stop" if CAPTURE_RANGE!="none" else ""} '
        f'{nvtxcap} '
        f'--sample=none --stats=true --output "{outdir}/nsys/{id}" {cmd}',
        cwd=(workdir if workdir and workdir!="-" else None),
        env=env
    )

    # nsys stats → kernel summary CSV (force re-export & overwrite for repeated runs)
    logging.info("[NSYS] Export stats CSV for id=%s", id)
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
        base = outdir / "ncu" / re.sub(r'[^A-Za-z0-9._-]+', '_', kname)[:120]
        logging.info("[NCU] id=%s kernel=%s pattern=%s base=%s", id, kname, pattern, base)
        # Compose capture options: preset set + optional sections for stalls+cache hit rates
        cap_opts = ''
        if use_set:
            cap_opts += f'--set {use_set} '
        if use_sections:
            # Pass multiple --section flags (CLI does not accept comma-joined list)
            cap_opts += ' '.join([f'--section {s}' for s in use_sections]) + ' '
        logging.info("[NCU] opts set=%s sections=%s", use_set, ','.join(use_sections) if use_sections else '')
        run(
            f'ncu -f -o "{base}" --kernel-name-base demangled --kernel-name "{pattern}" '
            f'{cap_opts}--cache-control all --clock-control base '
            f'--profile-from-start yes --target-processes all {cmd}',
            cwd=(workdir if workdir and workdir!="-" else None),
            env=env
        )
        rep=f"{base}.ncu-rep"
        logging.info("[NCU] Export raw CSV: %s_ncu_raw.csv", base)
        run(f'ncu --import "{rep}" --page raw --csv > "{base}_ncu_raw.csv"')
        logging.info("[NCU] Export metrics-id CSV: %s_metrics_id.csv", base)
        run(f'ncu --import "{rep}" --csv --metrics {NCU_METRICS} > "{base}_metrics_id.csv"')

def main():
    import argparse
    # 设置同时输出到终端与文件的日志
    log_path = OUT / "run_profiling.log"
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    # 清理旧 handler，避免重复
    logger.handlers = []
    sh = logging.StreamHandler(stream=sys.stdout)
    fh = logging.FileHandler(str(log_path), mode='w', encoding='utf-8')
    fmt = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
    sh.setFormatter(fmt)
    fh.setFormatter(fmt)
    logger.addHandler(sh)
    logger.addHandler(fh)
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=str(ROOT/"targets.tsv"))
    ap.add_argument("--topn", type=int, default=DEF_TOPN)
    args = ap.parse_args()

    logging.info("[START] run_profiling targets=%s topn=%d", args.targets, args.topn)
    rows = read_targets(args.targets)
    for r in rows:
        logging.info("[DO] id=%s", r['id'])
        try:
            do_one(r['id'], r['cmd'], r.get('workdir') or None, r.get('env') or None, args.topn)
        except SystemExit as e:
            logging.error("[FAIL] id=%s error=%s", r['id'], e)
            continue

    # Summarize + plots
    logging.info("[SUM] parse_and_plot out=%s", OUT)
    # 将日志文件路径传给 parse_and_plot.py 做追加
    run(f'{sys.executable} "{ROOT}/parse_and_plot.py" "{OUT}" --log "{log_path}"')

if __name__=="__main__":
    main()
