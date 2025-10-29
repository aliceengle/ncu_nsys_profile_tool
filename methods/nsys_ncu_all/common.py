#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""公共工具函数，供 nsys + ncu 捕获与解析脚本复用。"""

from __future__ import annotations

import csv
import logging
import os
import pathlib
import re
import subprocess
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd


METHOD_ROOT = pathlib.Path(__file__).resolve().parent
BASH_METRIC_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_OUT_BASE = BASH_METRIC_ROOT / "out_nsys_ncu_all"

DEF_TOPN = int(os.environ.get("TOPN", "10"))
NCU_METRICS = os.environ.get(
    "NCU_METRICS",
    ",".join([
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active",
        "sm__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_pipe_fp32.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "flop_sp_efficiency",
        "l1tex__t_sector_hit_rate.pct",
        "l1tex__avg_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
        "lts__avg_hit_rate.pct",
        "smsp__warp_issue_stalled_barrier_per_warp_active.avg",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.avg",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.avg",
        "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio",
        "smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio",
    ])
)

NCU_SET = os.environ.get("NCU_SET", "full")
CAPTURE_RANGE = os.environ.get("CAPTURE_RANGE", "none")
NVTX_NAME = os.environ.get("NVTX_NAME", "")
NVTX_DOMAIN = os.environ.get("NVTX_DOMAIN", "")


def resolve_out_base(out_arg: Optional[str] = None) -> pathlib.Path:
    """根据参数/环境变量决定输出根目录。"""
    if out_arg:
        return pathlib.Path(out_arg)
    env_dir = os.environ.get("OUT_DIR", "").strip()
    return pathlib.Path(env_dir) if env_dir else DEFAULT_OUT_BASE


def _env_diff_str(base: Dict[str, str], new: Optional[Dict[str, str]]) -> str:
    if not new:
        return "{}"
    parts: List[str] = []
    for k, v in new.items():
        bv = base.get(k)
        if bv != v:
            sval = str(v)
            if len(sval) > 200:
                sval = sval[:200] + "..."
            parts.append(f"{k}={sval}")
    return "{" + ", ".join(parts) + "}"


def run(cmd: str, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> None:
    base_env = os.environ.copy()
    env_show = _env_diff_str(base_env, env)
    logging.info("[RUN] CWD=%s ENV_DIFF=%s", cwd or os.getcwd(), env_show)
    logging.info("[RUN] CMD=%s", cmd)
    res = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if res.stdout:
        for line in res.stdout.splitlines():
            logging.info("[OUT] %s", line)
    logging.info("[RUN] RC=%s", res.returncode)
    if res.returncode != 0:
        logging.error("Command failed (rc=%s): %s", res.returncode, cmd)
        raise SystemExit(f"Command failed (rc={res.returncode}): {cmd}")


def run_quiet(cmd: str, cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None):
    res = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        shell=True,
        executable="/bin/bash",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    return res.returncode, (res.stdout or "")


def merge_env_from_str(env_str: Optional[str], base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = (base or os.environ.copy()).copy()
    if env_str and env_str != "-":
        for pair in env_str.split(','):
            if not pair.strip():
                continue
            if '=' in pair:
                k, v = pair.split('=', 1)
                env[k] = v
    return env


def read_targets(tsv_path: str) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(tsv_path, newline="") as f:
        first = f.readline()
        f.seek(0)
        has_header = False
        if first:
            header_lower = [x.strip().lower() for x in first.rstrip('\n').split('\t')]
            known = {"id", "name", "cmd", "workdir", "env", "envs"}
            if any(h in known for h in header_lower):
                has_header = True

        if has_header:
            reader = csv.DictReader(f, delimiter='\t')

            def get_by_sub(row: Dict[str, str], subs: Iterable[str]) -> str:
                for k, v in row.items():
                    kl = k.strip().lstrip('#').strip().lower()
                    for sub in subs:
                        if sub in kl:
                            return (v or '').strip()
                return ''

            for raw in reader:
                name = (raw.get('id') or raw.get('name') or raw.get('NAME') or get_by_sub(raw, ['name'])).strip()
                if not name or name.startswith('#'):
                    continue
                cmd = (raw.get('cmd') or raw.get('CMD') or get_by_sub(raw, ['cmd', 'command'])).strip()
                workdir = (raw.get('workdir') or raw.get('WORKDIR') or get_by_sub(raw, ['workdir', 'work', 'wd'])).strip()
                envs = (raw.get('env') or raw.get('ENVS') or raw.get('envs') or get_by_sub(raw, ['env'])).strip()
                rows.append({'id': name, 'cmd': cmd, 'workdir': workdir, 'env': envs})
        else:
            reader = csv.reader(f, delimiter='\t')
            for cols in reader:
                if not cols:
                    continue
                if cols[0].strip().startswith('#'):
                    continue
                cols = (cols + ['','','',''])[:4]
                name, cmd, workdir, envs = [c.strip() for c in cols]
                if not name:
                    continue
                rows.append({'id': name, 'cmd': cmd, 'workdir': workdir, 'env': envs})

    logging.info("[TARGETS] Parsed %d rows from %s", len(rows), tsv_path)
    for idx, row in enumerate(rows, 1):
        logging.info("[TARGETS] #%d id=%s cmd=%s workdir=%s env=%s", idx, row['id'], row['cmd'], row['workdir'], row['env'])
    return rows


def pick_topk_kernels(csv_path: str, topn: int) -> List[str]:
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.warning("[NSYS] kernel summary CSV not found: %s", csv_path)
        return []
    except pd.errors.EmptyDataError:
        logging.warning("[NSYS] kernel summary CSV is empty: %s", csv_path)
        return []

    if df is None or df.empty:
        logging.warning("[NSYS] kernel summary CSV has no rows: %s", csv_path)
        return []

    cols = df.columns.str.lower()
    name_idx = np.where(cols.str.contains('name'))[0]
    if not len(name_idx):
        logging.warning("[NSYS] no kernel name column detected in %s", csv_path)
        return []
    name_col = df.columns[name_idx[0]]

    time_pct_idx = np.where(cols.str.contains('time.*%'))[0]
    if len(time_pct_idx):
        sort_col = df.columns[time_pct_idx[0]]
    else:
        total_ns_idx = np.where(cols.str.contains('total.*time.*ns'))[0]
        sort_col = df.columns[total_ns_idx[0]] if len(total_ns_idx) else name_col

    try:
        series = df.sort_values(sort_col, ascending=False)[name_col].head(topn)
    except Exception as exc:  # noqa: BLE001
        logging.warning("[NSYS] failed to derive top kernels from %s: %s", csv_path, exc)
        return []

    names = [str(x) for x in series if str(x).strip()]
    out_txt = pathlib.Path(csv_path).with_suffix('.topk.txt')
    try:
        out_txt.write_text('\n'.join(names))
    except Exception:  # noqa: BLE001
        pass

    logging.info("[TOPK] Picked top-%d kernels (%s)", len(names), names)
    return names


def safe_regex_contains(text: str) -> str:
    return "regex:.*" + re.escape(text) + ".*"


def sanitize_segment(text: str) -> str:
    seg = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text))[:80]
    return seg or 'task'


def build_group_name(ids: Sequence[str]) -> str:
    safe_ids = [sanitize_segment(x) for x in ids if x]
    return '+'.join(safe_ids) if safe_ids else 'task'

