#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""使用 nsys 与单次 ncu (--set full) 对所有 kernel 进行捕获。"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import shutil
import sys
from typing import Dict, List, Optional


THIS_DIR = pathlib.Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import common  # noqa: E402


def _setup_logger(log_path: pathlib.Path) -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s %(levelname)s - %(message)s')
    sh = logging.StreamHandler(stream=sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    fh = logging.FileHandler(str(log_path), mode='w', encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def _discover_tools(env: Dict[str, str]) -> None:
    nsys_path = shutil.which('nsys', path=env.get('PATH'))
    ncu_path = shutil.which('ncu', path=env.get('PATH'))
    logging.info("[TOOLS] nsys=%s ncu=%s", nsys_path or 'NOT FOUND', ncu_path or 'NOT FOUND')
    if not nsys_path:
        raise SystemExit("nsys not found in PATH")
    if not ncu_path:
        raise SystemExit("ncu not found in PATH")


def _ensure_ncu_set(env: Dict[str, str]) -> None:
    if not common.NCU_SET:
        return
    rc, out = common.run_quiet('ncu --list-sets', env=env)
    if rc != 0 or not out:
        logging.warning("[NCU] failed to query available sets (rc=%s); 仍将尝试 --set %s", rc, common.NCU_SET)
        return
    available = {line.strip().split()[0] for line in out.splitlines() if line.strip() and not line.startswith('---')}
    normalized = {item.lower() for item in available}
    if common.NCU_SET.lower() not in normalized:
        logging.error("[NCU] set '%s' not available; available=%s", common.NCU_SET, list(sorted(available))[:10])
        raise SystemExit(f"Requested NCU set '{common.NCU_SET}' not in available list")


def _capture_one(row: Dict[str, str], out_root: pathlib.Path, use_nvtx: bool) -> None:
    target_id = row['id']
    workdir = row.get('workdir') or None
    env = common.merge_env_from_str(row.get('env'), os.environ.copy())

    _discover_tools(env)
    _ensure_ncu_set(env)

    outdir = out_root / target_id
    if os.environ.get("CLEAN_PER_ID", "0") == "1" and outdir.exists():
        shutil.rmtree(outdir, ignore_errors=True)
    (outdir / "nsys").mkdir(parents=True, exist_ok=True)
    (outdir / "ncu").mkdir(parents=True, exist_ok=True)

    nvtxcap = ''
    if common.CAPTURE_RANGE == "nvtx" and common.NVTX_NAME:
        suffix = f"@{common.NVTX_DOMAIN}" if common.NVTX_DOMAIN else ''
        nvtxcap = f"--nvtx-capture={common.NVTX_NAME}{suffix}"

    trace_items = ["cuda", "osrt"]
    if use_nvtx:
        trace_items.append("nvtx")
    trace_arg = ",".join(trace_items)

    logging.info("[NSYS] id=%s outdir=%s workdir=%s nvtx=%s", target_id, outdir, workdir or os.getcwd(), use_nvtx)
    common.run(
        f'nsys profile --force-overwrite=true --trace={trace_arg} '
        f'--capture-range={common.CAPTURE_RANGE} '
        f"{'--capture-range-end=stop' if common.CAPTURE_RANGE != 'none' else ''} "
        f'{nvtxcap} --sample=none --stats=true '
        f'--output "{outdir}/nsys/{target_id}" {row["cmd"]}',
        cwd=(workdir if workdir and workdir != "-" else None),
        env=env,
    )

    common.run(
        f'nsys stats --report cuda_gpu_kern_sum --format csv '
        f'--force-export=true --force-overwrite=true '
        f'--output "{outdir}/nsys/{target_id}" "{outdir}/nsys/{target_id}.nsys-rep"',
        env=env,
    )

    capture_tag = f"{common.sanitize_segment(target_id)}_all"
    capture_base = outdir / "ncu" / capture_tag
    logging.info("[NCU] id=%s capture once -> %s", target_id, capture_base)
    cmd_ncu = (
        f'ncu -f -o "{capture_base}" '
        f'--set {common.NCU_SET} --cache-control all --clock-control base '
        f'--profile-from-start yes --target-processes all '
        f'{"--nvtx " if use_nvtx else ""}{row["cmd"]}'
    )
    common.run(
        cmd_ncu,
        cwd=(workdir if workdir and workdir != "-" else None),
        env=env,
    )

    rep_path = f"{capture_base}.ncu-rep"
    full_raw_csv = f"{capture_base}_ncu_raw.csv"
    full_metrics_csv = f"{capture_base}_metrics_id.csv"
    common.run(f'ncu --import "{rep_path}" --page raw --csv > "{full_raw_csv}"')
    common.run(f'ncu --import "{rep_path}" --csv --metrics {common.NCU_METRICS} > "{full_metrics_csv}"')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=str(common.BASH_METRIC_ROOT / "targets.tsv"))
    parser.add_argument("--topn", type=int, default=common.DEF_TOPN, help="Top-N kernel 仅用于后续解析阶段")
    parser.add_argument("--nvtx", action="store_true", help="是否同时采集 NVTX 数据")
    parser.add_argument("--out-base", default=None, help="自定义输出根目录 (默认 out_nsys_ncu_all)")
    args = parser.parse_args()

    out_base = common.resolve_out_base(args.out_base)
    rows = common.read_targets(args.targets)

    seen: set[str] = set()
    ordered_ids: List[str] = []
    for r in rows:
        tid = r['id']
        if tid and tid not in seen:
            seen.add(tid)
            ordered_ids.append(tid)

    group_name = common.build_group_name(ordered_ids)
    out_root = out_base / group_name
    if os.environ.get("CLEAN_OUT", "0") == "1" and out_root.exists():
        shutil.rmtree(out_root, ignore_errors=True)
    out_root.mkdir(parents=True, exist_ok=True)

    log_path = out_root / "capture_once.log"
    _setup_logger(log_path)
    logging.info("[START] capture_once targets=%s group=%s out=%s", args.targets, group_name, out_root)
    logging.info("[CFG] NCU_SET=%s TOPN=%d", common.NCU_SET, args.topn)

    for row in rows:
        logging.info("[DO] id=%s", row['id'])
        try:
            _capture_one(row, out_root, use_nvtx=args.nvtx)
        except SystemExit as exc:  # noqa: BLE001
            logging.error("[FAIL] id=%s error=%s", row['id'], exc)
            continue

    logging.info("[DONE] capture stage finished. 输出根目录: %s", out_root)


if __name__ == "__main__":
    main()
