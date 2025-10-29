#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""解析单次捕获的 nsys/ncu 报告，导出 Top-N kernel 指标并绘制汇总。"""

from __future__ import annotations

import argparse
import logging
import os
import pathlib
import re
import sys
from typing import Dict, List


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


def _ensure_capture_exists(id_dir: pathlib.Path, capture_base: pathlib.Path) -> None:
    rep_path = capture_base.with_suffix('.ncu-rep')
    if not rep_path.exists():
        raise SystemExit(f"NCU capture file missing for {id_dir.name}: {rep_path}")


def _export_topk_for_id(row: Dict[str, str], out_root: pathlib.Path, topn: int) -> None:
    target_id = row['id']
    id_dir = out_root / target_id
    if not id_dir.exists():
        logging.warning("[SKIP] 输出目录不存在，可能尚未捕获: %s", id_dir)
        return

    capture_tag = f"{common.sanitize_segment(target_id)}_all"
    capture_base = id_dir / "ncu" / capture_tag
    _ensure_capture_exists(id_dir, capture_base)

    kern_csv = id_dir / "nsys" / f"{target_id}_cuda_gpu_kern_sum.csv"
    if not kern_csv.exists():
        logging.warning("[SKIP] nsys kernel summary 缺失: %s", kern_csv)
        return

    topk = common.pick_topk_kernels(str(kern_csv), topn)
    if not topk:
        logging.warning("[SKIP] 未能从 %s 中提取 Top-%d kernel", kern_csv, topn)
        return

    env = common.merge_env_from_str(row.get('env'), os.environ.copy())
    rep_path = f"{capture_base}.ncu-rep"
    exported = 0
    for kernel_name in topk:
        if not kernel_name:
            continue
        sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', kernel_name)[:120]
        base = id_dir / "ncu" / sanitized
        logging.info("[NCU] export metrics for id=%s kernel=%s", target_id, kernel_name)
        pattern = common.safe_regex_contains(kernel_name)
        common.run(
            f'ncu --import "{rep_path}" --kernel-name-base demangled '
            f'--kernel-name "{pattern}" --page raw --csv > "{base}_ncu_raw.csv"',
            env=env,
        )
        common.run(
            f'ncu --import "{rep_path}" --kernel-name-base demangled '
            f'--kernel-name "{pattern}" --csv --metrics {common.NCU_METRICS} > "{base}_metrics_id.csv"',
            env=env,
        )
        exported += 1
    logging.info("[NCU] id=%s 已导出 %d 个 kernel 指标", target_id, exported)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=str(common.BASH_METRIC_ROOT / "targets.tsv"))
    parser.add_argument("--topn", type=int, default=common.DEF_TOPN)
    parser.add_argument("--out-base", default=None)
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
    out_root.mkdir(parents=True, exist_ok=True)

    log_path = out_root / "analyze_topk.log"
    _setup_logger(log_path)
    logging.info("[START] analyze_topk targets=%s group=%s out=%s", args.targets, group_name, out_root)
    logging.info("[CFG] TOPN=%d", args.topn)

    for row in rows:
        logging.info("[TOPK] id=%s", row['id'])
        try:
            _export_topk_for_id(row, out_root, args.topn)
        except SystemExit as exc:  # noqa: BLE001
            logging.error("[FAIL] id=%s error=%s", row['id'], exc)
            continue

    logging.info("[SUM] 调用 parse_and_plot.py 生成汇总")
    parse_script = common.BASH_METRIC_ROOT / "parse_and_plot.py"
    common.run(f'{sys.executable} "{parse_script}" "{out_root}" --log "{log_path}"')
    logging.info("[DONE] analyze 阶段完成: %s", out_root)


if __name__ == "__main__":
    main()
