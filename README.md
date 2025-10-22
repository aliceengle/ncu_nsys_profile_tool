# ncu_nsys_profiler_pack (final)

一键采集并汇总以下 GPU 性能指标：
- 算力单元利用率（SM/Tensor/FP32）
- Cache 命中率（L1/TEX/L2）
- 内存带宽（DRAM 吞吐占峰值）
- Warp 效率（Barrier/Short/Long Scoreboard stalls）
- **Kernel 耗时**（来自 Nsight Systems `cuda_gpu_kern_sum`）
- **整体推理瓶颈**（基于 SM/DRAM/Cache 的启发式：Compute / Memory / Mixed / Memory/Latency）

## 依赖
```bash
pip install -r requirements.txt
```

## 目标列表
编辑 `targets.tsv`（TSV 三到四列：id, cmd, workdir, env）。

示例：
```
id	cmd	workdir	env
bert_inf	python infer.py --bs 8 --seq 128	./models	CUDA_VISIBLE_DEVICES=0
resnet_train	python train.py --epochs 1	./cv	CUDA_VISIBLE_DEVICES=0
```

> `workdir` 或 `env` 可留空（用 `-` 表示）。`env` 多个变量用逗号分隔，如 `A=1,B=2`。

## 运行
**Linux/macOS（bash）**
```bash
bash profile_and_report.sh
```

**跨平台（Python）**
```bash
python run_profiling.py
```

运行后在 `out/` 看到：
- `nsys/<id>.nsys-rep` 与 `<id>_cuda_gpu_kern_sum.csv`（Kernel 耗时/占比）；
- `ncu/<kernel>/*.ncu-rep` 与 `*_ncu_raw.csv`、`*_metrics_id.csv`；
- `summary_*.csv/.md` 与 `figs/*.png`。

## 重要开关（健壮性与可重复）
- **NVTX 非强依赖**：默认 `CAPTURE_RANGE=none`。若你给应用插了 NVTX，可设：
  - `CAPTURE_RANGE=nvtx`，并可选 `NVTX_NAME="range@domain"`（同 `nsys -p`）。
  - 若使用 NVTX 动态字符串，建议 `export NSYS_NVTX_PROFILER_REGISTER_ONLY=0` 以捕获非注册字符串。
- **重复运行/覆盖输出**：`nsys stats` 已加 `--force-export=true --force-overwrite=true`，确保从最新的 `.nsys-rep` 重新导出并覆盖同名 CSV。
- **ncu 多进程**：默认 `--target-processes all`。需要更强可比性时，可改用 `--replay-mode application` 与合适的 `--app-replay-match`。
- **缓存/频率控制**：默认 `--cache-control all --clock-control base` 以提高多次回放的一致性；若想保留“真实缓存状态”，可把 `cache-control` 改为 `none`。

## 参考（命令出处）
- Nsight Systems 用户指南（`--capture-range`/NVTX 捕获、`TMPDIR`、`stats` 报表/CSV、`--force-export` 与 `--force-overwrite` 等）。
- Nsight Compute CLI 文档（`--page raw --csv`、`--import`、`--metrics`、`--kernel-name-base/--kernel-name`、`--target-processes all`、`--cache-control`、`--clock-control`、回放与匹配等）。

> 详见随答复中的引用链接。
