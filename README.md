# ncu_nsys_profiler_pack（bash_metric 子仓库）

本子仓库提供一套以 Bash 为主的 Nsight Systems（nsys）+ Nsight Compute（ncu）采集与汇总流水线，适用于对深度学习推理/训练脚本进行端到端的 GPU 性能画像与瓶颈分析。

一键采集并汇总以下 GPU 性能指标：
- 算力单元利用率（SM/Tensor/FP32）
- Cache 命中率（L1/TEX/L2）
- 内存带宽（DRAM 吞吐占峰值）
- Warp 效率（Barrier/Short/Long Scoreboard stalls）
- **Kernel 耗时**（来自 Nsight Systems `cuda_gpu_kern_sum`）
- **整体推理瓶颈**（基于 SM/DRAM/Cache 的启发式：Compute / Memory / Mixed / Memory/Latency）

## 目录结构与作用
- `profile_and_report.sh`：核心 Bash 流水线脚本，读取 `targets.tsv` 批量执行：
  - nsys 采集（生成 `.nsys-rep`）；
  - nsys stats 导出（生成 `*_cuda_gpu_kern_sum.csv` 等）；
  - 自动选取 Top-N kernel 并生成正则；
  - ncu 采集（生成 `.ncu-rep` 与 raw CSV）。
- `run_profiling.py`：等价的 Python 实现，便于跨平台与小改造。
- `parse_and_plot.py`：将 ncu 与 nsys 的 CSV 聚合成汇总表并绘图。
- `targets.tsv`：任务定义表（TSV），一行一个目标。
- `out/`：默认输出根目录（nsys/ncu/汇总，含全局运行日志 `profile_and_report.log`）。
- 其它：`requirements.txt`、`utils.py` 辅助脚本。

## 依赖
环境依赖：
- 安装并确保 PATH 中可找到 `nsys` 与 `ncu`。
- `python3` 与 `pip`；安装本目录所需 Python 依赖：
  ```bash
  pip install -r requirements.txt
  ```

## 目标列表
编辑 `targets.tsv`（TSV 四列，使用 TAB 分隔）：
`NAME<TAB>CMD<TAB>WORKDIR<TAB>ENVS`

示例：
```
NAME	CMD	WORKDIR	ENVS
bert_inf	python infer.py --bs 8 --seq 128	./models	CUDA_VISIBLE_DEVICES=0
resnet_train	python train.py --epochs 1	./cv	CUDA_VISIBLE_DEVICES=0
```

> `WORKDIR` 或 `ENVS` 可留空（用 `-` 表示）。`ENVS` 多个变量用逗号分隔，如 `A=1,B=2`；留空或 `-` 不会传给 `env`，避免破坏 PATH。

## 运行
**Linux/macOS（bash）**
在 `scripts/bash_metric` 目录下执行：
```bash
bash profile_and_report.sh
```

**跨平台（Python）**
```bash
python run_profiling.py
```

运行后在 `out/` 会看到：
- `nsys/<id>.nsys-rep` 与 `<id>_cuda_gpu_kern_sum.csv`（Kernel 耗时/占比）；
- `ncu/<kernel>/*.ncu-rep` 与 `*_ncu_raw.csv`、`*_metrics_id.csv`；
- `profile_and_report.log`：单一全局日志，记录所有目标的全部阶段输出，重复执行会覆盖重写；
- `summary_*.csv/.md` 与 `figs/*.png`。

## 运行阶段与进度打印
`profile_and_report.sh` 内部将每个目标的执行划分为 6 个阶段，并打印进度：
1) 准备目录与上下文
2) nsys 采集
3) nsys 导出 CSV
4) 生成 TopN kernel 正则
5) ncu 采集
6) ncu 导出 raw CSV

每步开始会打印 `[STEP x/6] 开始：...`，完成后打印 `[STEP x/6] 完成：...；剩余 n 步`，失败时打印 `[STEP x/6] 失败：...` 并依据 `CONTINUE_ON_ERROR` 决定是否继续。

## 日志记录与配置
- 全局日志：`out/profile_and_report.log`（每次运行覆盖重写）。
- 自定义路径：运行前设置 `GLOBAL_LOG=/path/to/run.log`。
- 自定义输出根目录：设置 `OUT_DIR`（默认 `./out`）。
- 自定义目标表路径：设置 `TARGETS_TSV`（默认读取脚本同目录的 `targets.tsv`）。
- 查看运行过程：
  - `tail -f out/profile_and_report.log`
  - 或 `less +F out/profile_and_report.log`

## 重要开关（健壮性与可重复）
- **NVTX 非强依赖**：默认 `CAPTURE_RANGE=none`。若你给应用插了 NVTX，可设：
  - `CAPTURE_RANGE=nvtx`，并可选 `NVTX_NAME="range@domain"`（同 `nsys -p`）。
  - 若使用 NVTX 动态字符串，建议 `export NSYS_NVTX_PROFILER_REGISTER_ONLY=0` 以捕获非注册字符串。
- **重复运行/覆盖输出**：`nsys stats` 已加 `--force-export=true --force-overwrite=true`，确保从最新的 `.nsys-rep` 重新导出并覆盖同名 CSV。
- **ncu 多进程**：默认 `--target-processes all`。需要更强可比性时，可改用 `--replay-mode application` 与合适的 `--app-replay-match`。
- **ncu 指标集**：为兼容不同 ncu 版本，默认使用 `--set speed-of-light`（当 `NCU_FAST=1`）或 `--set full`，避免 section 名称差异造成错误。可通过 `NCU_SET` 覆盖。
  - 相关开关：`NCU_FAST`、`NCU_SET`、`NSYS_TRACE`、`NSYS_CAPTURE_RANGE`/`NSYS_CAPTURE_RANGE_END`、`NSYS_DURATION`、`NSYS_QUIET`。
- **缓存/频率控制**：默认 `--cache-control all --clock-control base` 以提高多次回放的一致性；若想保留“真实缓存状态”，可把 `cache-control` 改为 `none`。

## 参考（命令出处）
- Nsight Systems 用户指南（`--capture-range`/NVTX 捕获、`TMPDIR`、`stats` 报表/CSV、`--force-export` 与 `--force-overwrite` 等）。
- Nsight Compute CLI 文档（`--page raw --csv`、`--import`、`--metrics`、`--kernel-name-base/--kernel-name`、`--target-processes all`、`--cache-control`、`--clock-control`、回放与匹配等）。

> 本子仓库即用于批量采集与汇总 GPU 指标，适合在 `ncu_dir` 环境中，对 YOLO 或其它脚本进行快速端到端画像。
