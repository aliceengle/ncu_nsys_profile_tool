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
- `bash_metric/`：Python 包源码
  - `run_profiling.py`：端到端采集（nsys + ncu）主流程，提供 `bash-metric-run` CLI。
  - `parse_and_plot.py`：解析 ncu/nsys CSV 并绘制图表，提供 `bash-metric-parse` CLI。
  - `methods/nsys_ncu_all/`：全流程捕获与 Top-N 解析脚本（Python + Bash），提供 `bash-metric-capture-all` 与 `bash-metric-analyze-topk` CLI。
  - `targets.tsv`：默认目标列表示例，安装后可作为模板复制修改。
  - `fonts/`：内置常用中文字体（若系统缺字形时自动注册）。
- `requirements.txt`：与 pixi 默认环境一致的依赖清单，可供手工安装参考。
- `out/`：示例输出目录（包安装后不会自动创建，可通过环境变量自定义）。
- `.github/workflows/publish-pypi.yml`：GitHub Actions 工作流，根据打出的版本标签自动构建并发布到 PyPI（需要在仓库 Secrets 中配置 `PYPI_API_TOKEN`）。

## 依赖
建议使用 `pixi` 默认环境（`python 3.10` + Nsight CLI）。如单独安装，可执行：
```bash
pip install bash-metric
```
运行时仍需确保系统已安装 `nsys` 与 `ncu` 并可在 PATH 中找到。

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
**推荐：安装后直接使用 CLI**
```bash
bash-metric-run --targets /path/to/targets.tsv --topn 10
```
完成采集后，可单独解析历史结果：
```bash
bash-metric-parse /path/to/out_group --log /path/to/out_group/run.log
```
若只想执行捕获阶段或后处理阶段，可分别调用：
```bash
bash-metric-capture-all --targets xxx.tsv
bash-metric-analyze-topk --targets xxx.tsv
```

**源码运行（任选）**
```bash
python -m bash_metric.run_profiling
python -m bash_metric.parse_and_plot ./out --log ./out/run.log
```

运行后默认在 `out/` 会看到：
- `nsys/<id>.nsys-rep` 与 `<id>_cuda_gpu_kern_sum.csv`（Kernel 耗时/占比）；
- `ncu/<kernel>/*.ncu-rep` 与 `*_ncu_raw.csv`、`*_metrics_id.csv`；
- `profile_and_report.log`：单一全局日志，记录所有目标的全部阶段输出，重复执行会覆盖重写；
- `summary_*.csv/.md` 与 `figs/*.png`。

## 运行阶段与进度打印
`bash-metric-run` 会将每个目标的执行划分为 6 个阶段，并打印进度：
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

附加开关（方法脚本）：
- `NCU_MULTI_RUN`：1（默认）逐核运行，多份 `.ncu-rep` 与 `*_raw.csv`；0 单次运行合并正则，生成一份 `.ncu-rep` 与 `*_raw.csv`。
- `BASH_BIN`：指定 Bash 解析器（默认 `/bin/bash`），用于应用命令的 `-lc` 执行。

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

## 发布到 PyPI（自动化流水线）
1. 在仓库 Settings → Secrets and variables → Actions 中新增 `PYPI_API_TOKEN`，值为 PyPI 生成的 API Token，用户名固定为 `__token__`。  
2. 更新 `scripts/bash_metric/pyproject.toml` 的 `version`，提交并推送。  
3. 打标签（支持 `v1.2.3` 或 `bash-metric-v1.2.3` 等格式）并推送：`git tag v1.2.3 && git push origin v1.2.3`。  
4. GitHub Actions 工作流 `.github/workflows/publish-pypi.yml` 会自动构建并上传 sdist/wheel 到 PyPI。  
5. 在 Actions 详情页确认 `Publish to PyPI` job 成功，稍候 1～2 分钟即可在 PyPI 下载对应版本。

> 本子仓库即用于批量采集与汇总 GPU 指标，适合在 `ncu_dir` 环境中，对 YOLO 或其它脚本进行快速端到端画像。
