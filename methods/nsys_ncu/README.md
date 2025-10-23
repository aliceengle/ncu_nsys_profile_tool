# nsys_ncu 方法说明

本方法目录提供以 Bash 为主、Python 辅助的 Nsight Systems（nsys）+ Nsight Compute（ncu）采集方案，聚焦于稳定、可复现地抓取 Top‑N kernel 的关键性能指标，并生成便于二次分析的 CSV。

## 目录概览
- `profile_and_report.sh`：方法的主入口（Bash）。
  - 读取上上级目录的 `scripts/bash_metric/targets.tsv`。
  - 输出到上上级目录的 `scripts/bash_metric/out/`。
  - 统一全局日志：`out/profile_and_report.log`（每次运行覆盖重写）。
- `gen_topk_regex.py`：从 nsys 的 `*_cuda_gpu_kern_sum.csv` 选 Top‑N kernel 并生成安全正则（供 ncu 匹配）。

## 作用与设计
- 作用：端到端采集指定目标的 GPU 性能画像，覆盖 kernel 耗时（nsys）与核心指标（ncu）。
- 设计要点：
  - Bash 专注串接 nsys/ncu；复杂 CSV/正则生成、健壮性检查由 Python 辅助完成。
  - 阶段化进度打印（6 步）+ 单一全局日志，便于复盘与排查。
  - Top‑N kernel 支持“单次运行”与“逐次运行”两种匹配模式，兼顾速度与覆盖率。
  - 导出 ncu raw CSV 时仅捕获 stdout，避免日志污染；含基本 CSV 健康度检查。

## 运行环境
- 需要在 PATH 中可调用：`nsys`、`ncu`。
- `python3`（用于正则生成与小型解析辅助）。

## 目标列表（targets.tsv）
- 位置：`scripts/bash_metric/targets.tsv`
- 格式（TAB 分隔四列）：`NAME<TAB>CMD<TAB>WORKDIR<TAB>ENVS`
- 示例：
  - `yolo_infer	python ../yolo_infer.py --nvtx	.	-`
- 约定：
  - `WORKDIR`、`ENVS` 可为空或 `-`；`ENVS` 多个变量用逗号分隔，如 `CUDA_VISIBLE_DEVICES=0,UL_DEVICE=0`。
  - `ENVS` 为空或 `-` 时不传入 `env`，避免破坏 PATH。

## 使用方法
- 在 `scripts/bash_metric` 目录下执行：
  - `bash methods/nsys_ncu/profile_and_report.sh`
  - 或用顶层包装器：`bash profile_and_report.sh`
- 常用环境变量：
  - `TARGETS_TSV`：指定目标表路径（默认 `../../targets.tsv`）。
  - `OUT_DIR`：指定输出根目录（默认 `../../out`）。
  - `GLOBAL_LOG`：指定全局日志文件路径（默认 `out/profile_and_report.log`）。
  - `BASH_BIN`：指定 Bash 解析器（默认 `/bin/bash`）。
  - `NCU_TOPN`：Top‑N kernel 数量（默认 `5`）。
  - `NCU_MULTI_RUN`：多次运行模式（默认 `1`）。
    - `1`：逐核运行，每个 kernel 生成独立 `.ncu-rep` 与 `*_raw.csv`。
    - `0`：单次运行合并正则，仅生成一份 `.ncu-rep` 与 `*_raw.csv`（更快但可能遗漏）。
  - `NCU_FAST` 与 `NCU_SET`：采集集合选择。
    - `NCU_FAST=1` → `NCU_SET=speed-of-light`（较快的集合）。
    - 其它 → `NCU_SET=full`（更全面）。也可直接设置 `NCU_SET` 覆盖。
  - `NCU_LAUNCH_COUNT`：每个匹配 kernel 的回放次数（默认 `1`）。
  - `NSYS_TRACE`、`NSYS_CAPTURE_RANGE`、`NSYS_CAPTURE_RANGE_END`、`NSYS_DURATION`、`NSYS_QUIET`：控制 nsys 行为。

## 输出内容
- `out/<NAME>/nsys/`：
  - `<NAME>.nsys-rep`、`<NAME>.sqlite`、`<NAME>_cuda_gpu_kern_sum.csv` 等。
- `out/<NAME>/ncu/`：
  - 单次运行：`<NAME>.ncu-rep`、`<NAME>_raw.csv`。
  - 多次运行：`<NAME>__<kernel-sanitized>.ncu-rep`、`<NAME>__<kernel-sanitized>_raw.csv` 多份。
- `out/profile_and_report.log`：统一全局日志（覆盖写），含阶段进度与运行配置打印。

## 阶段流程（进度打印）
1) 准备目录与上下文
2) nsys 采集
3) nsys 导出 CSV
4) 生成 Top‑N kernel 正则
5) ncu 采集
6) ncu 导出 raw CSV（单次运行模式时）

每步打印 `[TARGET <name>] [STEP i/6]` 开始/完成/失败，以及下一步提示。

## 设计细节
- 正则生成（Python）：从 nsys 的 kernel 汇总 CSV 中，按耗时/时间列兼容性策略挑选 Top‑N，使用 `re.escape` 生成安全正则串联。
- ncu 采集集合：使用 `--set speed-of-light/full` 提升版本兼容性，避免 `--section` 名称差异导致错误。
- CSV 导出与检查：通过单独函数捕获 stdout 到文件，并对首行分隔符与日志污染做基本检查。
- 错误处理：每阶段失败打印并依据 `CONTINUE_ON_ERROR` 决定是否继续。

## 常见问题与排查
- “bash 不存在或不可执行”：设置 `BASH_BIN=/bin/bash` 或正确的 bash 路径。
- “nsys/ncu 不在 PATH”：确保工具安装并在 PATH 中，或用绝对路径调用。
- “只分析一个 kernel”：使用 `NCU_MULTI_RUN=1` 改为逐核运行；或增大 `NCU_LAUNCH_COUNT`。
- “raw CSV 非 CSV”：检查日志，确保导出使用的函数为捕获 stdout 的模式；确认 CSV 首行含逗号或分号分隔。
- “targets.tsv 未找到”：方法脚本默认读取 `../../targets.tsv`；从 `scripts/bash_metric` 目录运行，或指定 `TARGETS_TSV`。

## 示例命令
- 多次运行（逐个 kernel）：
  - `NCU_MULTI_RUN=1 NCU_TOPN=5 bash methods/nsys_ncu/profile_and_report.sh`
- 单次运行（合并正则）：
  - `NCU_MULTI_RUN=0 NCU_TOPN=5 bash methods/nsys_ncu/profile_and_report.sh`
- 自定义解析器与输出：
  - `BASH_BIN=/usr/bin/bash OUT_DIR=/tmp/out GLOBAL_LOG=/tmp/run.log bash methods/nsys_ncu/profile_and_report.sh`

