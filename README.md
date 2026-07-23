# MAML-ANODE

面向量子弱测量轨迹建模的元学习实验仓库。项目围绕 `TADE-AQNODE` 展开，目标是从支持集轨迹中推断任务参数 `(alpha, r)`，并基于物理约束重建 Bloch 动力学。

## 1. 项目在做什么

这个仓库提供了一条完整实验链路：

1. 生成带物理参数标签的轨迹任务数据。
2. 用 `Meta-AQNODE` / `TADE-AQNODE` 对多个任务进行元训练。
3. 在测试任务上输出参数误差、动力学误差和可视化图。
4. 对多次实验 run 做聚合分析，比较不同超参数或结构设置。

核心思想是把物理先验嵌入模型训练流程：

- `data_gen.py` 负责生成 TCL/Bloch 动力学轨迹任务。
- `models.py` 定义物理约束模型和元学习训练器。
- `train.py` 串起训练、测试、分析和可视化。
- `analysis.py`、`visualization.py`、`analyze_runs.py` 负责结果诊断与汇总。

## 2. 仓库主要文件

### 核心训练链路

- `data_gen.py`
  - 底层数据生成模块。
  - 给定 `(alpha, r)` 生成单个任务的多条轨迹，包含 `bloch`、`delta`、`gamma`、`dY`、`t` 等字段。

- `export_train_trajectory_dataset.py`
  - 训练集导出入口。
  - 在给定参数区间内按网格或随机方式采样任务，并把每个任务保存成一个 `.pt` 文件。

- `export_test_trajectory_dataset.py`
  - 测试集导出入口。
  - 用更宽的参数范围导出测试/OOD 任务。

- `models.py`
  - 模型与训练器核心。
  - 包含物理参数化模块、支持集编码器、内外环优化逻辑和测试预测逻辑。

- `train.py`
  - 主入口。
  - 负责读取导出的任务文件、划分 train/val、执行训练、保存测试结果，并自动调用分析与作图。

### 结果分析与可视化

- `analysis.py`
  - 为单次 run 生成结构化分析结果。
  - 输出 `summary.csv`、`training_history.json/csv`、`analysis/report.json`、`analysis/report.md` 等。

- `visualization.py`
  - 为单次 run 生成训练曲线、任务排名、参数诊断图、Bloch 球图和逐任务轨迹图。

- `visualize_dy.py`
  - 不依赖训练，直接对导出的任务数据做 `dY` 可视化。
  - 适合检查训练前数据分布、不同 `(alpha, r)` 的弱测量信号差异。

- `parameter_space_plot.py`
  - 画测试任务在 `(alpha, r)` 参数空间中的预测偏移情况。

- `analyze_runs.py`
  - 扫描多个 run 目录并做聚合分析。
  - 适合比较不同超参数、不同结构或不同损失权重设置。

### 实验记录

- `experiment_logger.py`
  - 统一记录 run 配置、数据集摘要、训练摘要、测试摘要和耗时信息。
  - 会在每个 run 目录下输出 `experiment_log.jsonl` 和 `experiment_log.md`。

## 3. 推荐使用顺序

如果你第一次使用这个仓库，建议严格按下面顺序操作。

### 第一步：生成训练任务

先导出训练任务数据：

```bash
python export_train_trajectory_dataset.py \
  --output-dir exported_tasks \
  --alpha-min 0.4 --alpha-max 0.7 \
  --r-min 0.2 --r-max 0.5 \
  --alpha-points 10 --r-points 10 \
  --num-traj 2000 \
  --overwrite
```

这一步会生成：

- `exported_tasks/task_alpha_...__r_....pt`
- `exported_tasks/manifest.json`
- `exported_tasks/manifest.csv`

这些文件是训练入口 `train.py` 的直接输入。

### 第二步：生成测试任务

再导出测试任务数据：

```bash
python export_test_trajectory_dataset.py \
  --output-dir exported_test_tasks \
  --alpha-min 0.2 --alpha-max 0.8 \
  --r-min 0.1 --r-max 0.6 \
  --num-random-points 100 \
  --num-traj 256 \
  --overwrite
```

这一步通常使用更宽的参数范围，用于测试插值与 OOD 泛化能力。

### 第三步：可选地先看数据质量

如果你想先检查 `dY` 信号和任务分布，再开始训练，可以运行：

```bash
python visualize_dy.py exported_tasks --max-tasks 16
```

它会输出：

- 单任务 `dY` 图
- 多任务均值对比图
- 均值/标准差对比图
- 分位数对比图
- `dy_visualization_manifest.json`

这一步适合确认：

- 参数采样是否合理
- 任务之间是否有足够可分性
- 弱测量信号是否存在明显塌缩或重叠

### 第四步：启动训练

准备好训练集和测试集后，运行主训练脚本：

```bash
python train.py \
  --train-data-dir exported_tasks \
  --test-data-dir exported_test_tasks \
  --num-epochs 50 \
  --tasks-per-epoch 5 \
  --device cuda:0
```

常用可调参数：

- `--num-epochs`：训练轮数
- `--tasks-per-epoch`：每轮采样多少训练任务
- `--outer-lr`：外环学习率
- `--inner-lr`：内环学习率
- `--inner-steps`：内环更新步数
- `--seq-len`：训练使用的时间窗长度
- `--w-alpha`、`--w-r`、`--w-alpha-frac`、`--w-r-frac`：参数损失相关权重
- `--val-ratio`：从训练任务中划出验证集的比例
- `--run-root`：run 根目录，默认是 `./results`
- `--run-name`：手动指定当前实验名称

默认情况下，训练结果会保存在：

```text
results/run_XXX/
```

其中包括：

- `config.json`
- `train_curriculum.json`
- `training_history.json`
- `training_history.csv`
- `test_results.json`
- `summary.csv`
- `experiment_log.jsonl`
- `experiment_log.md`
- `analysis/report.json`
- `analysis/report.md`
- `meta_aqnode_results/*.png`

### 第五步：查看单次 run 结果

完成训练后，优先查看下面几类结果：

- `summary.csv`
  - 每个测试任务一行，包含 `loss`、`bloch_mse`、`mse_x/y/z`、`err_alpha`、`err_r` 等指标。

- `analysis/report.md`
  - 单次 run 的文字摘要，适合快速浏览。

- `analysis/report.json`
  - 结构化分析结果，适合后续脚本再处理。

- `meta_aqnode_results/training_loss.png`
  - 训练损失曲线。

- `meta_aqnode_results/bloch_task_*.png`
  - 每个测试任务的 Bloch 球真值/预测轨迹对比。

- `meta_aqnode_results/xyz_task_*.png`
  - 每个测试任务的分量级轨迹与 `delta/gamma` 诊断图。

### 第六步：对多次实验做聚合分析

当 `results/` 下面已经积累了多次实验后，可以运行：

```bash
python analyze_runs.py --roots results --output-dir run_analysis
```

常用参数：

- `--roots`：扫描哪些 run 根目录
- `--output-dir`：聚合分析输出目录
- `--metrics`：纳入统计和作图的指标
- `--group-by`：按哪些字段做分组分析
- `--include-partial`：是否把不完整 run 也纳入
- `--limit-runs`：仅分析前 N 个 run

输出包括：

- `run_analysis/run_catalog.csv`
- `run_analysis/aggregate_report.json`
- `run_analysis/aggregate_report.md`
- 多张聚合统计图

## 4. 代码使用逻辑

如果你想理解代码，而不是只会跑命令，建议按下面顺序阅读：

1. `data_gen.py`
   - 先理解单任务数据是什么、每个字段代表什么。

2. `export_train_trajectory_dataset.py` 和 `export_test_trajectory_dataset.py`
   - 再理解任务文件是如何组织出来的。

3. `models.py`
   - 先看物理参数化，再看支持集编码器，最后看 `MetaTrainer`。

4. `train.py`
   - 把“读数据 -> 训练 -> 测试 -> 保存分析”的主流程串起来。

5. `analysis.py` 和 `visualization.py`
   - 理解每个 run 最终会产出哪些诊断指标和图。

6. `analyze_runs.py`
   - 理解多 run 比较是怎么做的。

## 5. 最常见工作流

### 工作流 A：从零开始跑一个实验

```bash
python export_train_trajectory_dataset.py --overwrite
python export_test_trajectory_dataset.py --overwrite
python train.py --device cuda:0
```

### 工作流 B：先看数据是否可分，再训练

```bash
python export_train_trajectory_dataset.py --overwrite
python visualize_dy.py exported_tasks --max-tasks 16
python train.py --device cuda:0
```

### 工作流 C：批量比较多个实验 run

```bash
python analyze_runs.py --roots results --output-dir run_analysis
```

## 6. 输出目录说明

### 数据导出目录

- `exported_tasks/`
  - 训练任务文件目录。

- `exported_test_tasks/`
  - 测试任务文件目录。

### 单次实验目录

- `results/run_XXX/`
  - 单次训练与测试的完整输出目录。

- `results/run_XXX/meta_aqnode_results/`
  - 单次 run 的主要图像结果目录。

- `results/run_XXX/analysis/`
  - 单次 run 的结构化分析目录。

### 聚合分析目录

- `run_analysis/`
  - 跨多个 run 的统计结果与对比图目录。

## 7. 环境依赖

当前代码直接依赖以下 Python 库：

- `torch`
- `numpy`
- `scipy`
- `matplotlib`

建议使用 Python 3.10 或兼容版本，并先安装上述依赖。

## 8. 使用建议

- 先只改一类变量再做实验，例如先改损失权重，再改编码器结构，避免多变量同时变化导致因果不清。
- 每次训练后优先看 `summary.csv`、`analysis/report.md` 和 `experiment_log.md`。
- 比较不同配置时，尽量保留同样的数据导出范围和随机种子设置。
- 不要把实验产物和缓存目录提交到 Git，例如 `results/`、`meta_aqnode_results/`、`__pycache__/`。

## 9. 一句话总览

如果只记住一条主线，可以记成：

```text
导出 train/test 任务 -> 可选检查 dY -> 运行 train.py -> 查看单次 run 诊断 -> 用 analyze_runs.py 做多实验比较
```
