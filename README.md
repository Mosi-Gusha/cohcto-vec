# 项目介绍

- 复现 COHCTO 与基线 PD3QN，统一在 VEC 环境下对比，使用相同的奖励与指标。
- 单一训练入口 `python -m algorithm.train --algo {cohcto|pd3qn}`，超参集中在 `algorithm/configs/*.yaml`。
- 环境侧包含 GGRN 服务关键度预测和拓扑优先级分配，状态/奖励对两种算法一致。
- 提供对比绘图脚本 `algorithm/plot_compare.py`，将两种算法指标绘制在同一张图上。

# 目录结构

- `algorithm/`
  - `env/`: VEC 环境与图工具（`vec.py`、`ggrn.py`、`priority.py`）
  - `cohcto/`: COHCTO 策略组件（单智能体 PPO，联合缓存+卸载分层动作）
  - `pd3qn/`: PD3QN 策略组件（双流 DQN + 优先经验回放）
  - `configs/`: 超参配置 `cohcto.yaml`、`pd3qn.yaml`
  - `train.py`: 统一训练脚本，按 `--algo` 选择算法并加载对应 YAML
  - `plot_compare.py`: COHCTO vs PD3QN 指标对比绘图
  - `logs/`: 默认日志输出目录（`cohcto/`, `pd3qn/`）
- `papers/`: 两篇参考论文 PDF
- `requirements.txt`: 依赖列表

# 快速上手

```bash
pip install -r requirements.txt

# 统一入口（参数默认来自 configs/*.yaml，可用 --config 指定自定义 YAML）
python -m algorithm.train --algo cohcto   # COHCTO：单智能体 PPO，分层动作（缓存+卸载）
python -m algorithm.train --algo pd3qn    # 基线 PD3QN（同 VEC 环境）

# 对比绘图（两算法指标同图）
python -m algorithm.plot_compare
```

# 日志与输出

- 默认写入 `algorithm/logs/<algo>/` 下的 `*_metrics.csv`；若安装 pandas/matplotlib，会生成同名 `*.png`。
- 指标 CSV 额外包含奖励分项（`r_cache`, `r_cost`, `r_success`, `r_drop`），便于分析奖励结构。
- 对比图默认输出到 `algorithm/logs/compare_metrics.png`。
- 可在对应 YAML 里设置 `log_dir` 自定义输出路径。

# MDP 建模与实现（面向非专业读者）

这部分用更直观的话说明：我们把系统看成一个"会反复交互的环境"。每一回合，系统给出当前情况（状态），算法做决定（动作），环境反馈好坏（奖励），然后进入下一个状态。

## COHCTO（PPO）

- 目标问题：在车、路侧 RSU 和云三层里，决定"把任务放到哪里算"以及"要不要提前缓存服务"，让任务尽量按时完成、缓存命中率高，同时少耗时、少耗能。
- 状态 S（环境告诉算法的"当前情况"）：
  - 当前时间步（归一化）。
  - 当前任务的截止时间和输入大小。
  - 任务依赖的简单特征（有多少父/子任务）。
  - 车与最近 RSU 的距离、覆盖情况、回传拥塞。
  - 当前任务需要的服务（one-hot）。
  - GGRN 预测的服务关键度（哪些服务更"关键"）。
  - RSU/车辆缓存里有哪些服务。
  - 各节点队列负载（忙不忙）。
  - `lambda_cache` 和 RSU 是否存活。
  这些由 `algorithm/env/vec.py::_build_state` 生成。
- 动作 A（算法做的决定）：
  - 选择卸载目标：车本地、某个 RSU、或云。
  - 可选地，把当前服务主动缓存到某个 RSU。
  - 由 `algorithm/env/vec.py::_decode_action` 解码，实际执行在 `step` 中。
- 奖励 R（做得好不好）：
  - 命中缓存有奖励。
  - 时延和能耗会扣分（越慢越耗电扣得越多）。
  - 任务按时完成有奖励。
  - 掉线/失败额外扣分。
  由 `algorithm/env/vec.py::step` 计算并返回。
- 转移过程（一步发生了什么）：
  - 环境按动作执行卸载和缓存。
  - 更新队列负载、车辆位置、GGRN 的时间窗口。
  - 输出下一状态。
  - 在 `algorithm/train.py::train_cohcto` 中收集一整段轨迹，再由 `algorithm/cohcto/ppo.py` 统一更新策略。

## P-D3QN（Double+Dueling DQN + PER）

- 目标问题：和 COHCTO 相同，只是用 DQN 系列算法作为对比基线。
- 状态/动作/奖励：完全复用同一个 MDP（同一环境、同一状态、同一动作、同一奖励），保证可比性。
- 学习过程：
  - 每一步把 (状态, 动作, 奖励, 下一状态) 存进经验池（PER）。
  - 用 Double Q 避免过估计，用 Dueling 结构分离"状态价值"和"动作优势"。
  - 反复从经验池采样更新 Q 网络。
  - 逻辑在 `algorithm/train.py::train_pd3qn` 和 `algorithm/pd3qn/agent.py`。

# 配置与调优

- 在 `algorithm/configs/*.yaml` 修改超参（episodes、batch_size、mini_batch_size、PPO actor/critic lr、clip_eps、环境参数、奖励权重、GGRN 预训练等）；CLI 只需指定 `--algo`，或用 `--config` 指定其他 YAML。
- 环境与奖励细节可在 `algorithm/env/vec.py` 调整；算法实现位于 `algorithm/cohcto/`（单 PPO，分层动作）与 `algorithm/pd3qn/`。
