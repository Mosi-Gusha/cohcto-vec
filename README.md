# 项目介绍

- 复现 COHCTO 与基线 PD3QN，统一在 VEC 环境下对比，使用相同的奖励与指标。
- 单一训练入口 `python -m algorithm.train --algo {cohcto|pd3qn}`，超参集中在 `algorithm/configs/*.yaml`。
- 环境侧包含 GGRN 服务关键度预测和拓扑优先级分配，状态/奖励对两种算法一致。

# 目录结构

- `algorithm/`
  - `env/`: VEC 环境与图工具（`vec.py`、`ggrn.py`、`priority.py`）
  - `cohcto/`: COHCTO 策略组件（单智能体 PPO，联合缓存+卸载分层动作）
  - `pd3qn/`: PD3QN 策略组件（双流 DQN + 优先经验回放）
  - `configs/`: 超参配置 `cohcto.yaml`、`pd3qn.yaml`
  - `train.py`: 统一训练脚本，按 `--algo` 选择算法并加载对应 YAML
  - `logs/`: 默认日志输出目录（`cohcto/`, `pd3qn/`）
- `papers/`: 两篇参考论文 PDF
- `requirements.txt`: 依赖列表

# 快速上手

```bash
pip install -r requirements.txt

# 统一入口（参数默认来自 configs/*.yaml，可用 --config 指定自定义 YAML）
python -m algorithm.train --algo cohcto   # COHCTO：单智能体 PPO，分层动作（缓存+卸载）
python -m algorithm.train --algo pd3qn    # 基线 PD3QN（同 VEC 环境）
```

# 日志与输出

- 默认写入 `algorithm/logs/<algo>/` 下的 `*_metrics.csv`；若安装 pandas/matplotlib，会生成同名 `*.png`。
- 可在对应 YAML 里设置 `log_dir` 自定义输出路径。

# 配置与调优

- 在 `algorithm/configs/*.yaml` 修改超参（episodes、batch_size、环境参数、奖励权重、GGRN 预训练等）；CLI 只需指定 `--algo`，或用 `--config` 指定其他 YAML。
- 环境与奖励细节可在 `algorithm/env/vec.py` 调整；算法实现位于 `algorithm/cohcto/`（单 PPO，分层动作）与 `algorithm/pd3qn/`。
