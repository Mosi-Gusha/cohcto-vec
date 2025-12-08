## 项目结构

- `algorithm/`
  - `env/`: VEC 环境与图工具（`vec.py`、`ggrn.py`、`priority.py`）。
  - `cohcto/`: COHCTO 算法组件（`ppo.py`）。
  - `pd3qn/`: PD3QN 算法组件（`agent.py`、`replay_buffer.py`）。
  - `configs/`: 超参配置 `cohcto.yaml`、`pd3qn.yaml`。
  - `train.py`: 统一训练脚本，按 `--algo` 选择算法并加载对应 YAML。
  - `logs/`: 默认日志输出 (`cohcto/`, `pd3qn/`)。
- `papers/`: 两篇参考论文 PDF。
- `requirements.txt`: 依赖列表。

## 快速上手

```bash
pip install -r requirements.txt

# 统一入口（参数从 configs/*.yaml 读取）
python -m algorithm.train --algo cohcto          # COHCTO：单智能体 PPO，分层动作（缓存+卸载）
python -m algorithm.train --algo pd3qn           # 基线 PD3QN（同 VEC 环境）
# 可用 --config 指定自定义 YAML 覆盖默认配置
```

## 日志 & 输出

- 默认写入 `algorithm/logs/<algo>/` 下的 `*_metrics.csv`；若安装 pandas/matplotlib，会生成同名 `*.png`。
- 可在对应 YAML 里设置 `log_dir` 自定义输出路径。

## 备注

- 配置推荐：在 `algorithm/configs/*.yaml` 修改超参（episodes、batch_size、env 参数、奖励权重、GGRN 预训练等）；CLI 只需指定 `--algo`。
- 环境与奖励细节可在 `algorithm/env/vec.py` 调整；算法实现在 `algorithm/cohcto/`（单 PPO，分层动作）与 `algorithm/pd3qn/`。
