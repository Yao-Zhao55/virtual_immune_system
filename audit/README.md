# audit/ — 复核与审计产物

## phase_a/（2026-11-19，冻结预测复核）

从深圳 ECS 冻结的逐细胞折外预测（`runs/full_P*`、`runs/bn_P*`，seed 17）重新计算全部指标，
与 `results/` 归档表逐项比对。

- **完整性**：full 784,064 行 = 196,016 细胞 × 4 模型；bn 392,032 行 = × 2 模型；无重复、无缺失。
- **一致性**：66 项患者级 macro-F1 与归档值全部一致。
- **A2 差异定位**：support≥30 复算值为 MLP 0.9281 / Transformer 0.9197，与
  `results/sensitivity_full.txt` 一致；原稿引用的 0.9279/0.9196 与两者均不符（差 0.0001–0.0002），
  以本目录复算值为准，详见 `discrepancies.tsv`。
- 输入哈希清单见 `manifest.tsv`；内外层患者划分见 `splits.tsv`；指标口径见 `protocol.yaml`。

> 权重与逐细胞预测本体不入库；已归档至 `oss://immune-system-data/gse184362/runs/`（432 对象，MD5 校验通过）。

## phase_b3/（2026-11-20，多种子稳健性，seeds 17/42/2026）

11 折 × 3 种子完整协议（每种子独立内层选择）。seed 17 产物来自 `runs/full_P*`，
seeds 42/2026 来自 11-19~20 在 A10 上的重训（`runs_ms/full_P*`，协议冻结一致）。

- **主比较 MLP − Transformer**：Δ = +0.0133，9/11 患者为正，bootstrap 95% [+0.0041, +0.0239]，
  三个种子方向一致 → 支持当前队列/协议下的稳健趋势（幅度约 1.3 pt，注意患者异质性）。
- 三种子均值：MLP 0.9242 > Logistic 0.9161 > Transformer 0.9109 > Deep Sets 0.8789。
- Logistic 三种子完全相同（确定性求解器）；Deep Sets 种子间波动最大（跨幅 2.8 pt）。
- 聚合顺序（B3）：患者内跨种子平均 → 患者配对差值 → 11 患者等权平均。
- 权重与逐细胞预测不入库：`oss://immune-system-data/gse184362/runs/`（seed 17）
  与 `oss://immune-system-data/gse184362/runs_ms/`（seeds 42/2026）。
