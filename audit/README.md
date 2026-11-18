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
