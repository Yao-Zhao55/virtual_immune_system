# virtual_immune_system · GSE184362 单细胞注释对照实验

用 Transformer / MLP / Deep Sets / Logistic 四种模型，在 **GSE184362**（甲状腺癌与癌旁，11 名患者、23 份样本）上做细胞类型注释，
采用**按患者留一（leave-one-patient-out）的嵌套交叉验证**，比较注意力架构相对线性与集合模型的增益。

> 状态：11 折正式实验运行中（2026-11-16 20:52 起），结果表将在全部折完成后写入 `results/`。

## 1. 数据

| 项 | 值 |
|---|---|
| 来源 | GEO [GSE184362](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE184362)，`GSE184362_RAW.tar` 925.8 MB |
| 样本 | 23 个 GSM / 11 名患者（原发灶、癌旁、淋巴结转移、皮下转移） |
| 细胞 | 原始 197,954 条条形码，QC 后 196,207，建模 196,016 |
| 基因 | 各样本 10x 矩阵取交集，33,694 个 gene ID |
| 类别 | T_NK / Epithelial / B / Myeloid / Fibroblast / Endothelial / Plasma，7 类 |

### 标签来源（重要）

GEO 只提供原始矩阵，作者仓库（[puweilin/scRNAseq_PTC](https://github.com/puweilin/scRNAseq_PTC)）只有 marker 代码、没有逐细胞标签表。
因此标签是**可审计的参考注释**，不是病理金标准：

```
QC(≥200 且 ≤6000 基因, 线粒体 ≤20%)
  → normalize_total(1e4) → log1p
  → HVG 2000（按 sample_id 分批）→ PCA 50 → neighbors(15) → Leiden(res=1.0, 41 簇)
  → 8 个 marker 模块打分（T_NK/B/Plasma/Myeloid/Mast/Endothelial/Fibroblast/Epithelial）
  → 簇级指派：最优模块须领先次优 ≥0.15，否则标 unlabeled
```

簇级指派与各模块分数见 `data/annotation_clusters.tsv`，可复核每一步。
每个类别在 11 名患者中均出现，满足「每类至少 3 名患者」的约束。

**所有下游指标衡量的是对该注释的复现能力**，不能解释为临床级细胞分型。

## 2. 方法

- **外层**：11 折，按患者留一，同一患者的所有组织同折；
- **内层**：外层训练集内再分 3 折，用于选学习率/正则与早停轮数，每个内层训练折独立筛选基因；
- **特征**：按原始总 UMI 归一化到 10,000 → `log1p` → 取训练折 HVG 前 512 个（保留零表达，不中心化）；
- **模型**：LogisticRegression（C=0.1/1.0）、2 层 MLP（512-256）、Deep Sets（基因嵌入 + 逐基因前馈 + 均值池化）、
  4 层 Pre-LN Transformer（宽度 128、4 头、无位置编码、均值池化）；
- **训练**：AdamW、交叉熵、weight decay 0.01、dropout 0.1，最多 30 epoch、早停 patience 5，batch 512，bf16 autocast，`torch.compile`；
- **指标**：患者等权 macro-F1（主）、总体准确率、平衡准确率、支持≥30 类别的 macro-F1（敏感性）、配对组织组成差。

## 3. 复现

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r code/requirements.txt

# 1) 下载并解包 GEO（约 926 MB）
python code/fetch_tar.py
python code/extract_tar.py

# 2) 重建样本表与参考标签
python code/build_manifests.py
python code/build_annotation.py --resolution 1.0 --min-margin 0.15

# 3) 生成建模用 AnnData
python code/prepare_data.py --samples data/samples.tsv --labels data/labels.tsv --output-dir runs/prepared

# 4) 单折（或全部 11 折）
python code/train_evaluate.py --input runs/prepared/cohort.h5ad --output-dir runs/full_P01 \
  --pilot-patient P01 --max-epochs 30 --patience 5 --batch-size 512 --inner-folds 3 \
  --n-genes 512 --amp bf16 --compile --device cuda
```

云端编排脚本在 `code/cloud/`（`install.sh` 建 venv、`run_worker.sh` 跑指定折、`progress.sh` 出进度、`restart.sh` 重启）。

## 4. 结果（11 折已完成，2026-11-17）

### 4.1 患者等权 macro-F1（全部类别，主指标）

| 留出患者 | Logistic | MLP | Deep Sets | Transformer |
|---|---|---|---|---|
| P01 | 0.8797 | 0.8825 | 0.8888 | **0.9018** |
| P02 | 0.8753 | 0.8544 | 0.7459 | 0.8220 |
| P03 | 0.9604 | **0.9645** | 0.8949 | 0.9449 |
| P04 | 0.9062 | **0.9457** | 0.8983 | 0.8920 |
| P05 | 0.9331 | **0.9428** | 0.9106 | 0.9359 |
| P06 | 0.8963 | **0.9018** | 0.7374 | 0.8660 |
| P07 | 0.9811 | 0.9822 | 0.9682 | 0.9796 |
| P08 | 0.9726 | **0.9775** | 0.9617 | 0.9704 |
| P09 | 0.9887 | **0.9897** | 0.9753 | 0.9758 |
| P10 | 0.7025 | 0.7126 | 0.6900 | **0.7346** |
| P11 | 0.9808 | 0.9847 | 0.9808 | **0.9870** |
| **均值** | 0.9161 | **0.9217** | 0.8774 | 0.9100 |

**Transformer − MLP：均值 −1.17 pt**，逐患者 +1.94 / −3.24 / −1.97 / −5.37 / −0.70 / −3.57 / −0.26 / −0.71 / −1.39 / **+2.20** / **+0.22**
→ **11 折中仅 3 名患者改善**（P01、P10、P11）。

### 4.2 指标敏感性（同一批折外预测重算）

| 指标 | Logistic | MLP | Deep Sets | Transformer | Transformer−MLP |
|---|---|---|---|---|---|
| macro-F1（全部类别） | 0.9161 | **0.9217** | 0.8774 | 0.9100 | −1.17 pt |
| macro-F1（每患者支持≥30 的类别） | 0.9548 | **0.9621** | 0.9319 | 0.9505 | −1.17 pt |
| 平衡准确率（macro recall） | 0.9478 | 0.9522 | 0.9198 | **0.9529** | +0.07 pt |

### 4.3 结论：未达到预注册门槛

预注册判据是「Transformer 相对 MLP 患者等权 macro-F1 至少高 2 个百分点，且多数患者改善」。实测为 **−1.17 pt、3/11 患者改善**，两条都不满足。

- Transformer 在**平衡准确率上与 MLP 持平**（0.9529 vs 0.9522），说明它的类别召回并不差，差距来自精确率（尤其髓系与上皮的混淆）；
- 它在类别分布均衡的患者（P07/P08/P09/P11，macro-F1 均 >0.97）上与 MLP 差距 ≤0.7 pt，在**类别极端偏斜的患者**（P02/P04/P06）上掉 3–5 pt——即**方差大、稳健性差**，而不是整体更弱；
- 唯一大幅获益的是 P10（+2.20 pt），该患者留出细胞最多、类别最不均衡，Transformer 在稀有类上更好，但不足以改变总体结论；
- Logistic 与 MLP 相差仅 0.56 pt，说明这套 512 维 HVG 表达特征本身线性可分性很强，注意力带来的收益空间本就有限。

每折明细见 `results/full_P01` … `results/full_P11`，汇总见 `results/aggregate_full.txt` 与 `results/sensitivity_full.txt`。

### 4.4 计算成本（单张 A10，bf16 + torch.compile）

| 模型 | 中位 epoch | 单次正式拟合 | 峰值显存 |
|---|---|---|---|
| Logistic | — | 14.9 s | CPU |
| MLP | 10 | 6.2 s | 0.02 GB |
| Deep Sets | 19 | 326 s | 3.68 GB |
| Transformer | 14 | 486 s | 4.20 GB |

11 折串行总耗时约 16.5 小时（单折 1.3–1.9 小时）。

## 5. 限制

- 标签为参考注释，非独立病理标注；结论是「对该注释的可复现性」；
- 单队列、11 名患者，患者级配对重采样只是描述性区间；
- 部分患者的某些类别细胞数极少（如 P02 上皮类仅 1 个），患者等权 macro-F1 会受小样本类别扰动，故并列报告支持度过滤与平衡准确率；
- 原始矩阵与模型权重不入库（体积原因），可用第 3 节脚本重建；
- 研究用途，不用于临床诊断或治疗决策。
