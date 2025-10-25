# VCDatasets 数据集概览

本文档详细说明了各个数据集包含的模态、任务类型和数据规模。

---

## 数据集列表

### 1. Mol-Instructions

**来源**: `zjunlp_mol_instructions`  
**论文**: [Mol-Instructions: A Large-Scale Biomolecular Instruction Dataset](https://github.com/zjunlp/Mol-Instructions)

#### 模态
- **Protein** (蛋白质序列)

#### 任务类型
从 `Protein-oriented_Instructions.zip` 中提取的任务：

1. **catalytic_activity.json**
   - 催化活性预测
   - 预测蛋白质的催化反应和EC编号

2. **domain_motif.json**
   - 结构域和motif识别
   - 识别蛋白质中的功能结构域和序列motif

3. **general_function.json**
   - 通用功能预测
   - 预测蛋白质的一般生物学功能

4. **protein_function.json**
   - 蛋白质功能注释
   - 详细的功能描述和注释

#### 数据格式
```json
{
  "instruction": "预测以下蛋白质序列的功能...",
  "input": "MKVLWAALLVTFLAGCQAKV...",
  "output": "该蛋白质属于...功能是...",
  "metadata": {
    "split": "train/test"
  }
}
```

#### 数据规模
- 训练集和测试集混合
- 预计数万条蛋白质-功能对

---

### 2. UniProtQA

**来源**: `pharmolix_uniprotqa`  
**描述**: 基于UniProt数据库的蛋白质问答数据集

#### 模态
- **Protein** (蛋白质序列)

#### 任务类型
基于蛋白质序列的多方面问答：

1. **功能问答**
   - "What is the function of this protein?"
   - 蛋白质的生物学功能描述

2. **命名问答**
   - "What are the official names of this protein?"
   - 蛋白质的标准名称和别名

3. **家族归属**
   - "What is the protein family that this protein belongs to?"
   - 蛋白质家族分类

4. **亚细胞定位**
   - "What are the subcellular locations of this protein?"
   - 蛋白质在细胞中的定位

5. **其他属性**
   - 蛋白质结构、修饰、相互作用等

#### 数据格式
```json
{
  "protein_id": {
    "sequence": "MSEYIRVTEDENDEP...",
    "data": [
      ["问题1", "答案1"],
      ["问题2", "答案2"],
      ...
    ]
  }
}
```

#### 数据规模
- 训练集: train.json
- 测试集: test.json, test_0.json ~ test_7.json
- 包含数千个蛋白质，每个蛋白质多个问答对
- 预计数万条问答对

---

### 3. Pika-DS

**来源**: `emcarrami_pika_ds`  
**描述**: 蛋白质知识和注释数据集

#### 模态
- **Protein** (蛋白质序列)

#### 任务类型
基于蛋白质的问答任务：

1. **Fields (字段注释)**
   - 催化活性、亚基组成、功能结构域、分类信息

2. **Summary (功能总结)**
   - 蛋白质功能的概括性描述

3. **QA (问答)**
   - 关于蛋白质各方面属性的问答对

#### 数据格式
- **pika_sequences.csv**: 蛋白质序列
- **pika_annotations.csv**: 注释信息（包含QA型注释）
- **pika_evogroup_split.csv**: 数据划分（train/test）

#### 特点
- 使用进化分组进行数据划分，避免数据泄露
- 每个UniProt ID可能有多个问答对
- 质量控制：过滤掉格式不规范的注释

#### 数据规模
- 约100K+蛋白质序列
- QA注释预计数十万条

---

### 4. ChatNT

**来源**: `instadeepai_chatnt_training_data`  
**论文**: [ChatNT - Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer)

#### 模态
- **DNA** (DNA序列)
- **RNA** (RNA序列)
- **Protein** (蛋白质序列)

#### 任务类型

##### DNA相关任务
1. **Promoter Detection**
   - promoter_all, promoter_tata, promoter_no_tata
   - 启动子识别和分类

2. **Enhancer Analysis**
   - NT_enhancers, NT_enhancers_types
   - deepstarr_developmental, deepstarr_housekeeping
   - 增强子识别和类型分类

3. **Splice Sites**
   - splice_sites_all, splice_sites_acceptors, splice_sites_donors
   - 剪接位点预测

4. **Chromatin Accessibility**
   - human_chromatin_accessibility_HepG2
   - 染色质可及性预测

5. **DNA Methylation**
   - human_dna_methylation_HUES64
   - DNA甲基化预测

6. **Histone Modifications**
   - H3K27ac, H3K4me1, H3K4me3, H3K4me2, H3K36me3, H3K79me3
   - H3K9ac, H3K14ac, H3, H4, H4ac
   - 组蛋白修饰预测

##### RNA相关任务
1. **RNA Degradation**
   - rna_degradation_human, rna_degradation_mouse
   - RNA降解预测

2. **lncRNA**
   - plant_lncrna_s_bicolor
   - 长非编码RNA识别

3. **Poly(A) Signal**
   - human_polya
   - poly(A)信号预测

4. **Plant RNA Tasks**
   - plant_pro_seq_m_esculenta
   - 植物RNA序列分析

##### Protein相关任务
1. **Protein Stability**
   - protein_stability
   - 蛋白质稳定性预测

2. **Protein Fluorescence**
   - protein_fluorescence
   - 蛋白质荧光强度预测

3. **Protein Meltome**
   - protein_meltome
   - 蛋白质热稳定性

##### Plant-specific Tasks
- plant_promoter_strength_leaf
- plant_promoter_strength_protoplast
- 植物启动子强度预测

#### 数据格式
```python
{
  "sequence": "ATCGATCG...",
  "exchanges": [
    {"role": "USER", "message": "预测该序列的..."},
    {"role": "ASSISTANT", "message": "预测结果是..."}
  ],
  "task_type": "promoter_detection",
  "task_modality": "dna"
}
```

#### 数据规模
- 训练集: train-00000 ~ train-00007 (8个分片)
- 测试集: test-00000
- 每个任务还有独立的目录存储
- 总计数十万到百万级别的样本

---

### 5. LLaMA-Gene

**来源**: `dnagpt_llama_gene_train_data`  
**论文**: [LLaMA-Gene](https://github.com/geneknowledge/LLaMA-Gene)

#### 模态
- **DNA** (DNA序列)
- **Protein** (蛋白质序列)

#### 任务类型

##### DNA任务
1. **Promoter Detection**
   - 启动子检测（二分类：promoter/non-promoter）
   - 使用标准的DNA序列输入

##### Protein任务
1. **Protein Function Prediction**
   - 蛋白质功能预测
   - 基于序列的功能注释

#### 数据格式
JSONL格式（每行一个JSON对象）：
```json
{
  "instruction": "Determine promoter detection of following dna sequence...",
  "input": "CGGGCCTGCCCCTCCGAG...",
  "output": "promoter"
}
```

#### 数据规模
- **DNA训练集**: sft_dna_train.json (~178K行)
- **DNA测试集**: sft_dna_eva.json (~19K行)
- **Protein训练集**: protein_sft_train.json (~62K行)
- **Protein测试集**: sft_protein_eva.json (~7K行)
- 总计约26万条样本

---

### 6. Biology-Instructions

**来源**: `gdrive_1OC3VpPKSQ0VHd9ZeZhnxI8EA2wTdrBg5`  
**描述**: 生物学指令数据集（包含多种交互任务）

#### 模态
- **RNA** (RNA序列)
- **Protein** (蛋白质序列)
- **Mixed** (混合模态)

#### 任务类型

1. **RNA-Protein Interaction**
   - RNA-蛋白质相互作用预测
   - 判断给定的RNA序列和蛋白质序列是否有相互作用
   - 包含正样本和负样本

2. **其他可能的任务**
   - 从文件名stage2_train.jsonl推测这是第二阶段训练数据
   - 可能包含更多生物学推理和问答任务

#### 数据格式
```json
{
  "input": "Is there any evidence of molecular interaction between <rna>AGCCAACG...</rna> and <protein>MSSNSASA...</protein>?",
  "output": "Computational predictions do not indicate...",
  "label": "negative",
  "task": "rna_protein_interaction"
}
```

#### 特点
- 使用特殊标记 `<rna>` 和 `<protein>` 来区分不同模态的序列
- 包含详细的任务标签和类别
- 问题形式多样化，增强模型的泛化能力

#### 数据规模
- stage2_train.jsonl: ~330万行
- 超大规模数据集

---

## 统计总结

### 按模态分类

| 模态 | 数据集 | 主要任务 |
|------|--------|----------|
| **Protein** | Mol-Instructions, UniProtQA, Pika-DS, ChatNT | 功能预测、问答、属性注释 |
| **DNA** | ChatNT, LLaMA-Gene | 启动子检测、增强子识别、表观遗传修饰 |
| **RNA** | ChatNT, Biology-Instructions | 降解预测、lncRNA识别、RNA-蛋白相互作用 |
| **Mixed** | Biology-Instructions | 跨模态相互作用预测 |

### 按任务类型分类

| 任务类型 | 涉及数据集 | 说明 |
|----------|------------|------|
| **问答 (QA)** | UniProtQA, Pika-DS, Mol-Instructions | 基于序列的知识问答 |
| **功能预测** | Mol-Instructions, LLaMA-Gene, ChatNT | 预测生物分子的功能和属性 |
| **序列分类** | LLaMA-Gene, ChatNT | 启动子检测、增强子识别等二分类/多分类 |
| **表观遗传学** | ChatNT | 染色质可及性、DNA甲基化、组蛋白修饰 |
| **相互作用预测** | Biology-Instructions | RNA-蛋白相互作用 |
| **稳定性预测** | ChatNT | 蛋白质稳定性、RNA降解 |

### 数据规模估计

| 数据集 | 预计规模 | 文件大小 |
|--------|----------|----------|
| Mol-Instructions | ~10K-50K | 中等 |
| UniProtQA | ~50K-100K | 中等 |
| Pika-DS | ~100K-500K | 大 |
| ChatNT | ~500K-1M | 很大 |
| LLaMA-Gene | ~260K | 大 |
| Biology-Instructions | ~3M | 超大 |
| **总计** | **~4-5M** | **数GB** |

---

## 使用建议

### 1. 任务选择
- **入门学习**: 从 LLaMA-Gene 开始（数据清晰、任务明确）
- **蛋白质研究**: 优先使用 UniProtQA 和 Pika-DS（高质量QA数据）
- **DNA/RNA研究**: 使用 ChatNT（任务全面、数据量大）
- **多模态学习**: 使用 Biology-Instructions（跨模态交互）

### 2. 数据集组合策略
- **蛋白质全面训练**: Mol-Instructions + UniProtQA + Pika-DS
- **DNA全面训练**: ChatNT (DNA部分) + LLaMA-Gene (DNA部分)
- **多任务学习**: 组合所有数据集

### 3. 数据质量控制
- 所有数据集都经过了验证和清洗
- 序列有效性检查（氨基酸/核苷酸字符验证）
- 格式统一化处理
- 详细的错误日志和统计报告

---

## 更新日志

- **2025-01-XX**: 初始版本，包含6个数据集的详细说明
- 数据处理脚本版本: v2.0

---

## 参考资料

1. [Mol-Instructions GitHub](https://github.com/zjunlp/Mol-Instructions)
2. [ChatNT/Nucleotide Transformer](https://github.com/instadeepai/nucleotide-transformer)
3. [LLaMA-Gene GitHub](https://github.com/geneknowledge/LLaMA-Gene)
4. [UniProt Database](https://www.uniprot.org/)
5. [Pika-DS Paper](相关论文链接)

