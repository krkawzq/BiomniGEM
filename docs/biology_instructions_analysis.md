# Biology-Instructions 数据集分析与清洗方案

## 一、数据集概览

**总数据量**: 3,330,232 条  
**数据来源**: Biology-Instructions  
**主要特点**: 包含DNA、RNA、蛋白质等多种生物序列的预测任务

## 二、任务类型分类

### 2.1 按模态分类

#### **RNA相关任务** (8类, ~2,109,473条, 63.4%)
1. **Isoform-Isoform** (1,575,557条, 47.31%)
   - 转录本异构体预测
   - 预测近端剪切位点的转录本比例
   
2. **Modification-Modification** (304,661条, 9.15%)
   - RNA修饰类型预测
   
3. **MeanRibosomeLoading** (76,319条, 2.29%)
   - 核糖体负载预测
   
4. **ProgrammableRNASwitches** (73,227条, 2.20%)
   - 可编程RNA开关的ON/OFF状态预测
   
5. **sirnaEfficiency** (53,592条, 1.61%)
   - siRNA敲低效率预测
   
6. **rna_protein_interaction** (14,994条, 0.45%)
   - RNA-蛋白质互作预测
   
7. **NoncodingRNAFamily** (5,670条, 0.17%)
   - 非编码RNA家族分类
   
8. **CRISPROnTarget** (1,453条, 0.04%)
   - CRISPR基因破坏成功率预测

#### **DNA相关任务** (34类, ~1,076,871条, 32.3%)

**A. 启动子相关** (6类, 189,424条)
- cpd-prom_core_all (47,356)
- cpd-prom_core_notata (42,452)
- cpd-prom_core_tata (4,904)
- pd-prom_300_all (47,356)
- pd-prom_300_notata (42,452)
- pd-prom_300_tata (4,904)

**B. 表观遗传修饰** (10类, 229,885条)
- emp-H3K4me3 (29,439)
- emp-H3K36me3 (27,904)
- emp-H4ac (27,275)
- emp-H3K14ac (26,438)
- emp-H3K4me1 (25,341)
- emp-H3K4me2 (24,545)
- emp-H3K79me3 (23,069)
- emp-H3K9ac (22,224)
- emp-H3 (11,971)
- emp-H4 (11,679)

**C. 转录因子结合位点** (10类, 208,362条)
- tf-m-1 (53,952)
- tf-h-0 (32,378)
- tf-h-1 (30,672)
- tf-h-3 (27,294)
- tf-h-2 (19,000)
- tf-h-4 (19,000)
- tf-m-4 (15,064)
- tf-m-0 (6,478)
- tf-m-2 (2,620)
- tf-m-3 (1,904)

**D. 增强子活性** (1类, 402,296条)
- enhancer_activity (402,296)

**E. 启动子-增强子互作** (6类, 14,288条)
- GM12878 (3,051)
- K562 (2,833)
- HeLa-S3 (2,513)
- HUVEC (2,164)
- NHEK (1,892)
- IMR90 (1,835)

#### **蛋白质相关任务** (6类, ~180,504条, 5.4%)
1. **Solubility** (62,478条, 1.88%)
   - 蛋白质溶解度预测
   
2. **Stability** (53,614条, 1.61%)
   - 蛋白质稳定性预测
   
3. **antibody_antigen** (22,359条, 0.67%)
   - 抗体-抗原互作预测
   
4. **Fluorescence** (21,446条, 0.64%)
   - 荧光蛋白亮度预测
   
5. **FunctionEC** (15,551条, 0.47%)
   - 酶功能EC编号预测
   
6. **Thermostability** (5,056条, 0.15%)
   - 热稳定性预测

## 三、数据清洗需求

### 3.1 移除instruction中的分类标记

**问题**: 约60%的数据在instruction中包含`[Classification: ...]`、`[Regression: ...]`等标记，不利于自动CoT标注。

**需要清除的标记模式**:
```
[Isoform prediction]                               : 300,013条
[Classification: ...]                              : 198,102条
[Regression: Enhancer activity prediction]         : 122,125条
[Classification: epigenetic marks prediction]      :  73,984条
[Classification: transcription factor binding...] :  62,871条
[RNA modification classification]                  :  58,038条
[Solubility Prediction Binary Classification]     :  18,648条
[Stability Prediction Regression]                  :  16,118条
... 等
```

**清洗策略**:
```python
import re

def clean_instruction(instruction):
    # 移除所有方括号标记
    instruction = re.sub(r'\[([^\]]+)\]\s*', '', instruction)
    # 清理多余空格
    instruction = re.sub(r'\s+', ' ', instruction).strip()
    return instruction
```

### 3.2 统一序列标签格式

**当前格式**: `<dna>`, `<rna>`, `<protein>`  
**需要确保**: 所有序列都有正确的开闭标签

### 3.3 任务类型重命名

将冗余的任务名称简化：
```
CRISPROnTarget-CRISPROnTarget  → crispr_on_target
Fluorescence-Fluorescence      → fluorescence
FunctionEC-FunctionEC          → function_ec
Isoform-Isoform               → isoform
... 等
```

## 四、任务合并方案

### 4.1 方案A: 细粒度分类 (推荐用于训练)

保持原有任务的细分，但进行重新组织：

```python
TASK_GROUPS = {
    'RNA': {
        'isoform': ['Isoform-Isoform'],
        'modification': ['Modification-Modification'],
        'ribosome_loading': ['MeanRibosomeLoading-MeanRibosomeLoading'],
        'rna_switches': ['ProgrammableRNASwitches-ProgrammableRNASwitches'],
        'sirna_efficiency': ['sirnaEfficiency-sirnaEfficiency'],
        'rna_protein_interaction': ['rna_protein_interaction'],
        'ncrna_family': ['NoncodingRNAFamily-NoncodingRNAFamily'],
        'crispr_efficiency': ['CRISPROnTarget-CRISPROnTarget'],
    },
    'DNA': {
        'promoter_core': ['cpd-prom_core_all', 'cpd-prom_core_notata', 'cpd-prom_core_tata'],
        'promoter_300bp': ['pd-prom_300_all', 'pd-prom_300_notata', 'pd-prom_300_tata'],
        'histone_H3K4': ['emp-H3K4me1', 'emp-H3K4me2', 'emp-H3K4me3'],
        'histone_H3_other': ['emp-H3', 'emp-H3K14ac', 'emp-H3K36me3', 'emp-H3K79me3', 'emp-H3K9ac'],
        'histone_H4': ['emp-H4', 'emp-H4ac'],
        'tf_human': ['tf-h-0', 'tf-h-1', 'tf-h-2', 'tf-h-3', 'tf-h-4'],
        'tf_mouse': ['tf-m-0', 'tf-m-1', 'tf-m-2', 'tf-m-3', 'tf-m-4'],
        'enhancer': ['enhancer_activity'],
        'prom_enh_interaction': ['promoter_enhancer_interaction-*'],
    },
    'Protein': {
        'solubility': ['Solubility-Solubility'],
        'stability': ['Stability-Stability'],
        'antibody_antigen': ['antibody_antigen'],
        'fluorescence': ['Fluorescence-Fluorescence'],
        'function_ec': ['FunctionEC-FunctionEC'],
        'thermostability': ['Thermostability-Thermostability'],
    }
}
```

### 4.2 方案B: 粗粒度分类 (推荐用于评估)

将相似任务完全合并：

```python
MERGED_TASKS = {
    # DNA任务合并
    'promoter_detection': [
        'cpd-prom_core_*', 'pd-prom_300_*'
    ],  # 189,424条
    
    'epigenetic_marks': [
        'emp-H3*', 'emp-H4*'
    ],  # 229,885条
    
    'tf_binding': [
        'tf-h-*', 'tf-m-*'
    ],  # 208,362条
    
    'enhancer_activity': [
        'enhancer_activity'
    ],  # 402,296条
    
    'regulatory_interaction': [
        'promoter_enhancer_interaction-*'
    ],  # 14,288条
    
    # RNA任务保持独立
    'rna_isoform': ['Isoform-Isoform'],  # 1,575,557条
    'rna_modification': ['Modification-Modification'],  # 304,661条
    'rna_translation': ['MeanRibosomeLoading-*', 'sirnaEfficiency-*'],  # 129,911条
    'rna_switches': ['ProgrammableRNASwitches-*'],  # 73,227条
    'rna_interaction': ['rna_protein_interaction'],  # 14,994条
    'rna_classification': ['NoncodingRNAFamily-*', 'CRISPROnTarget-*'],  # 7,123条
    
    # 蛋白质任务保持独立
    'protein_solubility': ['Solubility-*'],  # 62,478条
    'protein_stability': ['Stability-*', 'Thermostability-*'],  # 58,670条
    'protein_fluorescence': ['Fluorescence-*'],  # 21,446条
    'protein_function': ['FunctionEC-*'],  # 15,551条
    'protein_interaction': ['antibody_antigen'],  # 22,359条
}
```

## 五、Task提示词设计

为每个任务类别设计专门的提示词，替代`[Classification]`标记：

### 5.1 DNA任务提示词

#### Promoter Detection
```python
bio_promoter_abstract = "You are tasked with identifying whether a DNA sequence contains promoter elements that regulate gene transcription."

bio_promoter_description = [
    "You will be provided with a DNA sequence of varying length (70bp core promoter or 300bp region).",
    "Your task is to determine if this sequence functions as a promoter region.",
    "Promoter sequences typically contain regulatory elements such as TATA box, CAAT box, and Initiator elements.",
    "Consider: sequence motifs, GC content, position of regulatory elements, and presence of transcription factor binding sites.",
    "Some tasks distinguish between TATA-containing and TATA-less promoters.",
    "Your answer should be: 'Yes' (promoter) or 'No' (non-promoter), sometimes with additional classification of promoter type."
]
```

#### Epigenetic Marks
```python
bio_epigenetic_abstract = "You are tasked with predicting histone modifications and epigenetic marks in a DNA region."

bio_epigenetic_description = [
    "You will be provided with a DNA sequence from yeast or other organisms.",
    "Your task is to predict the presence of specific histone modifications (e.g., H3K4me3, H3K27ac, H4ac).",
    "Different histone modifications are associated with different chromatin states:",
    "  - H3K4me3: active promoter mark",
    "  - H3K36me3: gene body of actively transcribed genes",
    "  - H3K9ac/H3K14ac: transcriptional activation",
    "  - H4ac: open chromatin and active transcription",
    "Consider: genomic context, sequence composition, and co-occurrence of other epigenetic marks.",
    "Your answer should indicate presence or absence of the specific modification."
]
```

#### Transcription Factor Binding
```python
bio_tf_binding_abstract = "You are tasked with identifying transcription factor binding sites (TFBS) in DNA sequences."

bio_tf_binding_description = [
    "You will be provided with a 100bp DNA sequence from human or mouse genomes.",
    "Your task is to determine whether this region contains functional transcription factor binding sites.",
    "TFBS are short DNA motifs (typically 6-20bp) recognized by transcription factors.",
    "Consider: presence of known binding motifs, sequence conservation, DNA accessibility, and surrounding genomic context.",
    "Your answer should be: 'Yes' (contains TFBS) or 'No' (does not contain TFBS)."
]
```

#### Enhancer Activity
```python
bio_enhancer_abstract = "You are tasked with predicting the regulatory activity of DNA enhancer sequences."

bio_enhancer_description = [
    "You will be provided with a DNA sequence that may function as an enhancer.",
    "Your task is to predict the enhancer's transcriptional activity level.",
    "Enhancers can be classified as:",
    "  - Housekeeping (HK): constitutively active enhancers",
    "  - Developmental (Dev): tissue/stage-specific enhancers",
    "Consider: transcription factor binding site clusters, sequence conservation, and regulatory motif composition.",
    "Your answer should provide enrichment scores for HK and Dev enhancer activity."
]
```

#### Promoter-Enhancer Interaction
```python
bio_prom_enh_interaction_abstract = "You are tasked with predicting functional interactions between promoter and enhancer DNA sequences."

bio_prom_enh_interaction_description = [
    "You will be provided with two DNA sequences: one promoter region and one enhancer region.",
    "Your task is to determine whether these two regulatory elements interact functionally in specific cell types.",
    "Functional interactions are mediated by:",
    "  - Shared transcription factor binding profiles",
    "  - 3D chromatin looping bringing distant elements together",
    "  - Coordinated epigenetic signatures",
    "Consider: cell type-specific factors (GM12878, K562, HUVEC, HeLa-S3, IMR90, NHEK), sequence complementarity, and regulatory logic.",
    "Your answer should be: 'Yes' (functional interaction) or 'No' (no interaction)."
]
```

### 5.2 RNA任务提示词

#### RNA Isoform
```python
bio_isoform_abstract = "You are tasked with predicting alternative polyadenylation and transcript isoform usage."

bio_isoform_description = [
    "You will be provided with an RNA sequence containing polyadenylation signals.",
    "Your task is to predict the percentage of transcripts that use the proximal polyadenylation site.",
    "Alternative polyadenylation (APA) produces mRNA isoforms with different 3'UTR lengths.",
    "Consider: polyadenylation signals (AAUAAA and variants), upstream/downstream sequence elements, and regulatory protein binding sites.",
    "Your answer should be a percentage value (0-1) indicating the proximal isoform usage ratio."
]
```

#### RNA Modification
```python
bio_rna_modification_abstract = "You are tasked with identifying chemical modifications on RNA sequences."

bio_rna_modification_description = [
    "You will be provided with an RNA sequence.",
    "Your task is to predict the type of RNA modification present (e.g., m6A, Am, Pseudouridine, etc.).",
    "RNA modifications regulate RNA stability, translation, and localization.",
    "Consider: sequence context around modification sites, consensus motifs (e.g., DRACH for m6A), and secondary structure.",
    "Your answer should specify the type of modification detected."
]
```

#### Ribosome Loading & Translation
```python
bio_ribosome_loading_abstract = "You are tasked with predicting ribosome loading and translation efficiency from RNA sequences."

bio_ribosome_loading_description = [
    "You will be provided with an mRNA sequence (often 5'UTR or coding region).",
    "Your task is to predict the mean ribosome loading (ribosome density) on this transcript.",
    "Ribosome loading reflects translation efficiency and is influenced by:",
    "  - 5'UTR structure and length",
    "  - Kozak sequence context",
    "  - uORFs (upstream open reading frames)",
    "  - Codon usage and tRNA availability",
    "Your answer should be a numerical value representing ribosome loading intensity."
]
```

#### RNA Switches
```python
bio_rna_switches_abstract = "You are tasked with predicting the ON/OFF behavior of programmable RNA switches."

bio_rna_switches_description = [
    "You will be provided with an RNA sequence containing a riboswitch or RNA switch element.",
    "Your task is to predict:",
    "  - ON state expression level",
    "  - OFF state expression level",
    "  - ON/OFF ratio (dynamic range)",
    "RNA switches undergo conformational changes in response to ligands or signals.",
    "Consider: RNA secondary structure, aptamer domains, expression platforms, and switching mechanism.",
    "Your answer should provide three values: ON level, OFF level, and ON/OFF ratio."
]
```

#### siRNA Efficiency
```python
bio_sirna_efficiency_abstract = "You are tasked with predicting the knockdown efficiency of siRNA molecules."

bio_sirna_efficiency_description = [
    "You will be provided with an siRNA sequence and its target mRNA region.",
    "Your task is to predict the percentage of mRNA remaining after siRNA treatment.",
    "siRNA efficiency depends on:",
    "  - Sequence composition and GC content",
    "  - Thermodynamic properties of siRNA duplex",
    "  - Target site accessibility",
    "  - Off-target effects",
    "Your answer should be a percentage (0-100) indicating remaining mRNA level after knockdown."
]
```

#### RNA-Protein Interaction
```python
bio_rna_protein_interaction_abstract = "You are tasked with predicting interactions between RNA sequences and proteins."

bio_rna_protein_interaction_description = [
    "You will be provided with an RNA sequence and a protein sequence.",
    "Your task is to determine whether they form a biologically meaningful interaction.",
    "RNA-protein interactions are mediated by:",
    "  - RNA-binding domains (RRM, KH, dsRBD, etc.) in proteins",
    "  - RNA secondary structure elements (stem-loops, bulges)",
    "  - Sequence-specific recognition motifs",
    "  - Electrostatic complementarity",
    "Consider: presence of RBP domains, RNA structural features, and known binding patterns.",
    "Your answer should be: 'Positive' (interaction) or 'Negative' (no interaction)."
]
```

#### Non-coding RNA Family
```python
bio_ncrna_family_abstract = "You are tasked with classifying non-coding RNA sequences into functional families."

bio_ncrna_family_description = [
    "You will be provided with a non-coding RNA sequence.",
    "Your task is to identify which ncRNA family it belongs to (e.g., HACA-box, snoRNA, miRNA, lncRNA, etc.).",
    "Different ncRNA families have distinct:",
    "  - Sequence motifs and consensus sequences",
    "  - Secondary structure patterns",
    "  - Length distributions",
    "  - Functional roles",
    "Consider: conserved sequence elements, structural features, and length.",
    "Your answer should specify the ncRNA family classification."
]
```

#### CRISPR Efficiency
```python
bio_crispr_efficiency_abstract = "You are tasked with predicting the on-target efficiency of CRISPR guide RNAs."

bio_crispr_efficiency_description = [
    "You will be provided with a guide RNA sequence (typically ~20-23nt).",
    "Your task is to predict the gene disruption success rate (on-target activity).",
    "CRISPR efficiency is influenced by:",
    "  - GC content and distribution",
    "  - Presence of specific nucleotides at key positions",
    "  - Secondary structure of the guide RNA",
    "  - Target site chromatin accessibility",
    "Your answer should be a numerical score (0-1) representing gene disruption efficiency."
]
```

### 5.3 蛋白质任务提示词

#### Protein Solubility
```python
bio_protein_solubility_abstract = "You are tasked with predicting the solubility of proteins in aqueous solution."

bio_protein_solubility_description = [
    "You will be provided with a protein amino acid sequence.",
    "Your task is to predict whether the protein will be soluble when expressed.",
    "Protein solubility depends on:",
    "  - Hydrophobicity distribution",
    "  - Charged residue content and distribution",
    "  - Aggregation-prone regions",
    "  - Structural disorder propensity",
    "Consider: amino acid composition, sequence-based aggregation predictors, and known solubility determinants.",
    "Your answer should be: 'Soluble' or 'Insoluble'."
]
```

#### Protein Stability
```python
bio_protein_stability_abstract = "You are tasked with predicting protein stability and the effects of mutations."

bio_protein_stability_description = [
    "You will be provided with a protein sequence or mutation information.",
    "Your task is to predict protein stability or mutation-induced stability changes.",
    "Protein stability is determined by:",
    "  - Hydrophobic core packing",
    "  - Hydrogen bond networks",
    "  - Disulfide bonds",
    "  - Electrostatic interactions",
    "  - Conformational entropy",
    "Consider: amino acid composition, predicted secondary structure, and physicochemical properties.",
    "Your answer should be a stability score or classification (stable/unstable)."
]
```

#### Protein Thermostability
```python
bio_protein_thermostability_abstract = "You are tasked with predicting the melting temperature (Tm) of proteins."

bio_protein_thermostability_description = [
    "You will be provided with a protein sequence.",
    "Your task is to predict the temperature at which 50% of the protein is denatured (Tm).",
    "Thermostability is influenced by:",
    "  - Number and position of disulfide bonds",
    "  - Salt bridges and ion pairs",
    "  - Hydrophobic core size",
    "  - Protein compactness",
    "Consider: amino acid composition, predicted structural features, and sequence length.",
    "Your answer should be a temperature value in degrees Celsius."
]
```

#### Protein Fluorescence
```python
bio_protein_fluorescence_abstract = "You are tasked with predicting the fluorescence intensity of fluorescent protein variants."

bio_protein_fluorescence_description = [
    "You will be provided with a fluorescent protein sequence (typically GFP variants).",
    "Your task is to predict the relative fluorescence intensity or brightness.",
    "Fluorescence depends on:",
    "  - Chromophore chemical environment",
    "  - Protein folding quality",
    "  - Quantum yield",
    "  - Maturation efficiency",
    "Consider: mutations affecting the chromophore, structural stability, and solvent accessibility.",
    "Your answer should be a numerical fluorescence intensity value."
]
```

#### Protein Function (EC Number)
```python
bio_protein_function_ec_abstract = "You are tasked with predicting the enzymatic function and EC number of proteins."

bio_protein_function_ec_description = [
    "You will be provided with a protein amino acid sequence.",
    "Your task is to predict the enzyme's EC number (Enzyme Commission classification).",
    "EC numbers are hierarchical: EC X.X.X.X representing:",
    "  - First digit: enzyme class (oxidoreductase, transferase, etc.)",
    "  - Second digit: subclass",
    "  - Third digit: sub-subclass",
    "  - Fourth digit: specific enzyme",
    "Consider: presence of catalytic domains, active site motifs, and sequence similarity to known enzymes.",
    "Your answer should be one or more EC numbers (e.g., EC 3.2.2.9)."
]
```

#### Antibody-Antigen Interaction
```python
bio_antibody_antigen_abstract = "You are tasked with predicting binding affinity between antibody and antigen sequences."

bio_antibody_antigen_description = [
    "You will be provided with an antibody sequence and an antigen sequence.",
    "Your task is to determine whether the antibody binds to the antigen.",
    "Antibody-antigen binding is mediated by:",
    "  - CDR (Complementarity-Determining Region) loops in antibodies",
    "  - Epitope accessibility on antigens",
    "  - Shape and chemical complementarity",
    "  - Electrostatic and hydrophobic interactions",
    "Consider: antibody CDR sequences, antigen surface properties, and known binding patterns.",
    "Your answer should be: 'Positive' (binding) or 'Negative' (no binding)."
]
```

## 六、推荐清洗和处理流程

### 6.1 数据清洗流程

```python
import re
import json
from typing import Dict, List

def clean_biology_instructions_data(data: List[Dict]) -> List[Dict]:
    """
    清洗Biology-Instructions数据集
    """
    cleaned_data = []
    
    for item in data:
        # 1. 移除instruction中的分类标记
        instruction = item.get('instruction', '')
        instruction = re.sub(r'\[([^\]]+)\]\s*', '', instruction)
        instruction = re.sub(r'\s+', ' ', instruction).strip()
        
        # 2. 验证序列标签完整性
        # 确保<dna>, <rna>, <protein>标签成对出现
        for tag in ['dna', 'rna', 'protein']:
            open_count = instruction.count(f'<{tag}>')
            close_count = instruction.count(f'<{tag}>')
            if open_count != close_count:
                print(f"Warning: Mismatched {tag} tags in item")
                continue
        
        # 3. 简化task_type命名
        original_task = item['metadata']['task_type']
        simplified_task = simplify_task_name(original_task)
        
        # 4. 添加task_group和modality信息
        task_group, modality = get_task_group_and_modality(simplified_task)
        
        cleaned_item = {
            'instruction': instruction,
            'response': item.get('response', ''),
            'sequence': item.get('sequence', ''),
            'metadata': {
                'source': 'Biology-Instructions',
                'split': item['metadata'].get('split', 'train'),
                'task_type': simplified_task,
                'task_type_original': original_task,
                'task_group': task_group,
                'modality': modality
            }
        }
        
        cleaned_data.append(cleaned_item)
    
    return cleaned_data


def simplify_task_name(task_type: str) -> str:
    """简化任务名称"""
    mapping = {
        'CRISPROnTarget-CRISPROnTarget': 'crispr_efficiency',
        'Fluorescence-Fluorescence': 'fluorescence',
        'FunctionEC-FunctionEC': 'function_ec',
        'Isoform-Isoform': 'isoform',
        'MeanRibosomeLoading-MeanRibosomeLoading': 'ribosome_loading',
        'Modification-Modification': 'rna_modification',
        'NoncodingRNAFamily-NoncodingRNAFamily': 'ncrna_family',
        'ProgrammableRNASwitches-ProgrammableRNASwitches': 'rna_switches',
        'Solubility-Solubility': 'solubility',
        'Stability-Stability': 'stability',
        'Thermostability-Thermostability': 'thermostability',
        'sirnaEfficiency-sirnaEfficiency': 'sirna_efficiency',
    }
    
    # 对于已在mapping中的，直接返回
    if task_type in mapping:
        return mapping[task_type]
    
    # 对于emp-, cpd-, pd-, tf-等，保持原样
    return task_type


def get_task_group_and_modality(task_type: str) -> tuple:
    """获取任务组和模态信息"""
    
    dna_tasks = [
        'cpd-prom', 'pd-prom', 'emp-', 'tf-', 'enhancer', 
        'promoter_enhancer_interaction'
    ]
    
    rna_tasks = [
        'isoform', 'rna_modification', 'ribosome_loading', 
        'rna_switches', 'sirna_efficiency', 'rna_protein_interaction',
        'ncrna_family', 'crispr_efficiency'
    ]
    
    protein_tasks = [
        'solubility', 'stability', 'thermostability', 
        'fluorescence', 'function_ec', 'antibody_antigen'
    ]
    
    # 判断modality
    if any(task in task_type for task in dna_tasks):
        modality = 'dna'
    elif any(task in task_type for task in rna_tasks):
        modality = 'rna'
    elif any(task in task_type for task in protein_tasks):
        modality = 'protein'
    else:
        modality = 'mixed'
    
    # 判断task_group
    if 'prom' in task_type:
        task_group = 'promoter'
    elif 'emp-' in task_type:
        task_group = 'epigenetic'
    elif 'tf-' in task_type:
        task_group = 'tf_binding'
    elif 'enhancer' in task_type:
        task_group = 'enhancer'
    elif 'interaction' in task_type:
        task_group = 'interaction'
    else:
        task_group = task_type
    
    return task_group, modality
```

### 6.2 建议的数据拆分策略

考虑到数据量极度不均衡（Isoform占47%），建议：

1. **按任务类型分层采样**
   ```python
   # 对于数据量>100k的任务，采样到100k
   # 对于数据量<1k的任务，使用全部数据
   # 对于其他任务，保持原样
   ```

2. **为CoT标注创建优先级队列**
   ```python
   # 高优先级: 数据量适中(1k-50k)、任务明确
   # 中优先级: 数据量较大(50k-100k)但重要的任务
   # 低优先级: 数据量极大(>100k)的任务，采样后标注
   ```

3. **测试集构建**
   ```python
   # 每个任务类型至少保留500条作为测试集
   # 保持原始split信息，优先使用原有的test split
   ```

## 七、集成到现有tasks.py的方案

在`SynBioCoT/prompt/tasks.py`中添加Biology-Instructions任务定义：

```python
# 在文件末尾添加Biology-Instructions部分
tasks['Biology-Instructions'] = {
    'promoter_detection': {
        'abstract': bio_promoter_abstract,
        'description': bio_promoter_description,
        'task_type': 'dna',
        'file': 'vc/Biology_Instructions.csv',
        'filter': "task_group == 'promoter'"
    },
    'epigenetic_marks': {
        'abstract': bio_epigenetic_abstract,
        'description': bio_epigenetic_description,
        'task_type': 'dna',
        'file': 'vc/Biology_Instructions.csv',
        'filter': "task_group == 'epigenetic'"
    },
    # ... 其他任务
}
```

## 八、总结与建议

### 8.1 数据集特点
- ✅ 任务多样性好: 涵盖DNA/RNA/蛋白质多个领域
- ✅ 数据量大: 330万条，足够训练
- ⚠️ 数据不均衡: Isoform任务占47%，需要平衡采样
- ⚠️ 格式需清洗: 60%数据包含分类标记

### 8.2 清洗优先级
1. **P0 (必须)**: 移除所有`[Classification]`等标记
2. **P1 (重要)**: 验证序列标签完整性
3. **P2 (建议)**: 统一task_type命名，添加task_group

### 8.3 CoT标注建议
1. 按task_group分组标注，每组使用对应的reasoning guidance
2. 对于数据量>50k的任务，采样10%进行CoT标注
3. 优先标注中等规模任务(1k-50k)，ROI最高

### 8.4 评估策略
- 细粒度评估: 每个原始task_type单独评估
- 粗粒度评估: 按task_group合并评估
- 跨模态评估: 测试DNA/RNA/Protein任务的泛化能力

