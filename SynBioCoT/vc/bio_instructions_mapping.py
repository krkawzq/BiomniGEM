"""
Biology-Instructions 数据集映射配置

包含：
1. 任务归类映射表（粗粒度）- 将47个原始任务归并为15个大类
2. []标记映射表 - 将标记映射到额外的描述信息
"""

# ============================================================================
# 1. 任务归类映射表（粗粒度）
# ============================================================================

TASK_MERGE_MAPPING = {
    # DNA任务组（8个大类）
    # 1. 启动子检测 - 合并所有启动子相关任务
    'promoter_detection': [
        'cpd-prom_core_all',
        'cpd-prom_core_notata', 
        'cpd-prom_core_tata',
        'pd-prom_300_all',
        'pd-prom_300_notata',
        'pd-prom_300_tata',
    ],
    
    # 2. 表观遗传修饰 - 合并所有组蛋白修饰任务
    'epigenetic_marks': [
        'emp-H3',
        'emp-H3K14ac',
        'emp-H3K36me3',
        'emp-H3K4me1',
        'emp-H3K4me2',
        'emp-H3K4me3',
        'emp-H3K79me3',
        'emp-H3K9ac',
        'emp-H4',
        'emp-H4ac',
    ],
    
    # 3. 转录因子结合位点 - 合并人类和小鼠的TF结合任务
    'tf_binding': [
        'tf-h-0', 'tf-h-1', 'tf-h-2', 'tf-h-3', 'tf-h-4',  # 人类
        'tf-m-0', 'tf-m-1', 'tf-m-2', 'tf-m-3', 'tf-m-4',  # 小鼠
    ],
    
    # 4. 增强子活性
    'enhancer_activity': [
        'enhancer_activity',
    ],
    
    # 5. 启动子-增强子互作 - 合并所有细胞系
    'regulatory_interaction': [
        'promoter_enhancer_interaction-GM12878',
        'promoter_enhancer_interaction-HUVEC',
        'promoter_enhancer_interaction-HeLa-S3',
        'promoter_enhancer_interaction-IMR90',
        'promoter_enhancer_interaction-K562',
        'promoter_enhancer_interaction-NHEK',
    ],
    
    # RNA任务组（5个大类）
    # 6. RNA异构体和剪切
    'rna_isoform': [
        'Isoform-Isoform',
    ],
    
    # 7. RNA修饰和调控
    'rna_modification': [
        'Modification-Modification',
    ],
    
    # 8. RNA翻译调控 - 合并核糖体负载、siRNA效率、RNA开关
    'rna_translation_regulation': [
        'MeanRibosomeLoading-MeanRibosomeLoading',
        'sirnaEfficiency-sirnaEfficiency',
        'ProgrammableRNASwitches-ProgrammableRNASwitches',
    ],
    
    # 9. RNA分类和功能 - 合并非编码RNA分类、CRISPR效率
    'rna_function': [
        'NoncodingRNAFamily-NoncodingRNAFamily',
        'CRISPROnTarget-CRISPROnTarget',
    ],
    
    # 蛋白质任务组（3个大类）
    # 10. 蛋白质稳定性 - 合并溶解度、稳定性、热稳定性
    'protein_stability': [
        'Solubility-Solubility',
        'Stability-Stability',
        'Thermostability-Thermostability',
    ],
    
    # 11. 蛋白质功能和性质
    'protein_function': [
        'FunctionEC-FunctionEC',
        'Fluorescence-Fluorescence',
    ],
    
    # 交互任务组（2个大类）
    # 12. 分子互作 - 合并RNA-蛋白、抗体-抗原互作
    'molecular_interaction': [
        'rna_protein_interaction',
        'antibody_antigen',
    ],
}

# 反向映射：原始任务 -> 合并后任务
ORIGINAL_TO_MERGED = {}
for merged_task, original_tasks in TASK_MERGE_MAPPING.items():
    for original_task in original_tasks:
        ORIGINAL_TO_MERGED[original_task] = merged_task


# ============================================================================
# 2. []标记映射表 - 标记到额外描述的映射
# ============================================================================

BRACKET_TAG_DESCRIPTIONS = {
    # DNA分类任务标记
    '[Classification: core promoter dection]': [
        "This is a binary classification task.",
        "Focus on identifying core promoter elements (TATA box, Initiator, etc.) within the 70bp sequence.",
        "Some sequences may be distinguished by the presence or absence of TATA box.",
    ],
    
    '[Classification: promoter detection]': [
        "This is a binary classification task.",
        "Analyze the 300bp sequence for promoter characteristics.",
        "Consider both proximal promoter elements and upstream regulatory regions.",
    ],
    
    '[Classification: epigenetic marks prediction]': [
        "This is a binary classification task to detect histone modifications.",
        "Predict whether the DNA region carries specific epigenetic marks based on sequence features.",
        "Different marks indicate different chromatin states (active/repressive).",
    ],
    
    '[Classification: transcription factor binding sites detection]': [
        "This is a binary classification task.",
        "Identify whether the 100bp sequence contains transcription factor binding motifs.",
        "Consider both the presence of consensus motifs and sequence context.",
    ],
    
    '[Classification: promoter enhancer interaction detection]': [
        "This is a binary classification task.",
        "Predict functional interaction between two regulatory DNA sequences.",
        "Consider cell type-specific regulatory contexts.",
    ],
    
    # DNA回归任务标记
    '[Regression: Enhancer activity prediction]': [
        "This is a regression task.",
        "Predict two enrichment scores: HK (housekeeping) and Dev (developmental).",
        "Higher scores indicate stronger enhancer activity in the respective context.",
    ],
    
    # RNA分类任务标记
    '[Classification: RNA-protein interaction detection]': [
        "This is a binary classification task.",
        "Predict binding between RNA and protein sequences.",
        "Consider RNA-binding domains in proteins and RNA structural elements.",
    ],
    
    '[RNA modification classification]': [
        "This is a multi-class classification task.",
        "Identify the type of RNA modification present (e.g., m6A, Am, Pseudouridine).",
        "Consider sequence context and modification consensus motifs.",
    ],
    
    '[Noncoding RNA family classification]': [
        "This is a multi-class classification task.",
        "Classify the non-coding RNA into its functional family.",
        "Consider sequence motifs, structure patterns, and length characteristics.",
    ],
    
    # RNA回归任务标记
    '[Isoform prediction]': [
        "This is a regression task.",
        "Predict the percentage (0-1) of transcripts using the proximal polyadenylation site.",
        "Higher values indicate preference for proximal site usage.",
    ],
    
    '[Ribosome loading prediction]': [
        "This is a regression task.",
        "Predict the mean ribosome loading (ribosome density) on the transcript.",
        "The value reflects translation efficiency.",
    ],
    
    '[Programmable RNA switches prediction]': [
        "This is a regression task with three output values.",
        "Predict: ON state level, OFF state level, and ON/OFF ratio.",
        "The ON/OFF ratio indicates the dynamic range of the switch.",
    ],
    
    '[SiRNA Efficiency prediction]': [
        "This is a regression task.",
        "Predict the percentage of mRNA remaining after siRNA knockdown.",
        "Lower values indicate more efficient knockdown.",
    ],
    
    '[CRISPR On target prediction]': [
        "This is a regression task.",
        "Predict the gene disruption success rate (0-1) for the guide RNA.",
        "Higher values indicate better on-target efficiency.",
    ],
    
    # 蛋白质分类任务标记
    '[Classification: antibody-antigen interaction detection]': [
        "This is a binary classification task.",
        "Predict whether the antibody binds to the antigen.",
        "Consider CDR regions in antibodies and epitope characteristics.",
    ],
    
    '[Solubility Prediction Binary Classification]': [
        "This is a binary classification task.",
        "Predict whether the protein will be soluble when expressed.",
        "Consider hydrophobicity, aggregation propensity, and disorder.",
    ],
    
    # 蛋白质回归任务标记
    '[EC Number Prediction Multilabel Classification]': [
        "This is a multi-label classification task.",
        "Predict one or more EC numbers for the enzyme.",
        "EC numbers follow the format: EC X.X.X.X (class.subclass.sub-subclass.enzyme).",
    ],
    
    '[Fluorescence Prediction Regression]': [
        "This is a regression task.",
        "Predict the relative fluorescence intensity of the protein variant.",
        "Higher values indicate brighter fluorescence.",
    ],
    
    '[Stability Prediction Regression]': [
        "This is a regression task.",
        "Predict the protein stability score.",
        "Higher values typically indicate more stable proteins.",
    ],
    
    '[Thermostability Prediction Regression]': [
        "This is a regression task.",
        "Predict the melting temperature (Tm) in degrees Celsius.",
        "Tm is the temperature at which 50% of the protein is denatured.",
    ],
}


# ============================================================================
# 3. 辅助函数
# ============================================================================

def get_merged_task_type(original_task_type: str) -> str:
    """
    获取合并后的任务类型
    
    Args:
        original_task_type: 原始任务类型名称
        
    Returns:
        合并后的任务类型名称，如果未找到则返回原始名称
    """
    return ORIGINAL_TO_MERGED.get(original_task_type, original_task_type)


def extract_and_map_bracket_tags(instruction: str) -> tuple:
    """
    从instruction中提取[]标记，返回清理后的instruction和额外的描述列表
    
    Args:
        instruction: 原始instruction字符串
        
    Returns:
        (cleaned_instruction, extra_descriptions)
        - cleaned_instruction: 移除[]标记后的instruction
        - extra_descriptions: 根据标记映射得到的额外描述列表
    """
    import re
    
    # 提取所有[]标记
    matches = re.findall(r'\[([^\]]+)\]', instruction)
    
    # 收集额外的描述
    extra_descriptions = []
    for match in matches:
        tag = f"[{match}]"
        if tag in BRACKET_TAG_DESCRIPTIONS:
            extra_descriptions.extend(BRACKET_TAG_DESCRIPTIONS[tag])
    
    # 移除所有[]标记
    cleaned_instruction = re.sub(r'\[([^\]]+)\]\s*', '', instruction)
    cleaned_instruction = re.sub(r'\s+', ' ', cleaned_instruction).strip()
    
    return cleaned_instruction, extra_descriptions


def get_task_statistics():
    """
    获取任务合并统计信息
    
    Returns:
        字典，包含每个合并任务的原始任务数量
    """
    stats = {}
    for merged_task, original_tasks in TASK_MERGE_MAPPING.items():
        stats[merged_task] = {
            'count': len(original_tasks),
            'tasks': original_tasks
        }
    return stats


# ============================================================================
# 4. 模态分类
# ============================================================================

TASK_MODALITY = {
    # DNA任务
    'promoter_detection': 'dna',
    'epigenetic_marks': 'dna',
    'tf_binding': 'dna',
    'enhancer_activity': 'dna',
    'regulatory_interaction': 'dna',
    
    # RNA任务
    'rna_isoform': 'rna',
    'rna_modification': 'rna',
    'rna_translation_regulation': 'rna',
    'rna_function': 'rna',
    
    # 蛋白质任务
    'protein_stability': 'protein',
    'protein_function': 'protein',
    
    # 交互任务
    'molecular_interaction': 'multi',
}


if __name__ == '__main__':
    # 测试代码
    print("任务合并统计：")
    print("=" * 80)
    stats = get_task_statistics()
    for merged_task, info in stats.items():
        print(f"\n{merged_task} ({TASK_MODALITY[merged_task]}):")
        print(f"  包含 {info['count']} 个原始任务")
        print(f"  任务列表: {', '.join(info['tasks'][:3])}" + 
              (f" ... (共{info['count']}个)" if info['count'] > 3 else ""))
    
    print("\n" + "=" * 80)
    print(f"总计: {len(TASK_MERGE_MAPPING)} 个合并后的任务类型")
    print(f"覆盖: {len(ORIGINAL_TO_MERGED)} 个原始任务类型")
    
    print("\n" + "=" * 80)
    print("[]标记映射统计：")
    print(f"支持的标记数量: {len(BRACKET_TAG_DESCRIPTIONS)}")
    
    # 测试标记提取
    print("\n" + "=" * 80)
    print("标记提取测试：")
    test_instruction = "[Classification: promoter detection] Analyze this DNA sequence for promoter elements."
    cleaned, extras = extract_and_map_bracket_tags(test_instruction)
    print(f"原始: {test_instruction}")
    print(f"清理后: {cleaned}")
    print(f"额外描述: {extras}")

