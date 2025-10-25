#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
数据组织脚本 v2.0
对下载的数据集进行解压、验证和统一格式化

统一数据格式:
{
    "instruction": str,      # 问题或指令
    "sequence": str,         # 生物序列 (DNA/RNA/protein)
    "response": str,         # 答案或输出
    "metadata": {
        "source": str,       # 数据来源
        "split": str,        # train/test/validation
        "modality": str,     # protein/dna/rna
        "task_type": str,    # 可选: 任务类型
        **kwargs             # 其他元数据
    }
}
"""

import zipfile
import json
import pandas as pd
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter, defaultdict
import traceback

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('data_process.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


class DataValidator:
    """数据验证器"""
    
    @staticmethod
    def validate_item(item: Dict[str, Any], source: str) -> Tuple[bool, Optional[str]]:
        """
        验证单条数据的有效性
        
        Returns:
            (is_valid, error_message)
        """
        # 检查必需字段
        required_fields = ['instruction', 'sequence', 'response', 'metadata']
        for field in required_fields:
            if field not in item:
                return False, f"缺少必需字段: {field}"
            if not item[field]:
                return False, f"字段 {field} 为空"
        
        # 检查类型
        if not isinstance(item['instruction'], str):
            return False, "instruction 必须是字符串"
        if not isinstance(item['sequence'], str):
            return False, "sequence 必须是字符串"
        if not isinstance(item['response'], str):
            return False, "response 必须是字符串"
        if not isinstance(item['metadata'], dict):
            return False, "metadata 必须是字典"
        
        # 检查metadata必需字段
        metadata_required = ['source', 'split', 'modality']
        for field in metadata_required:
            if field not in item['metadata']:
                return False, f"metadata 缺少必需字段: {field}"
        
        # 检查序列长度
        if len(item['sequence']) == 0:
            return False, "序列长度为0"
        if len(item['sequence']) > 100000:  # 设置合理的上限
            return False, f"序列过长: {len(item['sequence'])}"
        
        # 检查modality有效性
        valid_modalities = ['protein', 'dna', 'rna']
        if item['metadata']['modality'] not in valid_modalities:
            return False, f"无效的modality: {item['metadata']['modality']}"
        
        return True, None
    
    @staticmethod
    def validate_sequence_content(sequence: str, modality: str) -> bool:
        """验证序列内容是否符合模态"""
        sequence = sequence.upper()
        
        if modality == 'protein':
            # 标准氨基酸字母
            valid_chars = set('ACDEFGHIKLMNPQRSTVWY')
            # 允许一些常见的特殊字符
            valid_chars.update('XUBZO*-')
        elif modality == 'dna':
            valid_chars = set('ATCGN-')
        elif modality == 'rna':
            valid_chars = set('AUCGN-')
        else:
            return True
        
        # 检查序列中是否有非法字符
        sequence_chars = set(sequence)
        invalid_chars = sequence_chars - valid_chars
        
        if invalid_chars:
            # 如果非法字符比例小于5%,认为可以接受(可能是特殊标记)
            invalid_ratio = len([c for c in sequence if c in invalid_chars]) / len(sequence)
            return invalid_ratio < 0.05
        
        return True


class DataStatistics:
    """数据统计器"""
    
    def __init__(self):
        self.stats = defaultdict(lambda: {
            'total': 0,
            'valid': 0,
            'invalid': 0,
            'splits': Counter(),
            'modalities': Counter(),
            'seq_lengths': [],
            'error_types': Counter()
        })
    
    def add_item(self, source: str, item: Dict[str, Any], is_valid: bool, error_msg: Optional[str] = None):
        """添加一条数据的统计信息"""
        self.stats[source]['total'] += 1
        
        if is_valid:
            self.stats[source]['valid'] += 1
            self.stats[source]['splits'][item['metadata']['split']] += 1
            self.stats[source]['modalities'][item['metadata']['modality']] += 1
            self.stats[source]['seq_lengths'].append(len(item['sequence']))
        else:
            self.stats[source]['invalid'] += 1
            if error_msg:
                self.stats[source]['error_types'][error_msg] += 1
    
    def generate_report(self) -> str:
        """生成统计报告"""
        report = ["\n" + "="*80]
        report.append("数据处理统计报告")
        report.append("="*80)
        
        total_valid = 0
        total_invalid = 0
        
        for source, stat in sorted(self.stats.items()):
            report.append(f"\n【{source}】")
            report.append(f"  总计: {stat['total']:,} 条")
            report.append(f"  有效: {stat['valid']:,} 条 ({stat['valid']/stat['total']*100:.2f}%)")
            report.append(f"  无效: {stat['invalid']:,} 条 ({stat['invalid']/stat['total']*100:.2f}%)")
            
            if stat['splits']:
                report.append(f"  数据集划分:")
                for split, count in stat['splits'].most_common():
                    report.append(f"    - {split}: {count:,}")
            
            if stat['modalities']:
                report.append(f"  模态分布:")
                for modality, count in stat['modalities'].most_common():
                    report.append(f"    - {modality}: {count:,}")
            
            if stat['seq_lengths']:
                seq_lens = stat['seq_lengths']
                report.append(f"  序列长度统计:")
                report.append(f"    - 平均: {sum(seq_lens)/len(seq_lens):.1f}")
                report.append(f"    - 最小: {min(seq_lens)}")
                report.append(f"    - 最大: {max(seq_lens)}")
                report.append(f"    - 中位数: {sorted(seq_lens)[len(seq_lens)//2]}")
            
            if stat['error_types']:
                report.append(f"  错误类型 (前5种):")
                for error, count in stat['error_types'].most_common(5):
                    report.append(f"    - {error}: {count}")
            
            total_valid += stat['valid']
            total_invalid += stat['invalid']
        
        report.append(f"\n{'='*80}")
        report.append(f"总计: {total_valid + total_invalid:,} 条")
        report.append(f"有效: {total_valid:,} 条 ({total_valid/(total_valid+total_invalid)*100:.2f}%)")
        report.append(f"无效: {total_invalid:,} 条 ({total_invalid/(total_valid+total_invalid)*100:.2f}%)")
        report.append("="*80 + "\n")
        
        return "\n".join(report)


def process_Mol_Instructions() -> List[Dict[str, Any]]:
    """
    处理 Mol-Instructions 数据集
    
    数据来源: zjunlp_mol_instructions/data/Protein-oriented_Instructions.zip
    格式: ZIP文件包含多个JSON文件,每个JSON是列表格式
    
    返回: 统一格式的数据列表
    """
    logger.info("="*80)
    logger.info("开始处理 Mol-Instructions 数据集")
    logger.info("="*80)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "datasets" / "zjunlp_mol_instructions" / "data"
    unzip_dir = base_dir / "datasets" / "zjunlp_mol_instructions" / "unzip_data"
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []
    
    zip_path = data_dir / "Protein-oriented_Instructions.zip"
    if not zip_path.exists():
        logger.error(f"ZIP文件不存在: {zip_path}")
        return []
    
    # 创建解压目录
    unzip_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 解压文件
        logger.info(f"解压文件: {zip_path.name}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(unzip_dir)
        
        protein_dir = unzip_dir / "Protein-oriented_Instructions"
        if not protein_dir.exists():
            logger.error(f"解压后目录不存在: {protein_dir}")
            return []
        
        # 定义要处理的文件
        target_files = [
            "catalytic_activity.json",
            "domain_motif.json",
            "general_function.json",
            "protein_function.json"
        ]
        
        all_data = []
        validator = DataValidator()
        statistics = DataStatistics()
        
        for filename in target_files:
            json_file = protein_dir / filename
            if not json_file.exists():
                logger.warning(f"文件不存在: {filename}")
                continue
            
            logger.info(f"处理文件: {filename}")
            
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data_list = json.load(f)
                
                logger.info(f"  从 {filename} 读取 {len(data_list)} 条原始数据")
                
                for idx, item in enumerate(data_list):
                    try:
                        # 转换为统一格式
                        processed_item = {
                            "instruction": item.get("instruction", ""),
                            "sequence": item.get("input", ""),
                            "response": item.get("output", ""),
                            "metadata": {
                                "source": f"Mol-Instructions/{json_file.stem}",
                                "split": item.get("metadata", {}).get("split", "unknown"),
                                "modality": "protein",
                                "task_type": json_file.stem
                            }
                        }
                        
                        # 验证数据
                        is_valid, error_msg = validator.validate_item(processed_item, "Mol-Instructions")
                        
                        # 额外验证序列内容
                        if is_valid and not validator.validate_sequence_content(
                            processed_item['sequence'], 'protein'
                        ):
                            is_valid = False
                            error_msg = "序列包含过多非法字符"
                        
                        statistics.add_item("Mol-Instructions", processed_item, is_valid, error_msg)
                        
                        if is_valid:
                            all_data.append(processed_item)
                        else:
                            if idx < 3:  # 只记录前几个错误的详细信息
                                logger.warning(f"  第 {idx+1} 条数据验证失败: {error_msg}")
                    
                    except Exception as e:
                        logger.warning(f"  处理第 {idx+1} 条数据时出错: {str(e)}")
                        statistics.stats["Mol-Instructions"]['invalid'] += 1
                        continue
                
                logger.info(f"  从 {filename} 成功处理 {sum(1 for item in all_data if item['metadata']['source'] == f'Mol-Instructions/{json_file.stem}')} 条有效数据")
            
            except json.JSONDecodeError as e:
                logger.error(f"  JSON解析失败: {filename}, 错误: {e}")
                continue
            except Exception as e:
                logger.error(f"  处理文件失败: {filename}, 错误: {e}")
                logger.error(traceback.format_exc())
                continue
        
        logger.info(f"\nMol-Instructions 处理完成: 共 {len(all_data)} 条有效数据")
        logger.info(statistics.generate_report())
        return all_data
    
    except Exception as e:
        logger.error(f"处理 Mol-Instructions 时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def process_UniProtQA() -> List[Dict[str, Any]]:
    """
    处理 UniProtQA 数据集
    
    数据来源: pharmolix_uniprotqa/train.json, test.json
    格式: 嵌套字典 {protein_id: {sequence: str, data: [[q, a], ...]}}
    
    返回: 统一格式的数据列表
    """
    logger.info("="*80)
    logger.info("开始处理 UniProtQA 数据集")
    logger.info("="*80)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "datasets" / "pharmolix_uniprotqa"
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []
    
    json_files = ["train.json", "test.json"]
    all_data = []
    validator = DataValidator()
    statistics = DataStatistics()
    
    for json_filename in json_files:
        json_path = data_dir / json_filename
        if not json_path.exists():
            logger.warning(f"文件不存在: {json_path}")
            continue
        
        split_name = json_filename.replace(".json", "")
        logger.info(f"处理文件: {json_filename}")
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data_dict = json.load(f)
            
            logger.info(f"  读取 {len(data_dict)} 个蛋白质条目")
            
            file_count = 0
            for protein_id, protein_data in data_dict.items():
                try:
                    if not isinstance(protein_data, dict):
                        logger.warning(f"  蛋白质 {protein_id} 的数据格式不正确")
                        continue
                    
                    sequence = protein_data.get("sequence", "")
                    data_list = protein_data.get("data", [])
                    
                    if not sequence or not data_list:
                        continue
                    
                    for qa_idx, qa_pair in enumerate(data_list):
                        if not isinstance(qa_pair, list) or len(qa_pair) != 2:
                            continue
                        
                        instruction, response = qa_pair
                        
                        processed_item = {
                            "instruction": str(instruction),
                            "sequence": str(sequence),
                            "response": str(response),
                            "metadata": {
                                "source": "UniProtQA",
                                "split": split_name,
                                "modality": "protein",
                                "protein_id": protein_id
                            }
                        }
                        
                        # 验证数据
                        is_valid, error_msg = validator.validate_item(processed_item, "UniProtQA")
                        
                        if is_valid and not validator.validate_sequence_content(
                            processed_item['sequence'], 'protein'
                        ):
                            is_valid = False
                            error_msg = "序列包含过多非法字符"
                        
                        statistics.add_item("UniProtQA", processed_item, is_valid, error_msg)
                        
                        if is_valid:
                            all_data.append(processed_item)
                            file_count += 1
                
                except Exception as e:
                    logger.warning(f"  处理蛋白质 {protein_id} 时出错: {str(e)}")
                    continue
            
            logger.info(f"  从 {json_filename} 成功处理 {file_count} 条有效数据")
        
        except json.JSONDecodeError as e:
            logger.error(f"  JSON解析失败: {json_filename}, 错误: {e}")
            continue
        except Exception as e:
            logger.error(f"  处理文件失败: {json_filename}, 错误: {e}")
            logger.error(traceback.format_exc())
            continue
    
    logger.info(f"\nUniProtQA 处理完成: 共 {len(all_data)} 条有效数据")
    logger.info(statistics.generate_report())
    return all_data


def process_Pika_DS() -> List[Dict[str, Any]]:
    """
    处理 Pika-DS 数据集
    
    数据来源: 
        - emcarrami_pika_ds/dataset/pika_sequences.csv
        - emcarrami_pika_ds/dataset/pika_annotations.csv
        - emcarrami_pika_ds/splits/pika_evogroup_split.csv
    
    格式: CSV文件,需要合并三个文件的信息
    
    返回: 统一格式的数据列表
    """
    logger.info("="*80)
    logger.info("开始处理 Pika-DS 数据集")
    logger.info("="*80)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "datasets" / "emcarrami_pika_ds" / "dataset"
    split_dir = base_dir / "datasets" / "emcarrami_pika_ds" / "splits"
    
    # 检查文件
    sequences_file = data_dir / "pika_sequences.csv"
    annotations_file = data_dir / "pika_annotations.csv"
    splits_file = split_dir / "pika_evogroup_split.csv"
    
    for file_path in [sequences_file, annotations_file, splits_file]:
        if not file_path.exists():
            logger.error(f"文件不存在: {file_path}")
            return []
    
    try:
        logger.info("读取CSV文件...")
        
        # 读取文件
        sequences_df = pd.read_csv(sequences_file)
        annotations_df = pd.read_csv(annotations_file)
        splits_df = pd.read_csv(splits_file)
        
        logger.info(f"  序列文件: {len(sequences_df)} 条")
        logger.info(f"  注释文件: {len(annotations_df)} 条")
        logger.info(f"  分割文件: {len(splits_df)} 条")
        
        # 验证文件一致性
        if len(sequences_df) != len(splits_df):
            logger.error(f"序列文件和分割文件行数不一致: {len(sequences_df)} vs {len(splits_df)}")
            return []
        
        # 排序确保一致性
        sequences_df = sequences_df.sort_values('uniprot_id').reset_index(drop=True)
        splits_df = splits_df.sort_values('uniprot_id').reset_index(drop=True)
        
        if not sequences_df['uniprot_id'].equals(splits_df['uniprot_id']):
            logger.error("序列文件和分割文件的 uniprot_id 不一致")
            return []
        
        logger.info("文件验证通过")
        
        # 筛选QA注释
        qa_annotations = annotations_df[annotations_df['type'] == 'qa'].copy()
        qa_annotations = qa_annotations.sort_values('uniprot_id').reset_index(drop=True)
        logger.info(f"找到 {len(qa_annotations)} 条QA注释")
        
        all_data = []
        validator = DataValidator()
        statistics = DataStatistics()
        
        # 统计信息
        skip_reasons = Counter()
        processed_count = 0
        
        # 使用双指针算法合并数据
        seq_idx = 0
        ann_idx = 0
        
        while seq_idx < len(sequences_df):
            seq_uniprot_id = sequences_df.iloc[seq_idx]['uniprot_id']
            sequence = sequences_df.iloc[seq_idx]['sequence']
            split_label = splits_df.iloc[seq_idx]['split']
            
            # 移动注释指针到当前序列
            while ann_idx < len(qa_annotations) and qa_annotations.iloc[ann_idx]['uniprot_id'] < seq_uniprot_id:
                ann_idx += 1
            
            # 处理所有匹配的注释
            temp_ann_idx = ann_idx
            while temp_ann_idx < len(qa_annotations) and qa_annotations.iloc[temp_ann_idx]['uniprot_id'] == seq_uniprot_id:
                annotation_text = qa_annotations.iloc[temp_ann_idx]['annotation']
                processed_count += 1
                
                if processed_count % 10000 == 0:
                    logger.info(f"  已处理 {processed_count:,} 条注释...")
                
                # 数据质量检查
                question_mark_count = annotation_text.count("?")
                if question_mark_count == 0:
                    skip_reasons['无问号'] += 1
                    temp_ann_idx += 1
                    continue
                elif question_mark_count > 1:
                    skip_reasons['多个问号'] += 1
                    temp_ann_idx += 1
                    continue
                
                if "[" in annotation_text:
                    skip_reasons['包含方括号'] += 1
                    temp_ann_idx += 1
                    continue
                
                # 分割问题和答案
                parts = annotation_text.split("?", 1)
                instruction = parts[0].strip() + "?"
                response = parts[1].strip()
                
                processed_item = {
                    "instruction": instruction,
                    "sequence": str(sequence),
                    "response": response,
                    "metadata": {
                        "source": "Pika-DS",
                        "split": split_label,
                        "modality": "protein",
                        "uniprot_id": seq_uniprot_id
                    }
                }
                
                # 验证数据
                is_valid, error_msg = validator.validate_item(processed_item, "Pika-DS")
                
                if is_valid and not validator.validate_sequence_content(
                    processed_item['sequence'], 'protein'
                ):
                    is_valid = False
                    error_msg = "序列包含过多非法字符"
                
                statistics.add_item("Pika-DS", processed_item, is_valid, error_msg)
                
                if is_valid:
                    all_data.append(processed_item)
                
                temp_ann_idx += 1
            
            seq_idx += 1
        
        # 输出跳过原因统计
        logger.info(f"\n数据过滤统计:")
        for reason, count in skip_reasons.most_common():
            logger.info(f"  - {reason}: {count:,}")
        
        logger.info(f"\nPika-DS 处理完成: 共 {len(all_data)} 条有效数据")
        logger.info(statistics.generate_report())
        return all_data
    
    except Exception as e:
        logger.error(f"处理 Pika-DS 时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def process_ChatNT() -> List[Dict[str, Any]]:
    """
    处理 ChatNT 数据集
    
    数据来源: instadeepai_chatnt_training_data/data/*.parquet
    格式: Parquet文件,包含sequence, exchanges, task_type等字段
    
    返回: 统一格式的数据列表
    """
    logger.info("="*80)
    logger.info("开始处理 ChatNT 数据集")
    logger.info("="*80)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "datasets" / "instadeepai_chatnt_training_data" / "data"
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []
    
    # 查找所有parquet文件
    parquet_files = sorted(data_dir.glob("*.parquet"))
    if not parquet_files:
        logger.error(f"未找到parquet文件: {data_dir}")
        return []
    
    logger.info(f"找到 {len(parquet_files)} 个parquet文件")
    
    try:
        all_data = []
        validator = DataValidator()
        statistics = DataStatistics()
        
        total_processed = 0
        total_invalid_exchange = 0
        
        for parquet_file in parquet_files:
            logger.info(f"处理文件: {parquet_file.name}")
            
            # 确定split标签
            if "test" in parquet_file.name:
                split_label = "test"
            elif "train" in parquet_file.name:
                split_label = "train"
            else:
                split_label = "unknown"
            
            try:
                df = pd.read_parquet(parquet_file)
                logger.info(f"  读取 {len(df)} 条数据")
                
                file_count = 0
                for idx in range(len(df)):
                    row = df.iloc[idx]
                    total_processed += 1
                    
                    if total_processed % 10000 == 0:
                        logger.info(f"  已处理 {total_processed:,} 条数据...")
                    
                    try:
                        # 解析exchanges
                        exchanges = row['exchanges']
                        if isinstance(exchanges, str):
                            exchanges = json.loads(exchanges)
                        
                        # 验证exchanges格式
                        if not isinstance(exchanges, list) or len(exchanges) != 2:
                            total_invalid_exchange += 1
                            continue
                        
                        if (exchanges[0].get('role') != 'USER' or 
                            exchanges[1].get('role') != 'ASSISTANT'):
                            total_invalid_exchange += 1
                            continue
                        
                        # 提取数据
                        instruction = exchanges[0]['message']
                        response = exchanges[1]['message']
                        sequence = row['sequence']
                        task_type = row.get('task_type', 'unknown')
                        modality = row.get('task_modality', 'dna')
                        
                        processed_item = {
                            "instruction": str(instruction),
                            "sequence": str(sequence),
                            "response": str(response),
                            "metadata": {
                                "source": f"ChatNT/{task_type}",
                                "split": split_label,
                                "modality": modality,
                                "task_type": task_type
                            }
                        }
                        
                        # 验证数据
                        is_valid, error_msg = validator.validate_item(processed_item, "ChatNT")
                        
                        if is_valid and not validator.validate_sequence_content(
                            processed_item['sequence'], modality
                        ):
                            is_valid = False
                            error_msg = "序列包含过多非法字符"
                        
                        statistics.add_item("ChatNT", processed_item, is_valid, error_msg)
                        
                        if is_valid:
                            all_data.append(processed_item)
                            file_count += 1
                    
                    except Exception as e:
                        if idx < 3:
                            logger.warning(f"  处理第 {idx+1} 条数据时出错: {str(e)}")
                        continue
                
                logger.info(f"  从 {parquet_file.name} 成功处理 {file_count} 条有效数据")
            
            except Exception as e:
                logger.error(f"  读取文件失败: {parquet_file.name}, 错误: {e}")
                continue
        
        logger.info(f"\n总计处理 {total_processed:,} 条数据")
        logger.info(f"无效的exchanges格式: {total_invalid_exchange:,}")
        logger.info(f"\nChatNT 处理完成: 共 {len(all_data)} 条有效数据")
        logger.info(statistics.generate_report())
        return all_data
    
    except Exception as e:
        logger.error(f"处理 ChatNT 时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def process_LLaMA_Gene() -> List[Dict[str, Any]]:
    """
    处理 LLaMA Gene 数据集
    
    数据来源: 
        - dnagpt_llama_gene_train_data/dna/*.json
        - dnagpt_llama_gene_train_data/protein/*.json
    
    格式: JSONL文件,每行一个JSON对象
    
    返回: 统一格式的数据列表
    """
    logger.info("="*80)
    logger.info("开始处理 LLaMA Gene 数据集")
    logger.info("="*80)
    
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / "datasets" / "dnagpt_llama_gene_train_data"
    
    if not data_dir.exists():
        logger.error(f"数据目录不存在: {data_dir}")
        return []
    
    # 定义要处理的文件
    files_to_process = [
        ("dna/sft_dna_eva.json", "dna", "test"),
        ("dna/sft_dna_train.json", "dna", "train"),
        ("protein/protein_sft_train.json", "protein", "train"),
        ("protein/sft_protein_eva.json", "protein", "test")
    ]
    
    all_data = []
    validator = DataValidator()
    statistics = DataStatistics()
    
    for file_path, modality, split in files_to_process:
        full_path = data_dir / file_path
        
        if not full_path.exists():
            logger.warning(f"文件不存在: {file_path}")
            continue
        
        logger.info(f"处理文件: {file_path}")
        
        valid_count = 0
        invalid_count = 0
        line_count = 0
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    line_count += 1
                    if line_count % 10000 == 0:
                        logger.info(f"  已处理 {line_count:,} 行...")
                    
                    try:
                        # 解析JSON
                        item = json.loads(line)
                        
                        # 验证必需字段
                        if not all(key in item for key in ['instruction', 'input', 'output']):
                            logger.warning(f"  第 {line_num} 行缺少必需字段")
                            invalid_count += 1
                            continue
                        
                        # 转换为统一格式
                        processed_item = {
                            "instruction": item["instruction"],
                            "sequence": item["input"],
                            "response": item["output"],
                            "metadata": {
                                "source": f"LLaMA-Gene/{modality}",
                                "split": split,
                                "modality": modality
                            }
                        }
                        
                        # 验证数据
                        is_valid, error_msg = validator.validate_item(processed_item, "LLaMA-Gene")
                        
                        if is_valid and not validator.validate_sequence_content(
                            processed_item['sequence'], modality
                        ):
                            is_valid = False
                            error_msg = "序列包含过多非法字符"
                        
                        statistics.add_item("LLaMA-Gene", processed_item, is_valid, error_msg)
                        
                        if is_valid:
                            all_data.append(processed_item)
                            valid_count += 1
                        else:
                            invalid_count += 1
                    
                    except json.JSONDecodeError as e:
                        if invalid_count < 3:
                            logger.warning(f"  第 {line_num} 行JSON解析失败: {e}")
                        invalid_count += 1
                    except Exception as e:
                        if invalid_count < 3:
                            logger.warning(f"  第 {line_num} 行处理失败: {e}")
                        invalid_count += 1
            
            logger.info(f"  {file_path}: 有效 {valid_count:,} 行, 无效 {invalid_count:,} 行")
        
        except Exception as e:
            logger.error(f"  读取文件失败: {file_path}, 错误: {e}")
            continue
    
    logger.info(f"\nLLaMA Gene 处理完成: 共 {len(all_data)} 条有效数据")
    logger.info(statistics.generate_report())
    return all_data


def process_Biology_Instructions() -> List[Dict[str, Any]]:
    """
    处理 Biology Instructions 数据集
    
    数据来源: gdrive_1OC3VpPKSQ0VHd9ZeZhnxI8EA2wTdrBg5/stage2_train.jsonl
    格式: JSONL文件,包含RNA-protein交互等任务
    
    返回: 统一格式的数据列表
    """
    logger.info("="*80)
    logger.info("开始处理 Biology Instructions 数据集")
    logger.info("="*80)
    
    base_dir = Path(__file__).parent.parent
    data_file = base_dir / "datasets" / "gdrive_1OC3VpPKSQ0VHd9ZeZhnxI8EA2wTdrBg5" / "stage2_train.jsonl"
    
    if not data_file.exists():
        logger.error(f"文件不存在: {data_file}")
        return []
    
    try:
        all_data = []
        validator = DataValidator()
        statistics = DataStatistics()
        
        line_count = 0
        valid_count = 0
        invalid_count = 0
        
        logger.info(f"处理文件: {data_file.name}")
        
        with open(data_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                line_count += 1
                if line_count % 100000 == 0:
                    logger.info(f"  已处理 {line_count:,} 行...")
                
                try:
                    # 解析JSON
                    item = json.loads(line)
                    
                    # 提取信息
                    input_text = item.get('input', '')
                    output_text = item.get('output', '')
                    task = item.get('task', 'unknown')
                    
                    # 尝试从input中提取序列
                    # 这个数据集的格式比较特殊,可能需要特殊处理
                    # 暂时将整个input作为instruction
                    processed_item = {
                        "instruction": input_text,
                        "sequence": "",  # 这个数据集可能没有独立的序列字段
                        "response": output_text,
                        "metadata": {
                            "source": "Biology-Instructions",
                            "split": "train",
                            "modality": "mixed",  # 可能包含多种模态
                            "task_type": task
                        }
                    }
                    
                    # 对于这个数据集,我们放宽序列长度的限制
                    # 因为它可能不是传统的序列格式
                    if len(input_text) > 0 and len(output_text) > 0:
                        all_data.append(processed_item)
                        valid_count += 1
                    else:
                        invalid_count += 1
                
                except json.JSONDecodeError as e:
                    if invalid_count < 3:
                        logger.warning(f"  第 {line_num} 行JSON解析失败: {e}")
                    invalid_count += 1
                except Exception as e:
                    if invalid_count < 3:
                        logger.warning(f"  第 {line_num} 行处理失败: {e}")
                    invalid_count += 1
        
        logger.info(f"\n总计处理 {line_count:,} 行")
        logger.info(f"有效: {valid_count:,}, 无效: {invalid_count:,}")
        logger.info(f"\nBiology Instructions 处理完成: 共 {len(all_data)} 条有效数据")
        
        return all_data
    
    except Exception as e:
        logger.error(f"处理 Biology Instructions 时发生错误: {str(e)}")
        logger.error(traceback.format_exc())
        return []


def save_to_json(data_list: List[Dict[str, Any]], output_file: str = 'dataset.json'):
    """
    将数据列表保存为JSON文件
    
    Args:
        data_list: 数据列表
        output_file: 输出文件名
    """
    if not data_list:
        logger.warning(f"数据列表为空,跳过保存: {output_file}")
        return
    
    base_dir = Path(__file__).parent.parent
    output_dir = base_dir / "datasets" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / output_file
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data_list, f, ensure_ascii=False, indent=2)
        
        file_size = output_path.stat().st_size / (1024 * 1024)  # MB
        logger.info(f"数据已保存到: {output_path}")
        logger.info(f"  文件大小: {file_size:.2f} MB")
        logger.info(f"  数据条数: {len(data_list):,}")
    
    except Exception as e:
        logger.error(f"保存文件时发生错误: {str(e)}")
        logger.error(traceback.format_exc())


def main():
    """主函数"""
    logger.info("\n" + "="*80)
    logger.info("开始数据处理流程")
    logger.info("="*80 + "\n")
    
    # 处理所有数据集
    datasets = [
        ("Mol-Instructions", process_Mol_Instructions, "dataset_Mol_Instructions.json"),
        ("UniProtQA", process_UniProtQA, "dataset_UniProtQA.json"),
        ("Pika-DS", process_Pika_DS, "dataset_Pika_DS.json"),
        ("ChatNT", process_ChatNT, "dataset_ChatNT.json"),
        ("LLaMA-Gene", process_LLaMA_Gene, "dataset_LLaMA_Gene.json"),
        ("Biology-Instructions", process_Biology_Instructions, "dataset_Biology_Instructions.json"),
    ]
    
    all_results = {}
    
    for name, process_func, output_file in datasets:
        try:
            logger.info(f"\n{'='*80}")
            logger.info(f"处理数据集: {name}")
            logger.info(f"{'='*80}\n")
            
            data = process_func()
            save_to_json(data, output_file)
            all_results[name] = len(data)
            
            logger.info(f"\n{name} 处理完成\n")
        
        except Exception as e:
            logger.error(f"处理 {name} 时发生严重错误: {str(e)}")
            logger.error(traceback.format_exc())
            all_results[name] = 0
    
    # 生成总结报告
    logger.info("\n" + "="*80)
    logger.info("数据处理总结")
    logger.info("="*80)
    
    total = 0
    for name, count in all_results.items():
        logger.info(f"{name:30s}: {count:>10,} 条")
        total += count
    
    logger.info("-"*80)
    logger.info(f"{'总计':30s}: {total:>10,} 条")
    logger.info("="*80 + "\n")
    
    logger.info("所有数据集处理完成!")


if __name__ == "__main__":
    main()
