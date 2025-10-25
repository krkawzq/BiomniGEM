import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd

from cellm.process.template import Template

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, nargs="+", required=True, help="输入数据路径")
parser.add_argument("--save_path", type=str, required=True, help="输出路径")
parser.add_argument("--template_path", type=str, required=True, help="模板路径")
parser.add_argument("--meta_path", type=str, required=True, help="meta.json 路径")
parser.add_argument("--cell_line", type=str, required=True, help="细胞系名称")
parser.add_argument("--clip", type=int, required=True, help="截取基因数")
args = parser.parse_args()

def main():
    with open(args.meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    reference = meta["reference"]
    gene_list = meta["groups"][reference]["gene_list"][: args.clip]

    cell_sentence = "<cell>" + ", ".join(gene_list) + "</cell>"
    appends = {
        "cell_line": args.cell_line,
        # 兼容模板中使用转义下划线的占位符 {cell\_line}
        "cell\\_line": args.cell_line,
        "gene_list": gene_list,
        "n_genes": len(gene_list),
        "cell_sentence": cell_sentence,
    }

    files = args.data_path
    if not files:
        raise FileNotFoundError(f"未找到匹配文件: {args.data_path}")

    # 确保输出目录存在
    save_dir = Path(args.save_path).parent
    os.makedirs(save_dir, exist_ok=True)
    
    template = Template(args.template_path)

    for f in files:
        fpath = Path(f)
        df = pd.read_csv(f)
        
        df = df[df["label"] != "not_use"]

        # 补齐模板所需列：gene（取 feature），perturbation（取 target）
        if "gene" not in df.columns and "feature" in df.columns:
            df["gene"] = df["feature"]
        if "perturbation" not in df.columns and "target" in df.columns:
            df["perturbation"] = df["target"]
        # 兼容 crispri/de_cls.json 使用的占位符名 {candidants}
        if "candidants" not in df.columns and "candidates" in df.columns:
            df["candidants"] = df["candidates"]

        out_df = template.apply(df, appends=appends)

        out_df.to_csv(args.save_path, index=False)
        print(f"✅ {f} 已处理并保存到 {args.save_path}")


if __name__ == "__main__":
    main()
