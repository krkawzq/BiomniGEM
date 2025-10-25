from cellm.process.template import Template
import argparse
import os
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True, help="输入数据路径")
parser.add_argument("--save_path", type=str, required=True, help="输出路径")
parser.add_argument("--template_path", type=str, required=True, help="模板路径")
args = parser.parse_args()

def main():
    # -- ensure path
    if not args.save_path.endswith(".csv"):
        print(f"Warning: Save path {args.save_path} is not a csv file")
        exit(1)
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    # -- load template and data
    print("--------------------------------")
    print(f"Loading template from {args.template_path}")
    print(f"Loading data from {args.data_path}")
    print("--------------------------------")
    template = Template(args.template_path)
    df = pd.read_csv(args.data_path)
    
    # -- format cell sentence
    print("Formatting cell sentences...")
    print("--------------------------------")
    def format_cellsentence(sentence):
        sentence = sentence.split(" ")[:200]
        sentence = ", ".join(sentence)
        sentence = "<cell>" + sentence + "</cell>"
        return sentence
    df["cell_sentence"] = df["cell_sentence"].apply(format_cellsentence)
    df["n_genes"] = 200
    
    # -- extract cell type list
    print("Extracting cell type list...")
    print("--------------------------------")
    cell_type_list = df["cell_type"].unique().tolist()
    print(f"Found {len(cell_type_list)} unique cell types")
    
    # -- apply template
    print("Applying template...")
    print("--------------------------------")
    df = template.apply(df, cell_type_list=cell_type_list)
    
    df["task"] = df["user"]
    df["answer"] = df["label"]
    
    # -- save result
    print(f"Saving to {args.save_path}...")
    print("--------------------------------")
    df.to_csv(args.save_path, index=False)
    print("--------------------------------")
    print("all done!")

if __name__ == "__main__":
    main()
