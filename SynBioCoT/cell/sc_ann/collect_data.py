import pandas as pd
import anndata as ad
import os
import cellm.process as cp
from cellm.process.utils import top_expr_genes
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", nargs="+", type=str, required=True)
parser.add_argument("--save_path", type=str, required=True)
parser.add_argument("--cell_type_map_path", type=str, required=True)
args = parser.parse_args()

process_pipe = cp.Pipeline([
    cp.StandardizeGeneName(),
    cp.CustomQC(
        mt_pct=(0, 20),
        ribo_pct=(0, 50),
        min_gene_count=200,
        max_gene_count=6000,
        total_counts_min=500,
        maintain_qc=False,
        in_place=True
    )
])

def apply(adatas, pipe):
    new = []
    for adata in adatas:
        new.append(pipe(adata))
    return new

def ensure_path(path):
    dir_path = os.path.dirname(path)
    if dir_path and not os.path.exists(dir_path):
        os.makedirs(dir_path)

def main():
    # -- checks
    print(f"Checking data paths: {args.data_path}")
    print(f"Checking save path: {args.save_path}")
    print(f"Checking cell type map path: {args.cell_type_map_path}")
    print("--------------------------------")
    paths = []
    for path in args.data_path:
        if not os.path.exists(path):
            print(f"Warning: Data path {path} not found")
            continue
        if not path.endswith(".h5ad") and not path.endswith(".h5"):
            print(f"Warning: Data path {path} is not a h5ad file")
            continue
        paths.append(path)
    if len(paths) == 0:
        print("No valid data paths provided")
        return
    if not args.save_path.endswith(".csv"):
        print(f"Warning: Save path {args.save_path} is not a csv file")
        exit(1)
    ensure_path(args.save_path)
    
    print("--------------------------------")
    print(f"Loading cell type map from {args.cell_type_map_path}")
    print("--------------------------------")
    # -- load cell type map
    cell_type_map = json.load(open(args.cell_type_map_path, "r"))
        
    # -- standardize gene name & filter cells with qc
    print("Processing data...")
    print("--------------------------------")
    adatas = []
    for path in paths:
        adata = ad.read_h5ad(path)
        adata = process_pipe(adata)
        adata.name = path.split("/")[-1].split(".")[0]
        adatas.append(adata)
        
    # -- calc gene set & calc cell sentence
    print("Calculating gene set...")
    print("--------------------------------")
    gene_set = top_expr_genes(adatas)
    print("Calculating cell sentence...")
    print("--------------------------------")
    cell_sentence_pipe = cp.CellSentence(
        gene_set=gene_set,
        clip=2000,
        desc=True,
        output_column="cell_sentence",
        count_column="n_genes"
    )
    
    apply(adatas, cell_sentence_pipe)
    
    # -- save obs
    print("Saving obs...")
    print("--------------------------------")
    def maintain(df, name):
        new_df = df[["cell_sentence", "n_genes"]].copy()
        new_df["cell_type"] = df["cell_type"].map(cell_type_map)
        new_df["dataset"] = name
        new_df = new_df[new_df["cell_type"].notna()]
        return new_df
    dfs = [maintain(adata.obs, adata.name) for adata in adatas]
    df = pd.concat(dfs)
    df.to_csv(args.save_path, index=False)
    print("--------------------------------")
    print("all done!")

if __name__ == "__main__":
    main()