# Converted from notebook: state/scripts/robin/de_embeddings.ipynb
# Cell 1 (code)
import anndata as ad
from pathlib import Path

# Cell 2 (markdown)
# ew: 
# but in any case de_df[["gene_col", "de_embedding"]].to_csv()

# Cell 3 (code)
import numpy as np
import pandas as pd
from anndata import AnnData
from pdex import parallel_differential_expression
from tqdm import tqdm


def compute_de_labels(adata: AnnData,
                           perturb_col: str = "target_gene",
                           cell_type_col: str = "cell_type",
                           control_var: str = "non-targeting",
                           alpha: float = 0.05) -> None:
    """
    Precompute DE gene labels (+1, -1, 0) using pdex.
    Stores results in adata.uns['de_labels']. 

    Parameters
    ----------
    adata : AnnData
        Input AnnData object.
    perturb_col : str
        Column in obs indicating perturbation/target gene.
    cell_type_col : str
        Column in obs indicating cell type.
    control_var : str
        Name of the control group in perturb_col.
    alpha : float
        FDR threshold for significance.
    """



    cell_types = adata.obs[cell_type_col].unique()
    target_genes = adata.obs[perturb_col].unique()

    result = {}

    # iterate over cell types
    for ct in tqdm(cell_types, desc="Computing DE labels with pdex"):
        # subset AnnData to current cell type
        adata_ct = adata[adata.obs[cell_type_col] == ct].copy()
        
        # Run pdex differential expression
        de_df = parallel_differential_expression(
            adata_ct,
            reference=control_var,
            groupby_key=perturb_col,
            metric="wilcoxon",  # default
            num_workers=20,
            batch_size=100,
        )

        # if "fdr" < 0.05 and "percent_change" > 0 set to 1, if "fdr" < 0.05 and "percent_change" < 0 set to -1, else 0
        de_df["de_embedding"] = de_df.apply(
            lambda row: 1 if row["fdr"] < alpha and row["percent_change"] > 0 else (-1 if row["fdr"] < alpha and row["percent_change"] < 0 else 0),
            axis=1
        )

        # Create a DataFrame from the DE results

        
        for target in target_genes:
            # filter for current target gene
            de_target = de_df[de_df["target"] == target]
            if not de_target.empty:
                result[f"{ct}_{target}"] = de_target["de_embedding"].values
            else:
                result[f"{ct}_{target}"] = 0

        
        #print(de_df[["gene_col", "de_embedding"]])
        

    adata.uns['de_labels'] = result
    print(f"Stored DE results in adata.uns['de_labels'] with {len(adata.uns['de_labels'])} entries.")

    

          

# Cell 4 (code)
path="/buffer/ag_bsc/pmsb_workflows_2025/team4_ensemble_assembly/robin/arc-state/state/datasets/base_dataset/hepg2.h5"
adata = ad.read_h5ad(path)#we should be able to load this faster, by using the .csv instead! 

#if 'de_labels' not in adata.uns:
#    compute_de_labels(adata, perturb_col="target_gene", cell_type_col="cell_type", control_var="non-targeting", alpha=0.05)
#print(adata.uns[["gene_col", "de_embedding"]])

de_df = pd.DataFrame(adata.uns['de_labels']).T.reset_index()
de_df.columns = ['cell_type_target_gene', 'de_embedding']
print(de_df.head())

# Cell 5 (code)
path_to_test="/buffer/ag_bsc/pmsb_workflows_2025/team4_ensemble_assembly/robin/arc-state/state/datasets/base_dataset/DES_test/"
path_to_compare="/buffer/ag_bsc/pmsb_workflows_2025/team4_ensemble_assembly/robin/arc-state/state/datasets/base_dataset/DES/"

# collect csv files from both directories
test_dir = Path(path_to_test)
compare_dir = Path(path_to_compare)

test_files = {p.name: p for p in test_dir.glob("*.csv")}
compare_files = {p.name: p for p in compare_dir.glob("*.csv")}

print(f"Test dir: {test_dir} -> {len(test_files)} CSV files")
print(f"Compare dir: {compare_dir} -> {len(compare_files)} CSV files")

test_names = set(test_files.keys())
compare_names = set(compare_files.keys())

def _normalize(fname: str) -> str:
    # remove leading "hepg2_" or "hepg_" if present
    if fname.startswith("hepg2_"):
        return fname[len("hepg2_"):]
    if fname.startswith("hepg_"):
        return fname[len("hepg_"):]
    return fname

# rebuild mappings keyed by normalized names (e.g. "ARPC2.csv")
_new_test_files = {}
for name, p in test_files.items():
    nk = _normalize(name)
    if nk in _new_test_files:
        print(f"Warning: duplicate normalized test name {nk}; keeping first ({_new_test_files[nk].name})")
    else:
        _new_test_files[nk] = p

_new_compare_files = {}
for name, p in compare_files.items():
    nk = _normalize(name)
    if nk in _new_compare_files:
        print(f"Warning: duplicate normalized compare name {nk}; keeping first ({_new_compare_files[nk].name})")
    else:
        _new_compare_files[nk] = p

# replace original dicts and name sets with normalized-key versions
test_files = _new_test_files
compare_files = _new_compare_files
test_names = set(test_files.keys())
compare_names = set(compare_files.keys())

print(f"After normalizing prefixes -> test files: {len(test_files)}, compare files: {len(compare_files)}")

common_names = sorted(test_names & compare_names)
only_in_test = sorted(test_names - compare_names)
only_in_compare = sorted(compare_names - test_names)

print(f"Common files: {len(common_names)}")
if only_in_test:
    print(f"Files only in test ({len(only_in_test)}): {only_in_test}")
if only_in_compare:
    print(f"Files only in compare ({len(only_in_compare)}): {only_in_compare}")

# compare contents for common files
for name in common_names:
    p_test = test_files[name]
    p_cmp = compare_files[name]

    # fast byte-wise check first
    try:
        if p_test.read_bytes() == p_cmp.read_bytes():
            print(f"{name}: IDENTICAL (byte-wise)")
            continue
    except Exception as e:
        print(f"{name}: error reading bytes: {e}")

    # fall back to reading as DataFrame and compare
    try:
        df_test = pd.read_csv(p_test)
        df_cmp = pd.read_csv(p_cmp)
    except Exception as e:
        print(f"{name}: ERROR reading CSVs as DataFrame: {e}")
        continue

    if df_test.equals(df_cmp):
        print(f"{name}: IDENTICAL (DataFrame.equals)")
        continue

    # summarize differences
    diffs = []
    if df_test.shape != df_cmp.shape:
        diffs.append(f"shape differs: test {df_test.shape} vs compare {df_cmp.shape}")
    if list(df_test.columns) != list(df_cmp.columns):
        diffs.append("columns differ")

    # compute a simple cell-wise diff for common columns and rows
    common_cols = [c for c in df_test.columns if c in df_cmp.columns]
    if common_cols:
        a = df_test[common_cols].reset_index(drop=True).fillna("__NA__").astype(str)
        b = df_cmp[common_cols].reset_index(drop=True).fillna("__NA__").astype(str)
        rows = min(len(a), len(b))
        if rows > 0:
            comp = (a.iloc[:rows] != b.iloc[:rows])
            n_diff_cells = int(comp.values.sum())
            diffs.append(f"{n_diff_cells} differing cells in first {rows} rows across {len(common_cols)} common columns")
        else:
            diffs.append("no overlapping rows to compare")
    else:
        diffs.append("no common columns to compare")

    print(f"{name}: DIFFER ({'; '.join(diffs)})")



# Cell 6 (code)
import numpy as np
import pandas as pd
from anndata import AnnData
from tqdm import tqdm

def compute_expression_ranks(
    adata: AnnData,
    groupby_key: str = "target_gene",
    celltype_key: str = "cell_type",
) -> None:
    """
    Precompute gene expression ranks for each (cell_type, target_gene).
    Stores results in adata.uns['expr_ranks']. 

    Each entry is a vector of length n_vars where
        rank[i] = rank of gene i's mean expression within that subset.
    """
    results = {}

    cell_types = adata.obs[celltype_key].unique()
    target_genes = adata.obs[groupby_key].unique()

    # single progress bar over all combinations
    combos = [(ct, tg) for ct in cell_types for tg in target_genes]
    for ct, tg in tqdm(combos, desc="Computing expression ranks", total=len(combos)):
        # subset to matching cells
        mask = (adata.obs[celltype_key] == ct) & (adata.obs[groupby_key] == tg)
        if mask.sum() == 0:
            continue

        subset = adata[mask]

        # mean expression across cells for each gene
        mean_expr = np.asarray(subset.X.mean(axis=0)).ravel()

        # compute ranks (highest expression = rank 1)
        ranks = pd.Series(mean_expr).rank(method="first", ascending=False).to_numpy()

        results[f"{ct}_{tg}"] = ranks

    adata.uns["rank_embedding"] = results
    print(f"Stored rank vectors in adata.uns['rank_embedding'] with {len(results)} entries.")

# Cell 7 (code)

data_dir = Path("/raid/kreid/v_cell/competition_support_set")

# Cell 8 (code)
anndata_paths = [file for file in data_dir.glob("*.h5")]

# Cell 9 (code)
for path in anndata_paths:
    print(f"Processing {path.name}")
    adata = ad.read_h5ad(path)

    # Compute DE labels
    compute_de_labels(adata, perturb_col="target_gene", cell_type_col="cell_type", control_var="non-targeting", alpha=0.05)

    # Compute expression ranks
    compute_expression_ranks(adata, groupby_key="target_gene", celltype_key="cell_type")

    # Save updated AnnData
    adata.write_h5ad(path)  # Overwrite original file or save to a new path if needed

# Cell 10 (empty)
