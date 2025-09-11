#!/usr/bin/env python
"""
Holdout Ground Truth Dataset Builder

This script creates evaluation datasets from TOML configuration files by:
1. Reading dataset paths and holdout perturbations from TOML
2. Loading individual .h5/.h5ad files for each cell type
3. Filtering to only holdout perturbations + controls
4. Combining into a single evaluation dataset
5. Outputting both .h5ad and count summary CSV

The script is designed to be robust against:
- Missing files or perturbations (skips gracefully)
- Inconsistent gene naming (requires real gene names)
- File format variations (handles both .h5 and .h5ad)
"""

import argparse
import os
import logging
from pathlib import Path
from typing import Dict, List, Set, Optional

import toml
import scanpy as sc
import pandas as pd
import anndata as ad
import numpy as np

# logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def load_toml_config(toml_path: str) -> Dict:
    """Load and validate TOML configuration file."""
    if not os.path.exists(toml_path):
        raise FileNotFoundError(f"TOML config not found: {toml_path}")
    
    config = toml.load(toml_path)
    logger.info(f"Loaded TOML config from: {toml_path}")
    
    # validate required sections
    if 'datasets' not in config:
        raise ValueError("TOML config missing required [datasets] section")
    if 'fewshot' not in config:
        raise ValueError("TOML config missing required [fewshot] section")
    
    return config


def extract_holdout_perturbations(config: Dict, split: str) -> Dict[str, Dict[str, Set[str]]]:
    """
    Extract holdout perturbations from TOML config.
    
    Returns:
        Dict mapping dataset_name -> cell_type -> set of perturbations
        
    Example:
        {
            'replogle_h1': {
                'k562': {'ACLY', 'BRD9', ...},
                'jurkat': {'ACLY', 'BRD9', ...}
            }
        }
    """
    holdouts = {}
    fewshot_section = config['fewshot']
    
    for key, perturbation_lists in fewshot_section.items():
        # Parse key format: "dataset_name.cell_type"
        if '.' not in key:
            logger.warning(f"Skipping malformed fewshot key (no cell type): {key}")
            continue
            
        dataset_name, cell_type = key.split('.', 1)
        
        # Get perturbations for the requested split
        if split not in perturbation_lists:
            logger.warning(f"Split '{split}' not found in {key}, skipping")
            continue
            
        perturbations = set(perturbation_lists[split])
        if not perturbations:
            logger.warning(f"No perturbations found for {key}.{split}, skipping")
            continue
        
        # Store in nested dict structure
        if dataset_name not in holdouts:
            holdouts[dataset_name] = {}
        holdouts[dataset_name][cell_type] = perturbations
        
        logger.info(f"Found {len(perturbations)} {split} perturbations for {dataset_name}.{cell_type}")
    
    return holdouts


def find_dataset_files(config: Dict, holdouts: Dict[str, Dict[str, Set[str]]]) -> Dict[str, List[str]]:
    """
    Find .h5/.h5ad files for each dataset based on cell types needed.
    
    Returns:
        Dict mapping dataset_name -> list of file paths
    """
    dataset_files = {}
    datasets_section = config['datasets']
    
    for dataset_name in holdouts.keys():
        if dataset_name not in datasets_section:
            logger.error(f"Dataset '{dataset_name}' referenced in fewshot but not found in [datasets]")
            continue
            
        dataset_dir = datasets_section[dataset_name]
        if not os.path.isdir(dataset_dir):
            logger.error(f"Dataset directory not found: {dataset_dir}")
            continue
        
        # Find .h5/.h5ad files matching the cell types we need
        cell_types_needed = set(holdouts[dataset_name].keys())
        found_files = []
        
        dataset_path = Path(dataset_dir)
        for pattern in ['*.h5', '*.h5ad']:
            for file_path in dataset_path.glob(pattern):
                # Extract cell type from filename (assumes format like "k562.h5")
                file_stem = file_path.stem.lower()
                
                # Check if this file matches any needed cell type
                for cell_type in cell_types_needed:
                    if cell_type.lower() in file_stem:
                        found_files.append(str(file_path))
                        logger.info(f"Found file for {dataset_name}.{cell_type}: {file_path.name}")
                        break
        
        if found_files:
            dataset_files[dataset_name] = found_files
        else:
            logger.warning(f"No files found for dataset '{dataset_name}' in {dataset_dir}")
    
    return dataset_files


def load_and_validate_file(file_path: str, required_columns: List[str]) -> Optional[ad.AnnData]:
    """
    Load a single .h5/.h5ad file and validate it has required columns and real gene names.
    
    Returns:
        AnnData object if valid, None if should be skipped
    """
    try:
        # Load the file
        if file_path.endswith('.h5ad'):
            adata = sc.read_h5ad(file_path)
        else:
            # For .h5 files, use custom STATE format reader
            adata = _read_state_h5_file(file_path, required_columns)
            if adata is None:
                return None
        
        # Validate required columns exist
        missing_cols = [col for col in required_columns if col not in adata.obs.columns]
        if missing_cols:
            logger.warning(f"File {file_path} missing required columns {missing_cols}, skipping")
            return None
        
        # Validate gene names are real (not generic g0, g1, etc.)
        if adata.var_names is None or len(adata.var_names) == 0:
            logger.warning(f"File {file_path} has no gene names, skipping")
            return None
            
        # Check if gene names look generic (starts with 'g' followed by digits)
        first_few_genes = adata.var_names[:5].tolist()
        if all(name.startswith('g') and name[1:].isdigit() for name in first_few_genes):
            logger.warning(f"File {file_path} has generic gene names (g0, g1, ...), skipping")
            return None
        
        # Ensure unique names to prevent downstream issues
        adata.obs_names_make_unique()
        adata.var_names_make_unique()
        
        logger.info(f"Successfully loaded {file_path}: {adata.shape[0]} cells, {adata.shape[1]} genes")
        return adata
        
    except Exception as e:
        logger.warning(f"Error loading {file_path}: {e}")
        return None


def filter_to_holdouts(adata: ad.AnnData, holdout_perts: Set[str], 
                      pert_col: str, control_pert: str) -> ad.AnnData:
    """
    Filter AnnData to only holdout perturbations and controls.
    
    Returns filtered AnnData with only relevant cells.
    """
    # Find perturbations that actually exist in this file
    available_perts = set(adata.obs[pert_col].unique())
    found_holdouts = holdout_perts.intersection(available_perts)
    
    if not found_holdouts:
        logger.info(f"No holdout perturbations found in this file (requested: {len(holdout_perts)})")
        return None
    
    logger.info(f"Found {len(found_holdouts)}/{len(holdout_perts)} holdout perturbations in file")
    
    # Create mask for holdout perturbations + control
    holdout_mask = adata.obs[pert_col].isin(found_holdouts)
    control_mask = adata.obs[pert_col] == control_pert
    combined_mask = holdout_mask | control_mask
    
    if not combined_mask.any():
        logger.warning(f"No cells found matching holdouts or control '{control_pert}'")
        return None
    
    # Filter and annotate
    filtered = adata[combined_mask].copy()
    
    # Add metadata about what was found
    n_holdout_cells = holdout_mask.sum()
    n_control_cells = control_mask.sum()
    logger.info(f"Extracted {n_holdout_cells} holdout + {n_control_cells} control = {filtered.shape[0]} total cells")
    
    return filtered


def combine_datasets(dataset_parts: List[ad.AnnData]) -> ad.AnnData:
    """
    Combine multiple AnnData objects into one, ensuring gene alignment.
    
    Uses inner join to only keep genes present in ALL datasets.
    This prevents gene name mismatches that break evaluation.
    """
    if not dataset_parts:
        raise ValueError("No valid datasets to combine")
    
    logger.info(f"Combining {len(dataset_parts)} dataset parts...")
    
    # Log gene counts before combining
    gene_counts = [part.shape[1] for part in dataset_parts]
    logger.info(f"Gene counts per part: {gene_counts}")
    
    # Combine with inner join (only genes in ALL parts)
    combined = ad.concat(dataset_parts, axis=0, join='inner', merge='same')
    
    # Ensure unique observation names
    combined.obs_names_make_unique()
    
    logger.info(f"Combined dataset: {combined.shape[0]} cells × {combined.shape[1]} genes")
    logger.info(f"Genes after inner join: {combined.shape[1]} (intersection of all parts)")
    
    return combined


def generate_summary_stats(adata: ad.AnnData, pert_col: str, control_pert: str) -> pd.DataFrame:
    """Generate summary statistics of perturbations in the dataset."""
    # Count cells per perturbation
    pert_counts = adata.obs[pert_col].value_counts().reset_index()
    pert_counts.columns = ['target_gene', 'n_cells']
    
    # Separate controls from holdouts
    is_control = pert_counts['target_gene'] == control_pert
    holdout_counts = pert_counts[~is_control].sort_values('n_cells', ascending=False)
    control_counts = pert_counts[is_control]
    
    logger.info(f"Summary: {len(holdout_counts)} unique holdout perturbations, "
                f"{control_counts['n_cells'].sum() if len(control_counts) > 0 else 0} control cells")
    
    return holdout_counts


def _read_state_h5_file(file_path: str, required_columns: List[str]) -> Optional[ad.AnnData]:
    """Read STATE format .h5 files with custom structure."""
    import h5py
    from scipy.sparse import csr_matrix
    
    try:
        with h5py.File(file_path, "r") as f:
            # We need to get the dimensions of the dataset before we can read the data - the compset doesn't store shape
            obs_data = {}
            for col in required_columns:
                try:
                    obs_col = _read_obs_col(f, col)
                    obs_data[col] = obs_col
                except KeyError:
                    logger.warning(f"Required column '{col}' not found in {file_path}")
                    return None
            n_obs = len(obs_data[required_columns[0]])
            
            # Load gene names to get dimensions - we do this here so we can log it later. TODO: change when we do this when we are comfortable with its effectiveness
            gene_names = _read_gene_names(f)
            if gene_names is None:
                logger.warning(f"No gene names found in {file_path}")
                return None
            n_vars = len(gene_names)
            
            # Load expression matrix (prefer X_hvg, fallback to X)
            if f.get("obsm/X_hvg") is not None and isinstance(f["obsm/X_hvg"], h5py.Dataset):
                X = f["obsm/X_hvg"][:]
            elif f.get("X") is not None:
                xnode = f["X"]
                if isinstance(xnode, h5py.Dataset):
                    X = xnode[:]
                elif isinstance(xnode, h5py.Group) and all(k in xnode for k in ("data","indices","indptr")):
                    # CSR sparse matrix - infer shape if missing
                    data = xnode["data"][:]
                    indices = xnode["indices"][:]
                    indptr = xnode["indptr"][:]
                    if "shape" in xnode:
                        shape = tuple(xnode["shape"][:])
                    else:
                        shape = (n_obs, n_vars)  # infer from obs/var dimensions
                    X = csr_matrix((data, indices, indptr), shape=shape)
                else:
                    logger.warning(f"Unsupported X layout in {file_path}")
                    return None
            else:
                logger.warning(f"No X data found in {file_path}")
                return None
            
            # Validate dimensions
            if X.shape[0] != n_obs or X.shape[1] != n_vars:
                logger.warning(f"Shape mismatch in {file_path}: got {X.shape}, expected ({n_obs}, {n_vars})")
                return None
            
            # Create AnnData
            obs = pd.DataFrame(obs_data)
            obs.index = pd.Index([f"{Path(file_path).stem}_{i}" for i in range(obs.shape[0])], dtype=str)
            
            var = pd.DataFrame(index=pd.Index(gene_names, name="gene"))
            
            adata = ad.AnnData(X=X, obs=obs, var=var)
            return adata
            
    except Exception as e:
        logger.warning(f"Error reading STATE h5 file {file_path}: {e}")
        return None


def _read_obs_col(f, col: str) -> np.ndarray:
    """Read categorical or string observation column from h5py file."""
    import h5py
    
    # Try categorical storage: obs/<col>/{categories,codes}
    if f.get(f"obs/{col}") is not None and isinstance(f[f"obs/{col}"], h5py.Group):
        cats = f[f"obs/{col}/categories"][:].astype(str)
        codes = f[f"obs/{col}/codes"][:]
        return cats[codes]
    # Try flat string dataset: obs/<col>
    if f.get(f"obs/{col}") is not None and isinstance(f[f"obs/{col}"], h5py.Dataset):
        return f[f"obs/{col}"][:].astype(str)
    raise KeyError(f"obs/{col} not found")

def _read_gene_names(f) -> Optional[np.ndarray]:
    import h5py
    # Preferred explicit gene name field
    if f.get("var/gene_name") is not None and isinstance(f["var/gene_name"], h5py.Group):
        cats = f["var/gene_name/categories"][:].astype(str)
        codes = f["var/gene_name/codes"][:]
        return cats[codes]
    if f.get("var/gene_name") is not None and isinstance(f["var/gene_name"], h5py.Dataset):
        arr = f["var/gene_name"][:]
        return arr.astype(str)

    # AnnData default index locations
    for key in ("var/_index", "var/index"):
        if f.get(key) is not None and isinstance(f[key], h5py.Dataset):
            arr = f[key][:]
            # bytes -> str if needed
            if hasattr(arr, "dtype") and arr.dtype.kind in ("S", "O"):
                try:
                    arr = arr.astype(str)
                except Exception:
                    arr = np.array([x.decode("utf-8", "ignore") if isinstance(x, (bytes, bytearray)) else str(x) for x in arr])
            return arr

    # Nothing found
    return None

def main():
    parser = argparse.ArgumentParser(
        description="Build holdout ground truth dataset from TOML configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python prepare_holdout_ground_truth.py \\
        --toml_config examples/fewshot.toml \\
        --split test \\
        --output_dir ./eval_data \\
        --output_h5ad holdout_test.h5ad \\
        --output_csv holdout_test_counts.csv
        """
    )
    
    parser.add_argument('--toml_config', type=str, required=True,
                      help='Path to TOML configuration file')
    parser.add_argument('--split', type=str, required=True, choices=['val', 'test'],
                      help='Which split to extract (val or test)')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for results')
    parser.add_argument('--output_h5ad', type=str, default='holdout_ground_truth.h5ad',
                      help='Output .h5ad filename')
    parser.add_argument('--output_csv', type=str, default='holdout_counts.csv',
                      help='Output counts CSV filename')
    parser.add_argument('--pert_col', type=str, default='target_gene',
                      help='Column name for perturbations')
    parser.add_argument('--cell_type_col', type=str, default='cell_type',
                      help='Column name for cell types')
    parser.add_argument('--control_pert', type=str, default='non-targeting',
                      help='Control perturbation identifier')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    try:
        # Step 1: Load and parse TOML configuration
        logger.info("=" * 60)
        logger.info("STEP 1: Loading TOML configuration")
        config = load_toml_config(args.toml_config)
        
        # Step 2: Extract holdout perturbations for the requested split
        logger.info("=" * 60)
        logger.info(f"STEP 2: Extracting {args.split} holdout perturbations")
        holdouts = extract_holdout_perturbations(config, args.split)
        
        if not holdouts:
            logger.error(f"No holdout perturbations found for split '{args.split}'")
            return
        
        total_requested = sum(len(cell_perts) for dataset_perts in holdouts.values() 
                            for cell_perts in dataset_perts.values())
        logger.info(f"Total perturbations requested across all datasets: {total_requested}")
        
        # Step 3: Find dataset files
        logger.info("=" * 60)
        logger.info("STEP 3: Finding dataset files")
        dataset_files = find_dataset_files(config, holdouts)
        
        if not dataset_files:
            logger.error("No valid dataset files found")
            return
        
        # Step 4: Process each file
        logger.info("=" * 60)
        logger.info("STEP 4: Processing dataset files")
        dataset_parts = []
        total_found_perts = set()
        
        required_columns = [args.pert_col, args.cell_type_col]
        
        for dataset_name, file_paths in dataset_files.items():
            logger.info(f"\nProcessing dataset: {dataset_name}")
            
            for file_path in file_paths:
                logger.info(f"  Loading file: {os.path.basename(file_path)}")
                
                # Load and validate file
                adata = load_and_validate_file(file_path, required_columns)
                if adata is None:
                    continue
                
                # Determine which cell type this file represents
                file_stem = Path(file_path).stem.lower()
                matched_cell_type = None
                for cell_type in holdouts[dataset_name].keys():
                    if cell_type.lower() in file_stem:
                        matched_cell_type = cell_type
                        break
                
                if matched_cell_type is None:
                    logger.warning(f"Could not determine cell type for {file_path}, skipping")
                    continue
                
                logger.info(f"  Matched to cell type: {matched_cell_type}")
                
                # Get holdout perturbations for this cell type
                holdout_perts = holdouts[dataset_name][matched_cell_type]
                
                # Filter to holdouts + controls
                filtered = filter_to_holdouts(adata, holdout_perts, args.pert_col, args.control_pert)
                if filtered is None:
                    continue
                
                # Add metadata for tracking
                filtered.obs['__dataset__'] = dataset_name
                filtered.obs['__cell_type__'] = matched_cell_type
                filtered.obs['__source_file__'] = os.path.basename(file_path)
                
                # Track which perturbations we actually found
                found_in_file = set(filtered.obs[args.pert_col].unique())
                found_holdouts = found_in_file.intersection(holdout_perts)
                total_found_perts.update(found_holdouts)
                
                dataset_parts.append(filtered)
                logger.info(f"  Added {filtered.shape[0]} cells to combined dataset")
        
        if not dataset_parts:
            logger.error("No valid data found after processing all files")
            return
        
        # Step 5: Combine all datasets
        logger.info("=" * 60)
        logger.info("STEP 5: Combining datasets")
        combined_adata = combine_datasets(dataset_parts)
        
        # Step 6: Generate summary and save results
        logger.info("=" * 60)
        logger.info("STEP 6: Generating summary and saving results")
        
        # Summary statistics
        logger.info("FINAL SUMMARY:")
        logger.info(f"  Total perturbations requested: {total_requested}")
        logger.info(f"  Total perturbations found: {len(total_found_perts)}")
        logger.info(f"  Coverage: {len(total_found_perts)/total_requested*100:.1f}%")
        logger.info(f"  Final dataset shape: {combined_adata.shape}")
        
        # Generate counts CSV
        holdout_counts = generate_summary_stats(combined_adata, args.pert_col, args.control_pert)
        csv_path = os.path.join(args.output_dir, args.output_csv)
        holdout_counts.to_csv(csv_path, index=False)
        logger.info(f"Saved perturbation counts to: {csv_path}")
        
        # Save combined dataset
        h5ad_path = os.path.join(args.output_dir, args.output_h5ad)
        combined_adata.write_h5ad(h5ad_path, compression='gzip')
        logger.info(f"Saved combined dataset to: {h5ad_path}")
        
        logger.info("=" * 60)
        logger.info("SUCCESS: Holdout dataset creation completed!")
        
    except Exception as e:
        logger.error(f"FAILED: {e}")
        raise


if __name__ == "__main__":
    main()