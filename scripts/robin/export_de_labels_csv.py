"""Export adata.uns['de_labels'] to CSV and verify equality.

Writes a CSV with columns: cell_type_target_gene,de_embedding
where de_embedding is a JSON array string (e.g. [0,1,0,...]).

Usage:
    python export_de_labels_csv.py --input /path/to/file.h5ad --output /path/to/out.csv

If --verify is passed (default true), the script will read the CSV back and compare
the parsed arrays to the values in adata.uns['de_labels'] and report whether they match.
"""
from pathlib import Path
import argparse
import json
import sys

import anndata as ad
import pandas as pd


def export_de_labels(adata: ad.AnnData) -> pd.DataFrame:
    if 'de_labels' not in adata.uns:
        raise KeyError("adata.uns['de_labels'] not found; cannot export")

    de_dict = adata.uns['de_labels']
    # Build DataFrame like original script: index -> de_embedding
    df = pd.DataFrame(de_dict).T.reset_index()
    df.columns = ['cell_type_target_gene', 'de_embedding']

    # Serialize de_embedding as JSON strings for robust roundtrip
    df['de_embedding'] = df['de_embedding'].apply(lambda v: json.dumps(list(v) if hasattr(v, '__iter__') and not isinstance(v, (str, bytes)) else v))
    return df


def verify_csv_matches_h5ad(csv_path: Path, adata: ad.AnnData) -> bool:
    df = pd.read_csv(csv_path)
    if 'cell_type_target_gene' not in df.columns or 'de_embedding' not in df.columns:
        print("CSV missing expected columns")
        return False

    # Load dict from adata for comparison
    de_dict = adata.uns.get('de_labels', {})

    ok = True
    for _, row in df.iterrows():
        key = row['cell_type_target_gene']
        try:
            csv_val = json.loads(row['de_embedding'])
        except Exception:
            # maybe plain string like "0" or similar
            csv_val = row['de_embedding']

        h5_val = de_dict.get(key)
        # normalize h5_val to list for comparison
        if hasattr(h5_val, '__iter__') and not isinstance(h5_val, (str, bytes)):
            h5_list = list(h5_val)
        else:
            h5_list = h5_val

        if csv_val != h5_list:
            print(f"DIFFER for {key}: csv={type(csv_val).__name__} len={len(csv_val) if hasattr(csv_val,'__len__') else 'NA'} vs h5ad={type(h5_list).__name__} len={len(h5_list) if hasattr(h5_list,'__len__') else 'NA'}")
            ok = False

    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', '-i', required=True, help='Path to .h5ad input file')
    p.add_argument('--output', '-o', required=True, help='Path to output CSV file')
    p.add_argument('--no-verify', dest='verify', action='store_false', help='Skip verification step')

    args = p.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    if not inp.exists():
        print(f"Input not found: {inp}")
        sys.exit(2)

    adata = ad.read_h5ad(inp)

    try:
        df = export_de_labels(adata)
    except KeyError as e:
        print(e)
        sys.exit(3)

    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote CSV to {out}")

    if args.verify:
        print("Verifying CSV matches h5ad...")
        ok = verify_csv_matches_h5ad(out, adata)
        if ok:
            print("VERIFICATION: MATCH")
            sys.exit(0)
        else:
            print("VERIFICATION: MISMATCH")
            sys.exit(4)


if __name__ == '__main__':
    main()
