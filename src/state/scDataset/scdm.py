# scdm.py
from __future__ import annotations
import os
import glob
import tomllib  # py>=3.11 (use `tomli` if on older Python)
import numpy as np
import anndata as ad
from typing import List
import torch
import pytorch_lightning as pl

# scDataset core
from scdataset import scDataset, MultiIndexable, Streaming, BlockShuffling


def _discover_h5ads_from_toml(toml_config_path: str) -> list[str]:
    with open(toml_config_path, "rb") as f:
        cfg = tomllib.load(f)
    # same shape as cell-load's config: a [datasets] table with name -> dir
    ds = cfg.get("datasets", {})
    files = []
    for _, d in ds.items():
        files.extend(sorted(glob.glob(os.path.join(d, "*.h5ad"))))
    if not files:
        raise FileNotFoundError(f"No .h5ad files found from TOML: {toml_config_path}")
    return files


def _stack_from_h5ads(
    paths: list[str],
    *,
    embed_key: str,
    pert_col: str,
    batch_col: str,
    cell_type_key: str,
    barcode: bool = True,
):
    """Load only what we need from multiple per-cell-type H5ADs."""
    embs = []
    perts = []
    batches = []
    cts = []
    barcodes = []

    for p in paths:
        a = ad.read_h5ad(
            p, backed=None
        )  # small files; if huge, consider backed='r' and np.array(...)
        if embed_key not in a.obsm:
            raise KeyError(f"{p} missing obsm['{embed_key}']")
        E = np.asarray(a.obsm[embed_key])  # (n, d)
        embs.append(E)

        ob = a.obs
        for col in (pert_col, batch_col, cell_type_key):
            if col not in ob.columns:
                raise KeyError(f"{p} missing obs['{col}']")
        perts.append(ob[pert_col].astype(str).to_numpy())
        batches.append(ob[batch_col].astype(str).to_numpy())
        cts.append(ob[cell_type_key].astype(str).to_numpy())

        if barcode and ("barcode" in ob.columns):
            barcodes.append(ob["barcode"].astype(str).to_numpy())

    emb = np.vstack(embs)
    pert = np.concatenate(perts)
    bt = np.concatenate(batches)
    ct = np.concatenate(cts)
    bc = np.concatenate(barcodes) if barcodes else None
    return emb, pert, bt, ct, bc


# ---------- Utilities ----------
def one_hot(names: np.ndarray, vocab: List[str]) -> np.ndarray:
    idx = {g: i for i, g in enumerate(vocab)}
    out = np.zeros((len(names), len(vocab)), dtype=np.float32)
    for r, g in enumerate(map(str, names)):
        j = idx.get(g)
        if j is not None:
            out[r, j] = 1.0
    return out


# ---------- scDataset wrapper that emits STATE-compatible batches ----------
class SCDatasetPertCtrl:
    def __init__(
        self,
        *,
        emb: np.ndarray,  # (N, D)
        pert: np.ndarray,  # (N,)
        batch: np.ndarray,  # (N,)
        cell_type: np.ndarray,  # (N,)
        barcode_arr: np.ndarray | None,
        control_pert: str,
        gene_vocab: list[str] | None = None,
        batch_size: int = 1024,
        fetch_factor: int = 4,
        strategy: str = "block",
        block_size: int = 2048,
        shuffle: bool = True,
        indices: np.ndarray | None = None,
    ):
        self.emb = emb
        self.pert = pert
        self.batch = batch
        self.cell_type = cell_type
        self.barcode = barcode_arr

        N = emb.shape[0]
        if indices is None:
            indices = np.arange(N, dtype=np.int64)
        self.indices = indices

        # Build multi-indexable view from arrays
        self.multi = MultiIndexable(
            __dummy__=np.zeros((len(indices), 1), dtype=np.float32),
            pert=self.pert[indices],
            bt=self.batch[indices],
            ct=self.cell_type[indices],
            bc=self.barcode[indices] if self.barcode is not None else None,
        )

        # pick controls by batch from arrays
        self._picker = self._mk_picker(control_pert)

        self.gene_vocab = gene_vocab or sorted(set(map(str, self.pert)))
        strat = (
            BlockShuffling(indices=self.indices, block_size=block_size, shuffle=shuffle)
            if strategy == "block"
            else Streaming(indices=self.indices, shuffle=shuffle)
        )
        self.dataset = scDataset(
            self.multi,
            strat,
            batch_size=batch_size,
            fetch_factor=fetch_factor,
            fetch_callback=self._fetch_cb,
            batch_transform=self._batch_tf,
        )

    def _mk_picker(self, control_pert: str):
        ctrl_global = np.where(self.pert == control_pert)[0]
        ctrl_by_batch = {}
        for b in np.unique(self.batch):
            mask = (self.batch == b) & (self.pert == control_pert)
            ctrl_by_batch[str(b)] = np.where(mask)[0]
        rng = np.random.default_rng()
        return (ctrl_by_batch, ctrl_global, rng)

    def _fetch_cb(self, collection, idx_local):
        orig = self.indices[np.asarray(idx_local, dtype=np.int64)]
        ctrl_by_batch, ctrl_global, rng = self._mk_picker_state
        ctrl_idx = []
        for i in orig:
            b = str(self.batch[i])
            pool = ctrl_by_batch.get(b)
            if pool is None or len(pool) == 0:
                pool = ctrl_global
            if len(pool) == 0:
                raise RuntimeError("No control cells available; check control_pert.")
            ctrl_idx.append(rng.choice(pool))
        self._last_orig = orig  # keep for batch_tf
        return {"pert_idx": orig, "ctrl_idx": np.asarray(ctrl_idx, dtype=np.int64)}

    @property
    def _mk_picker_state(self):
        # (cached view; built from arrays)
        return self._picker

    def _batch_tf(self, fetched):
        pidx = fetched["pert_idx"]
        cidx = fetched["ctrl_idx"]
        pert_names = self.pert[pidx]
        cell_type = self.cell_type[pidx]
        batch = self.batch[pidx]
        barcode = self.barcode[pidx] if self.barcode is not None else None

        pert_cell_emb = torch.as_tensor(self.emb[pidx], dtype=torch.float32)
        ctrl_cell_emb = torch.as_tensor(self.emb[cidx], dtype=torch.float32)
        pert_oh = torch.as_tensor(one_hot(pert_names, self.gene_vocab))
        out = {
            "pert_cell_emb": pert_cell_emb,
            "ctrl_cell_emb": ctrl_cell_emb,
            "pert_emb": pert_oh,
            "pert_name": list(map(str, pert_names)),
            "cell_type": list(map(str, cell_type)),
            "batch": list(map(str, batch)),
        }
        if barcode is not None:
            out["pert_cell_barcode"] = list(map(str, barcode))
        return out


# ---- DataModule that accepts TOML (cell-load style) ----
class SCDatasetPerturbationModule(pl.LightningDataModule):
    def __init__(
        self,
        *,
        toml_config_path: str,  # <— same UX as cell-load
        embed_key: str,
        pert_col: str,
        batch_col: str,
        cell_type_key: str,
        control_pert: str = "non-targeting",
        gene_vocab: list[str] | None = None,
        # loader/stream params:
        batch_size: int = 1024,
        fetch_factor: int = 4,
        strategy: str = "block",
        block_size: int = 2048,
        shuffle_train: bool = True,
        shuffle_val: bool = False,
        num_workers: int = 8,
        prefetch_factor: int = 2,
        # optional explicit splits (global row indices after stacking)
        train_indices: np.ndarray | None = None,
        val_indices: np.ndarray | None = None,
    ):
        super().__init__()
        self.toml_config_path = toml_config_path
        self.embed_key = embed_key
        self.pert_col = pert_col
        self.batch_col = batch_col
        self.cell_type_key = cell_type_key
        self.control_pert = control_pert
        self.gene_vocab = gene_vocab

        self.batch_size = batch_size
        self.fetch_factor = fetch_factor
        self.strategy = strategy
        self.block_size = block_size
        self.shuffle_train = shuffle_train
        self.shuffle_val = shuffle_val
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor

        self.train_indices = train_indices
        self.val_indices = val_indices

        # filled in setup():
        self._arrays = None
        self._train = None
        self._val = None

    def setup(self, stage: str | None = None):
        files = _discover_h5ads_from_toml(self.toml_config_path)
        emb, pert, bt, ct, bc = _stack_from_h5ads(
            files,
            embed_key=self.embed_key,
            pert_col=self.pert_col,
            batch_col=self.batch_col,
            cell_type_key=self.cell_type_key,
            barcode=True,
        )
        N = emb.shape[0]
        if self.train_indices is None or self.val_indices is None:
            idx = np.arange(N, dtype=np.int64)
            rng = np.random.default_rng(42)
            rng.shuffle(idx)
            k = int(0.9 * N)
            self.train_indices = idx[:k]
            self.val_indices = idx[k:]

        self._train = SCDatasetPertCtrl(
            emb=emb,
            pert=pert,
            batch=bt,
            cell_type=ct,
            barcode_arr=bc,
            control_pert=self.control_pert,
            gene_vocab=self.gene_vocab,
            batch_size=self.batch_size,
            fetch_factor=self.fetch_factor,
            strategy=self.strategy,
            block_size=self.block_size,
            shuffle=True,
            indices=self.train_indices,
        )
        self._val = SCDatasetPertCtrl(
            emb=emb,
            pert=pert,
            batch=bt,
            cell_type=ct,
            barcode_arr=bc,
            control_pert=self.control_pert,
            gene_vocab=self.gene_vocab or self._train.gene_vocab,
            batch_size=self.batch_size,
            fetch_factor=self.fetch_factor,
            strategy="stream",
            block_size=self.block_size,
            shuffle=False,
            indices=self.val_indices,
        )

    def train_dataloader(self):
        return self._train.dataloader(self.num_workers, self.prefetch_factor)

    def val_dataloader(self):
        return self._val.dataloader(self.num_workers, self.prefetch_factor)
