# src/smoothing.py
"""
smoothing.py

ArXiv release pipeline — cache concept-neighbor structures used by:
- KNN-TextEmb baseline
- inference-time concept smoothing ("+SmoothTextEmb")

This script MUST align embeddings to the global concept_id order produced by:
  outputs/<out_subdir>/_global/concept_ids.npy

It then computes cosine kNN neighbors and L1-normalized nonnegative weights
(verbatim Colab semantics: clip negative cosine sims to 0; uniform fallback).

Config-driven:
  python -m src.smoothing --config config/table1.yaml

Outputs (written to outputs/<out_subdir>/_global/):
  emb_knn_idx_K{K}.npy
  emb_knn_w_K{K}.npy
  smooth_idx_K{K}.npy
  smooth_w_K{K}.npy
  neighbors_meta.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from src.utils import cfg_get, load_yaml_config
import src.prepare_cutoff as pc


# =============================================================================
# Embedding alignment helpers (match Colab)
# =============================================================================
_E_all: np.ndarray | None = None
_vocab: pd.DataFrame | None = None
_label_to_row: Dict[int, int] | None = None


def load_embedding_sources(
    *,
    emb_path: Path,
    vocab_path: Path,
) -> Tuple[np.ndarray, pd.DataFrame, Dict[int, int]]:
    """
    Load concept embeddings + vocab and build label->row mapping.

    Colab semantics:
      _vocab = pd.read_csv(CONCEPT_VOCAB_PATH)
      _E_all = np.load(CONCEPT_EMB_PATH)["embeddings"].astype(np.float32)
      assert _E_all.shape[0] == len(_vocab)
      _label_to_row = {int(lbl): int(i) for i, lbl in enumerate(_vocab["label"].values)}
    """
    if not vocab_path.exists():
        raise FileNotFoundError(f"Missing {vocab_path}")
    if not emb_path.exists():
        raise FileNotFoundError(f"Missing {emb_path}")

    vocab = pd.read_csv(vocab_path)
    if "label" not in vocab.columns:
        raise ValueError(f"Concept vocab missing 'label' column. Got cols={vocab.columns.tolist()}")

    emb_npz = np.load(emb_path)
    if "embeddings" not in emb_npz.files:
        raise KeyError(
            f"NPZ missing key 'embeddings'. Keys={emb_npz.files}. "
            f"Expected concepts_embeddings.npz['embeddings']."
        )
    E_all = emb_npz["embeddings"].astype(np.float32)

    if E_all.shape[0] != len(vocab):
        raise ValueError(
            f"Embeddings rows must match vocabulary rows. "
            f"E_all.shape[0]={E_all.shape[0]} vs len(vocab)={len(vocab)}"
        )

    label_to_row = {int(lbl): int(i) for i, lbl in enumerate(vocab["label"].values)}
    print("Loaded embeddings:", E_all.shape, "| vocab rows:", len(vocab))
    return E_all, vocab, label_to_row


def concept_embeddings_for_current_universe(concept_ids: np.ndarray, *, emb_path: Path, vocab_path: Path) -> np.ndarray:
    """
    Return embedding matrix aligned to the given concept_id order.

    Matches Colab:
    - output shape: (len(concept_ids), d)
    - missing labels => row of zeros (do not drop or reindex)
    """
    global _E_all, _vocab, _label_to_row
    if _E_all is None or _vocab is None or _label_to_row is None:
        _E_all, _vocab, _label_to_row = load_embedding_sources(emb_path=emb_path, vocab_path=vocab_path)

    assert _E_all is not None
    assert _label_to_row is not None

    concept_ids = concept_ids.astype(int)
    d = int(_E_all.shape[1])
    out = np.zeros((len(concept_ids), d), dtype=np.float32)

    missing = 0
    for i, cid in enumerate(concept_ids.tolist()):
        r = _label_to_row.get(int(cid))
        if r is None:
            missing += 1
            continue
        out[i] = _E_all[r]

    if missing > 0:
        print(f"[WARN] Missing embeddings for {missing}/{len(concept_ids)} concepts; filled with zeros.")
    return out


# =============================================================================
# Neighbor computation (match Colab semantics)
# =============================================================================
def _row_l2_normalize(E: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(E, axis=1, keepdims=True).astype(np.float32)
    return (E / (norms + eps)).astype(np.float32)


def _l1_norm_rows_nonneg(sim: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Colab semantics:
      sim = maximum(sim, 0)
      w = sim / (sum(sim) + eps)
      if a row has sum==0 (all nonpositive), make it uniform.
    """
    sim = np.maximum(sim.astype(np.float32), np.float32(0.0))
    row_sum = sim.sum(axis=1, keepdims=True)
    w = sim / (row_sum + np.float32(eps))

    zero_rows = (row_sum.squeeze(1) <= np.float32(eps))
    if np.any(zero_rows):
        k = sim.shape[1]
        w[zero_rows] = np.float32(1.0 / k)
    return w.astype(np.float32)


def compute_cosine_knn_tables(E_aligned: np.ndarray, *, max_k: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute cosine kNN for all rows in E_aligned.

    Returns:
      idxs: (n, max_k) int32
      sims: (n, max_k) float32   where sims = 1 - cosine_distance, self removed.
    """
    if max_k <= 0:
        raise ValueError("max_k must be positive")

    E_norm = _row_l2_normalize(E_aligned)
    n = int(E_norm.shape[0])
    n_neighbors = min(max_k + 1, n)  # +1 because we drop self

    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(E_norm)
    dists, idxs = nn.kneighbors(E_norm)

    # drop self neighbor
    if idxs.shape[1] > 1:
        dists = dists[:, 1:]
        idxs = idxs[:, 1:]
    else:
        dists = np.zeros((n, 0), dtype=np.float32)
        idxs = np.zeros((n, 0), dtype=np.int32)

    sims = (np.float32(1.0) - dists.astype(np.float32)).astype(np.float32)
    idxs = idxs.astype(np.int32)

    # pad if needed (rare; small n)
    if idxs.shape[1] < max_k:
        pad = max_k - idxs.shape[1]
        if idxs.shape[1] == 0:
            idxs_pad = np.zeros((n, pad), dtype=np.int32)
            sims_pad = np.zeros((n, pad), dtype=np.float32)
        else:
            idxs_pad = np.repeat(idxs[:, -1:], pad, axis=1)
            sims_pad = np.repeat(sims[:, -1:], pad, axis=1)
        idxs = np.concatenate([idxs, idxs_pad], axis=1)
        sims = np.concatenate([sims, sims_pad], axis=1)

    return idxs[:, :max_k], sims[:, :max_k]


def save_neighbor_tables(
    *,
    out_dir: Path,
    emb_idxs: np.ndarray,
    emb_sims: np.ndarray,
    knn_ks: Iterable[int],
    smooth_k: int,
    meta: dict,
    overwrite: bool,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    def _write(name: str, arr: np.ndarray) -> None:
        p = out_dir / name
        if p.exists() and not overwrite:
            print("[skip]", p.name)
            return
        np.save(p, arr)
        print("[write]", p.name, arr.shape, arr.dtype)

    max_k = int(emb_idxs.shape[1])

    # KNN-TextEmb tables for each K
    for K in sorted(set(int(x) for x in knn_ks)):
        if K <= 0 or K > max_k:
            raise ValueError(f"KNN K must be in [1, {max_k}]. Got K={K}")
        idx = emb_idxs[:, :K].astype(np.int32)
        w = _l1_norm_rows_nonneg(emb_sims[:, :K]).astype(np.float32)
        _write(f"emb_knn_idx_K{K}.npy", idx)
        _write(f"emb_knn_w_K{K}.npy", w)

    # Smoothing table (one K)
    K = int(smooth_k)
    if K <= 0 or K > max_k:
        raise ValueError(f"smooth_k must be in [1, {max_k}]. Got smooth_k={K}")
    sidx = emb_idxs[:, :K].astype(np.int32)
    sw = _l1_norm_rows_nonneg(emb_sims[:, :K]).astype(np.float32)
    _write(f"smooth_idx_K{K}.npy", sidx)
    _write(f"smooth_w_K{K}.npy", sw)

    meta_path = out_dir / "neighbors_meta.json"
    if meta_path.exists() and not overwrite:
        print("[skip]", meta_path.name)
    else:
        meta_path.write_text(json.dumps(meta, indent=2))
        print("[write]", meta_path.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Cache embedding kNN neighbors + smoothing tables (Colab semantics).")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    parser.add_argument("--knn-ks", type=int, nargs="*", default=None, help="Override K values for KNN-TextEmb tables.")
    parser.add_argument("--smooth-k", type=int, default=None, help="Override K for smoothing neighbor table.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing neighbor tables.")
    args = parser.parse_args()

    cfg, cfg_dir = load_yaml_config(args.config)

    # Apply config to prepare_cutoff module so DATA_DIR/OUT_DIR/paths align everywhere.
    pc.apply_config(cfg, cfg_dir=cfg_dir)

    out_subdir = str(cfg_get(cfg, "output.out_subdir", "table1"))
    global_dir = pc.OUT_DIR / out_subdir / "_global"
    global_dir.mkdir(parents=True, exist_ok=True)

    # Concept universe ordering must come from global artifacts
    concept_ids_path = global_dir / "concept_ids.npy"
    if not concept_ids_path.exists():
        raise FileNotFoundError(
            f"Missing global concept_ids at {concept_ids_path}. "
            f"Run src.prepare_cutoff first to build global artifacts."
        )
    concept_ids = np.load(concept_ids_path).astype(int)
    print("Loaded concept_ids:", concept_ids.shape, "| unique:", len(np.unique(concept_ids)))

    # Input paths from config
    emb_path = pc.DATA_DIR / str(cfg_get(cfg, "paths.concept_embeddings", "concepts_embeddings.npz"))
    vocab_path = pc.DATA_DIR / str(cfg_get(cfg, "paths.concept_vocab", "concepts_vocabulary.csv"))

    # Neighbor params from config (unless CLI override)
    cfg_knn_ks = cfg_get(cfg, "knn.ks", [20, 50, 100])
    knn_ks = [int(x) for x in (args.knn_ks if args.knn_ks is not None else cfg_knn_ks)]
    cfg_smooth_k = int(cfg_get(cfg, "smoothing.k", 50))
    smooth_k = int(args.smooth_k if args.smooth_k is not None else cfg_smooth_k)

    max_k = max([smooth_k] + knn_ks) if knn_ks else smooth_k

    # Align embeddings to universe order
    E_aligned = concept_embeddings_for_current_universe(concept_ids, emb_path=emb_path, vocab_path=vocab_path)

    # Compute cosine neighbors up to max_k
    emb_idxs, emb_sims = compute_cosine_knn_tables(E_aligned, max_k=max_k)

    meta = {
        "data_dir": str(pc.DATA_DIR),
        "out_dir": str(pc.OUT_DIR),
        "concept_emb_path": str(emb_path),
        "concept_vocab_path": str(vocab_path),
        "concept_ids_path": str(concept_ids_path),
        "n_concepts": int(len(concept_ids)),
        "emb_dim": int(E_aligned.shape[1]),
        "knn_ks": knn_ks,
        "smooth_k": int(smooth_k),
        "metric": "cosine",
        "weighting": "l1_norm_rows_nonneg(sim=clip(1-cos_dist,0), uniform_fallback)",
    }

    save_neighbor_tables(
        out_dir=global_dir,
        emb_idxs=emb_idxs,
        emb_sims=emb_sims,
        knn_ks=knn_ks,
        smooth_k=smooth_k,
        meta=meta,
        overwrite=bool(args.overwrite),
    )


if __name__ == "__main__":
    main()
