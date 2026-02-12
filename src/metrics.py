from __future__ import annotations
# src/metrics.py

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Set, Tuple

import numpy as np


@dataclass(frozen=True)
class EvalResult:
    mrr: float
    recall_10: float
    recall_100: float
    ndcg_100: float


def _eval_from_full_scores(scores, seen_idx, pos_set, K1=10, K2=100):
    """
    Colab tie-safe full-rank evaluator (verbatim semantics).

    scores: np.ndarray shape (n_objects,)
    seen_idx: np.ndarray of object indices to mask out (already-seen under target<=T in Mode B)
    pos_set: python set of positive object indices

    Returns dict with keys:
      'mrr', 'recall@10', 'recall@100', f'ndcg@{K2}'
    """
    scores = np.asarray(scores, dtype=np.float64)

    # mask already-seen candidates
    if seen_idx is not None and len(seen_idx) > 0:
        scores = scores.copy()
        scores[np.asarray(seen_idx, dtype=np.int32)] = -np.inf

    # deterministic tie-break: prefer smaller object_idx when scores are equal
    finite = np.isfinite(scores)
    if finite.any():
        scale = float(np.max(np.abs(scores[finite])))
    else:
        scale = 1.0
    eps = np.finfo(np.float64).eps * (scale + 1.0)
    scores = scores - eps * np.arange(scores.size, dtype=np.float64)

    n_pos = len(pos_set)
    if n_pos == 0:
        return {"mrr": 0.0, "recall@10": 0.0, "recall@100": 0.0, f"ndcg@{int(K2)}": 0.0}

    # ----- FULL MRR (exact rank of best positive among ALL candidates) -----
    pos_idx = np.fromiter(pos_set, dtype=np.int32, count=n_pos)
    pos_scores = scores[pos_idx]
    finite_pos = np.isfinite(pos_scores)
    if not finite_pos.any():
        return {"mrr": 0.0, "recall@10": 0.0, "recall@100": 0.0, f"ndcg@{int(K2)}": 0.0}

    best_pos_score = pos_scores[finite_pos].max()
    # rank = 1 + number of items with strictly higher score (ties broken by eps)
    rank = 1 + int(np.sum(scores > best_pos_score))
    mrr = 1.0 / rank

    # ----- Exact top-K for Recall/NDCG -----
    K2 = int(K2)
    K1 = int(K1)
    if K2 <= 0:
        return {"mrr": float(mrr), "recall@10": 0.0, "recall@100": 0.0, f"ndcg@{int(K2)}": 0.0}

    # If K2 > n_objects, clip to size to avoid argpartition error
    K2_eff = min(K2, scores.size)
    topk_unsorted = np.argpartition(scores, -K2_eff)[-K2_eff:]
    topk_sorted = topk_unsorted[np.argsort(scores[topk_unsorted])[::-1]]

    hit = np.fromiter((i in pos_set for i in topk_sorted), dtype=np.bool_, count=len(topk_sorted))

    r10 = hit[: min(K1, len(hit))].sum() / n_pos
    r100 = hit[: min(K2_eff, 100)].sum() / n_pos

    denom = np.log2(np.arange(2, K2_eff + 2))
    dcg = (hit.astype(np.float32) / denom).sum()
    m = min(n_pos, K2_eff)
    idcg = (np.ones(m, dtype=np.float32) / denom[:m]).sum()
    ndcg = float(dcg / idcg) if idcg > 0 else 0.0

    return {
        "mrr": float(mrr),
        "recall@10": float(r10),
        "recall@100": float(r100),
        f"ndcg@{int(K2)}": float(ndcg),
    }



def evaluate_method_over_concepts(
    *,
    concept_idxs: Sequence[int],
    get_scores_row,  # callable: concept_idx -> np.ndarray (n_objects,)
    train_seen: dict[int, Set[int]],
    test_pos: dict[int, Set[int]],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a method over a list of concept indices.

    Returns arrays (mrr, r10, r100, ndcg100) aligned to concept_idxs order.
    """
    mrrs = np.zeros(len(concept_idxs), dtype=np.float32)
    r10s = np.zeros(len(concept_idxs), dtype=np.float32)
    r100s = np.zeros(len(concept_idxs), dtype=np.float32)
    ndcgs = np.zeros(len(concept_idxs), dtype=np.float32)

    for i, cidx in enumerate(concept_idxs):
        seen_set = train_seen.get(int(cidx), set())
        pos_set  = test_pos.get(int(cidx), set())
        scores_row = get_scores_row(int(cidx))

        seen_idx = np.fromiter(seen_set, dtype=np.int32) if len(seen_set) else np.array([], dtype=np.int32)

        met = _eval_from_full_scores(scores_row, seen_idx, pos_set, K1=10, K2=100)

        mrrs[i]  = met["mrr"]
        r10s[i]  = met["recall@10"]
        r100s[i] = met["recall@100"]
        ndcgs[i] = met["ndcg@100"]    


    return mrrs, r10s, r100s, ndcgs
