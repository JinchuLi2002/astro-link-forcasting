# src/train_eval.py
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.utils import cfg_get, load_yaml_config
import src.prepare_cutoff as pc
from src.metrics import _eval_from_full_scores


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(os.environ.get("ASTRO_OUT_DIR", str(REPO_ROOT / "outputs")))


def build_mask_csr(mask_pairs: pd.DataFrame, *, G: dict) -> sp.csr_matrix:
    concept_id_to_idx = G["concept_id_to_idx"]
    object_id_to_idx = G["object_id_to_idx"]
    n_concepts = G["n_concepts"]
    n_objects = G["n_objects"]

    df = mask_pairs[["label", "object_id"]].drop_duplicates().copy()
    r = df["label"].map(lambda x: concept_id_to_idx.get(int(x), None))
    c = df["object_id"].map(lambda x: object_id_to_idx.get(str(x), None))
    if r.isna().any() or c.isna().any():
        raise ValueError(f"Mask mapping failed: missing rows={int(r.isna().sum())}, cols={int(c.isna().sum())}")

    X = sp.coo_matrix(
        (np.ones(len(df), dtype=np.float32), (r.to_numpy(np.int32), c.to_numpy(np.int32))),
        shape=(n_concepts, n_objects),
    ).tocsr()
    return X


def _score_from_weighted_rows(X_csr: sp.csr_matrix, row_idx: np.ndarray, row_w: np.ndarray, n_objects: int) -> np.ndarray:
    if row_idx is None or len(row_idx) == 0:
        return np.zeros(n_objects, dtype=np.float32)
    row_idx = np.asarray(row_idx, dtype=np.int32)
    row_w = np.asarray(row_w, dtype=np.float32)
    if row_idx.size != row_w.size:
        raise ValueError("row_idx and row_w size mismatch")
    s = X_csr[row_idx].T.dot(row_w)
    return np.asarray(s).ravel().astype(np.float32, copy=False)


class ConceptKNNAA:
    def __init__(
        self,
        X_train: sp.csr_matrix,
        K: int = 50,
        mix_global: float = 0.10,
        global_scores: np.ndarray | None = None,
        use_binary_for_similarity: bool = True,
        seed: int = 0,
    ):
        self.X = X_train.tocsr()
        self.n_concepts, self.n_objects = self.X.shape
        self.K = int(K)
        self.mix_global = float(mix_global)
        self.global_scores = None if global_scores is None else global_scores.astype(np.float32)

        if use_binary_for_similarity:
            Xbin = self.X.copy()
            Xbin.data = np.ones_like(Xbin.data, dtype=np.float32)
            self.Xsim = Xbin.tocsr()
        else:
            self.Xsim = self.X

        deg_obj = np.asarray(self.Xsim.getnnz(axis=0)).ravel().astype(np.float32)
        w_obj = np.zeros_like(deg_obj, dtype=np.float32)
        mask = deg_obj > 0
        w_obj[mask] = 1.0 / np.log1p(deg_obj[mask])
        self.w_obj = w_obj

        self.nbr_idx: List[np.ndarray] = []
        self.nbr_w: List[np.ndarray] = []

        for c_idx in range(self.n_concepts):
            obj_idx = self.Xsim.getrow(c_idx).indices
            if obj_idx.size == 0:
                self.nbr_idx.append(np.array([], dtype=np.int32))
                self.nbr_w.append(np.array([], dtype=np.float32))
                continue

            sim_vec = self.Xsim[:, obj_idx].dot(self.w_obj[obj_idx])
            sim_vec = np.asarray(sim_vec).ravel().astype(np.float32)
            sim_vec[c_idx] = 0.0

            pos = np.where(sim_vec > 0)[0]
            if pos.size == 0:
                self.nbr_idx.append(np.array([], dtype=np.int32))
                self.nbr_w.append(np.array([], dtype=np.float32))
                continue

            if pos.size > self.K:
                top = pos[np.argpartition(sim_vec[pos], -self.K)[-self.K:]]
            else:
                top = pos

            top = top[np.argsort(sim_vec[top])[::-1]]
            w = sim_vec[top]
            w = w / (w.sum() + 1e-12)

            self.nbr_idx.append(top.astype(np.int32))
            self.nbr_w.append(w.astype(np.float32))


def _load_knn_tables(global_dir: Path, *, KNN_KS: Sequence[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray]]:
    emb_idx: Dict[int, np.ndarray] = {}
    emb_w: Dict[int, np.ndarray] = {}
    for K in KNN_KS:
        K = int(K)
        p_idx = global_dir / f"emb_knn_idx_K{K}.npy"
        p_w = global_dir / f"emb_knn_w_K{K}.npy"
        if not p_idx.exists() or not p_w.exists():
            raise FileNotFoundError(f"Missing {p_idx.name} / {p_w.name} in {global_dir} (run src.smoothing).")
        emb_idx[K] = np.load(p_idx).astype(np.int32, copy=False)
        emb_w[K] = np.load(p_w).astype(np.float32, copy=False)
    return emb_idx, emb_w


def _load_smoothing_tables(global_dir: Path, *, smooth_k: int) -> Tuple[np.ndarray, np.ndarray]:
    p_idx = global_dir / f"smooth_idx_K{int(smooth_k)}.npy"
    p_w = global_dir / f"smooth_w_K{int(smooth_k)}.npy"
    if not p_idx.exists() or not p_w.exists():
        raise FileNotFoundError(f"Missing {p_idx.name} / {p_w.name} in {global_dir} (run src.smoothing).")
    return (
        np.load(p_idx).astype(np.int32, copy=False),
        np.load(p_w).astype(np.float32, copy=False),
    )


def evaluate_popularity_like(
    *,
    obj_scores: np.ndarray,
    mask_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    eligible_concept_ids: np.ndarray,
    concept_id_to_idx: Dict[int, int],
    object_id_to_idx: Dict[str, int],
    ndcg_k: int = 100,
) -> pd.DataFrame:
    from collections import defaultdict

    train_seen = defaultdict(set)
    for r in mask_pairs.itertuples(index=False):
        c = concept_id_to_idx[int(r.label)]
        o = object_id_to_idx[str(r.object_id)]
        train_seen[c].add(o)

    test_pos = defaultdict(set)
    for r in test_pairs.itertuples(index=False):
        c = concept_id_to_idx[int(r.label)]
        o = object_id_to_idx[str(r.object_id)]
        test_pos[c].add(o)

    scores64 = np.asarray(obj_scores, dtype=np.float64)
    rows = []
    for cid in eligible_concept_ids:
        c_idx = concept_id_to_idx[int(cid)]
        seen = train_seen.get(c_idx, set())
        pos = test_pos.get(c_idx, set())
        seen_idx = None
        if seen:
            seen_idx = np.fromiter(seen, dtype=np.int32, count=len(seen))
        met = _eval_from_full_scores(scores=scores64, seen_idx=seen_idx, pos_set=pos, K1=10, K2=int(ndcg_k))
        met["label"] = int(cid)
        rows.append(met)
    return pd.DataFrame(rows)


def stratified_macro_summary(
    per_concept_df: pd.DataFrame,
    eligible_ids: np.ndarray,
    *,
    strata: Dict[str, np.ndarray],
    ndcg_k: int = 100,
    strata_to_report: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    out_rows = []
    eligible_set = set(int(x) for x in eligible_ids)
    names = list(strata.keys()) if not strata_to_report else list(strata_to_report)

    for stratum_name in names:
        idset = strata[stratum_name]
        ids = sorted(list(eligible_set & set(int(x) for x in idset)))
        sub = per_concept_df[per_concept_df["label"].isin(ids)]
        if len(sub) == 0:
            out_rows.append({"stratum": stratum_name, "n_concepts": 0})
            continue
        out_rows.append(
            {
                "stratum": stratum_name,
                "n_concepts": int(len(sub)),
                "mrr": float(sub["mrr"].mean()),
                "recall@10": float(sub["recall@10"].mean()),
                "recall@100": float(sub["recall@100"].mean()),
                f"ndcg@{ndcg_k}": float(sub[f"ndcg@{ndcg_k}"].mean()),
            }
        )
    return pd.DataFrame(out_rows)


def _add_run(
    all_runs: List[pd.DataFrame],
    per_concept_df: pd.DataFrame,
    *,
    method: str,
    cutoff: int,
    seed: int,
    eligible: np.ndarray,
    strata: Dict[str, np.ndarray],
    ndcg_k: int,
    strata_to_report: Optional[Sequence[str]],
) -> None:
    summ = stratified_macro_summary(per_concept_df, eligible, strata=strata, ndcg_k=ndcg_k, strata_to_report=strata_to_report)
    summ["method"] = method
    summ["cutoff"] = int(cutoff)
    summ["seed"] = int(seed)
    all_runs.append(summ)


def build_recent_obj_scores(*, edges_train: pd.DataFrame, T: int, W: int, object_id_to_idx: Dict[str, int], n_objects: int) -> np.ndarray:
    lo = int(T) - int(W) + 1
    recent = edges_train[(edges_train["year"] >= lo) & (edges_train["year"] <= int(T))]
    if len(recent) == 0:
        return np.zeros(n_objects, dtype=np.float32)

    recent_pairs = (
        recent.groupby(["label", "object_id"], as_index=False)["raw_w"]
        .sum()
        .rename(columns={"raw_w": "total_w"})
    )
    recent_pairs["value"] = np.log1p(recent_pairs["total_w"].astype(np.float32))

    r_idx = recent_pairs["object_id"].map(lambda x: object_id_to_idx[str(x)]).to_numpy(dtype=np.int32)
    r_val = recent_pairs["value"].to_numpy(dtype=np.float32)

    out = np.zeros(n_objects, dtype=np.float32)
    np.add.at(out, r_idx, r_val)
    return out


def fit_als(X_train: sp.csr_matrix, *, seed: int, als_cfg: dict, use_gpu_train: bool) -> Any:
    try:
        import implicit  # noqa: F401
    except ImportError as e:
        raise ImportError("implicit is required for ALS training. Install with: pip install implicit") from e

    if use_gpu_train:
        try:
            from implicit.gpu.als import AlternatingLeastSquares as ALSClass  # type: ignore
        except Exception:
            from implicit.als import AlternatingLeastSquares as ALSClass  # type: ignore
            print("WARNING: implicit.gpu.als not available; falling back to CPU ALS class.")
    else:
        from implicit.als import AlternatingLeastSquares as ALSClass  # type: ignore

    alpha = float(als_cfg.get("alpha", 10.0))
    Cui = (X_train * alpha).astype(np.float32).tocsr()

    try:
        model = ALSClass(
            factors=int(als_cfg.get("factors", 128)),
            regularization=float(als_cfg.get("regularization", 0.05)),
            iterations=int(als_cfg.get("iterations", 30)),
            random_state=int(seed),
            use_gpu=bool(use_gpu_train),
        )
    except TypeError:
        model = ALSClass(
            factors=int(als_cfg.get("factors", 128)),
            regularization=float(als_cfg.get("regularization", 0.05)),
            iterations=int(als_cfg.get("iterations", 30)),
            random_state=int(seed),
        )

    model.fit(Cui)
    return model


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "to_numpy"):
        return x.to_numpy()
    if hasattr(x, "get"):
        return x.get()
    return np.asarray(x)


def extract_als_factors(model: Any, *, n_concepts: int, n_objects: int) -> Tuple[np.ndarray, np.ndarray]:
    U = np.asarray(_to_numpy(model.user_factors), dtype=np.float32)
    V = np.asarray(_to_numpy(model.item_factors), dtype=np.float32)

    if U.shape[0] == n_concepts and V.shape[0] == n_objects:
        return U, V
    if U.shape[0] == n_objects and V.shape[0] == n_concepts:
        return V, U
    raise ValueError(f"Unexpected ALS factor shapes: U={U.shape}, V={V.shape} (expected {n_concepts}, {n_objects})")


def run_from_config(cfg: dict, *, cfg_dir: Path) -> pd.DataFrame:
    pc.apply_config(cfg, cfg_dir=cfg_dir)

    global OUT_DIR
    OUT_DIR = pc.OUT_DIR

    out_subdir = str(cfg_get(cfg, "output.out_subdir", "table1"))
    global_dir = OUT_DIR / out_subdir / "_global"
    global_dir.mkdir(parents=True, exist_ok=True)

    cutoffs = [int(x) for x in cfg_get(cfg, "cutoffs", [2017, 2019, 2021, 2023])]
    min_train_pos = int(cfg_get(cfg, "min_train_pos", 10))
    ndcg_k = int(cfg_get(cfg, "metrics.ndcg_k", 100))

    strata_to_report = cfg_get(cfg, "output.strata_to_report", None)
    if strata_to_report is not None:
        strata_to_report = list(strata_to_report)

    # method toggles
    run_random = bool(cfg_get(cfg, "methods.random", True))
    run_popularity = bool(cfg_get(cfg, "methods.popularity", True))
    run_recent = bool(cfg_get(cfg, "methods.recent_popularity", True))
    run_knn_aa = bool(cfg_get(cfg, "methods.knn_aa", True))
    run_knn_textemb = bool(cfg_get(cfg, "methods.knn_textemb", True))
    run_als_flag = bool(cfg_get(cfg, "methods.als", True))
    add_smoothing_variants = bool(cfg_get(cfg, "methods.add_smoothing_variants", True))

    # KNN params
    KNN_KS = [int(x) for x in cfg_get(cfg, "knn.ks", [5, 10, 25, 50, 100])]
    KMAX = int(max(KNN_KS)) if KNN_KS else 0

    # smoothing params
    smooth_enabled = bool(cfg_get(cfg, "smoothing.enabled", True))
    smooth_lam = float(cfg_get(cfg, "smoothing.lam", 0.5))
    smooth_k = int(cfg_get(cfg, "smoothing.k", 100))

    # ALS
    als_cfg = dict(cfg_get(cfg, "als", {}))
    als_seeds = [int(x) for x in cfg_get(cfg, "als.seeds", [0, 1, 2, 3, 4])]
    use_gpu_train = bool(cfg_get(cfg, "als.use_gpu_train", True))

    # prepare_cutoff config
    cfg_train = pc.edge_config_from_dict(cfg_get(cfg, "edge_configs.train", {}), name="train_all")
    cfg_target = pc.edge_config_from_dict(cfg_get(cfg, "edge_configs.target", {}), name="target_all")

    force_rebuild_global = bool(cfg_get(cfg, "runtime.force_rebuild_global", False))
    overwrite_cutoff = bool(cfg_get(cfg, "runtime.overwrite_cutoff", False))

    G = pc.ensure_global_artifacts(cfg_train, cfg_target, out_subdir=f"{out_subdir}/_global", force_rebuild=force_rebuild_global)
    STRATA = G.get("STRATA")
    if STRATA is None:
        raise ValueError("Global artifacts missing STRATA.")

    n_concepts = int(G["n_concepts"])
    n_objects = int(G["n_objects"])

    # neighbor caches
    if run_knn_textemb or (smooth_enabled and add_smoothing_variants):
        EMB_NBR_IDX, EMB_NBR_W = _load_knn_tables(global_dir, KNN_KS=KNN_KS)
    else:
        EMB_NBR_IDX, EMB_NBR_W = {}, {}

    if smooth_enabled and add_smoothing_variants:
        SMOOTH_NBR_IDX, SMOOTH_NBR_W = _load_smoothing_tables(global_dir, smooth_k=smooth_k)
    else:
        SMOOTH_NBR_IDX, SMOOTH_NBR_W = None, None

    # Random scores
    rng = np.random.default_rng(int(cfg_get(cfg, "baselines.random_seed", 0)))
    RAND_OBJ_SCORES = rng.random(n_objects).astype(np.float32)

    all_runs: List[pd.DataFrame] = []

    for T in cfg["cutoffs"]:
        print("\n" + "=" * 90)
        print(f"Cutoff year T={T}")
        print("=" * 90)

        train_pairs, test_pairs, eligible, X_train, train_seen, test_pos, mask_pairs = pc.prepare_cutoff(
            int(T),
            min_train_pos=min_train_pos,
            write=True,
            out_subdir=out_subdir,
            overwrite=overwrite_cutoff,
        )

        X_mask = build_mask_csr(mask_pairs, G=G)
        X_csr = X_train.tocsr()
        eligible_idx = np.array([G["concept_id_to_idx"][int(cid)] for cid in eligible], dtype=np.int32)

        # ---------------- Baselines ----------------
        if run_random:
            rand_pc = evaluate_popularity_like(
                obj_scores=RAND_OBJ_SCORES,
                mask_pairs=mask_pairs,
                test_pairs=test_pairs,
                eligible_concept_ids=eligible,
                concept_id_to_idx=G["concept_id_to_idx"],
                object_id_to_idx=G["object_id_to_idx"],
                ndcg_k=ndcg_k,
            )
            _add_run(all_runs, rand_pc, method="Random", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)
            if smooth_enabled and add_smoothing_variants:
                _add_run(all_runs, rand_pc, method="Random+SmoothTextEmb", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)

        if run_popularity:
            pop_obj_scores = np.asarray(X_train.sum(axis=0)).ravel().astype(np.float32)
            pop_pc = evaluate_popularity_like(
                obj_scores=pop_obj_scores,
                mask_pairs=mask_pairs,
                test_pairs=test_pairs,
                eligible_concept_ids=eligible,
                concept_id_to_idx=G["concept_id_to_idx"],
                object_id_to_idx=G["object_id_to_idx"],
                ndcg_k=ndcg_k,
            )
            _add_run(all_runs, pop_pc, method="Popularity", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)
            if smooth_enabled and add_smoothing_variants:
                _add_run(all_runs, pop_pc, method="Popularity+SmoothTextEmb", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)

        if run_recent:
            for W in [int(x) for x in cfg_get(cfg, "baselines.recent_windows", [3, 5])]:
                recent_obj_scores = build_recent_obj_scores(
                    edges_train=G["edges_train"],
                    T=int(T),
                    W=int(W),
                    object_id_to_idx=G["object_id_to_idx"],
                    n_objects=n_objects,
                )
                rpc = evaluate_popularity_like(
                    obj_scores=recent_obj_scores,
                    mask_pairs=mask_pairs,
                    test_pairs=test_pairs,
                    eligible_concept_ids=eligible,
                    concept_id_to_idx=G["concept_id_to_idx"],
                    object_id_to_idx=G["object_id_to_idx"],
                    ndcg_k=ndcg_k,
                )
                _add_run(all_runs, rpc, method=f"RecentPopularity_W{W}", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)
                if smooth_enabled and add_smoothing_variants:
                    _add_run(all_runs, rpc, method=f"RecentPopularity_W{W}+SmoothTextEmb", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)

        # ---------------- KNN-AA ----------------
        if run_knn_aa and KMAX > 0:
            knn_cfg = dict(cfg_get(cfg, "knn.aa", {}))
            knnAA = ConceptKNNAA(
                X_train=X_csr,
                K=KMAX,
                mix_global=float(knn_cfg.get("mix_global", 0.0)),
                global_scores=None,
                use_binary_for_similarity=bool(knn_cfg.get("use_binary_for_similarity", True)),
                seed=int(knn_cfg.get("seed", 0)),
            )
            AA_NBR_IDX = knnAA.nbr_idx
            AA_NBR_W = knnAA.nbr_w

            def _score_knn_aa(c_idx: int, K: int) -> np.ndarray:
                idx = AA_NBR_IDX[c_idx]
                if idx.size == 0:
                    return np.zeros(n_objects, dtype=np.float32)
                idx = idx[:K]
                w = AA_NBR_W[c_idx][:K].astype(np.float32, copy=False)
                w = w / (w.sum() + 1e-12)
                return _score_from_weighted_rows(X_csr, idx, w, n_objects)

            def _score_knn_aa_smooth(c_idx: int, K: int) -> np.ndarray:
                assert SMOOTH_NBR_IDX is not None and SMOOTH_NBR_W is not None
                weights: Dict[int, float] = {}

                idx0 = AA_NBR_IDX[c_idx]
                if idx0.size > 0:
                    idx0 = idx0[:K]
                    w0 = AA_NBR_W[c_idx][:K].astype(np.float32, copy=False)
                    w0 = w0 / (w0.sum() + 1e-12)
                    base_scale = float(1.0 - smooth_lam)
                    for j, wj in zip(idx0, w0):
                        jj = int(j)
                        weights[jj] = weights.get(jj, 0.0) + base_scale * float(wj)

                nbrs = SMOOTH_NBR_IDX[c_idx]
                w_s = SMOOTH_NBR_W[c_idx]
                for m, wm in zip(nbrs, w_s):
                    m = int(m)
                    idxm = AA_NBR_IDX[m]
                    if idxm.size == 0:
                        continue
                    idxm = idxm[:K]
                    wm_row = AA_NBR_W[m][:K].astype(np.float32, copy=False)
                    wm_row = wm_row / (wm_row.sum() + 1e-12)
                    scale = float(smooth_lam) * float(wm)
                    for j, wj in zip(idxm, wm_row):
                        jj = int(j)
                        weights[jj] = weights.get(jj, 0.0) + scale * float(wj)

                if not weights:
                    return np.zeros(n_objects, dtype=np.float32)

                idx_all = np.fromiter(weights.keys(), dtype=np.int32)
                w_all = np.fromiter(weights.values(), dtype=np.float32)
                w_all = w_all / (w_all.sum() + 1e-12)
                return _score_from_weighted_rows(X_csr, idx_all, w_all, n_objects)

            for K in KNN_KS:
                rows = []
                for cid, c_idx in zip(eligible, eligible_idx):
                    c_idx = int(c_idx)
                    scores = _score_knn_aa(c_idx, int(K))
                    seen_idx = X_mask[c_idx].indices
                    pos = test_pos.get(c_idx, set())
                    met = _eval_from_full_scores(scores, seen_idx, pos, K1=10, K2=ndcg_k)
                    met["label"] = int(cid)
                    rows.append(met)
                knnAA_pc = pd.DataFrame(rows)
                _add_run(all_runs, knnAA_pc, method=f"ConceptKNN_AA_K{int(K)}", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)

                if smooth_enabled and add_smoothing_variants:
                    rows = []
                    for cid, c_idx in zip(eligible, eligible_idx):
                        c_idx = int(c_idx)
                        scores = _score_knn_aa_smooth(c_idx, int(K))
                        seen_idx = X_mask[c_idx].indices
                        pos = test_pos.get(c_idx, set())
                        met = _eval_from_full_scores(scores, seen_idx, pos, K1=10, K2=ndcg_k)
                        met["label"] = int(cid)
                        rows.append(met)
                    knnAA_s = pd.DataFrame(rows)
                    _add_run(all_runs, knnAA_s, method=f"ConceptKNN_AA_K{int(K)}+SmoothTextEmb", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)

        # ---------------- KNN-TextEmb ----------------
        if run_knn_textemb and KMAX > 0:
            def _score_knn_emb(c_idx: int, K: int) -> np.ndarray:
                idx = EMB_NBR_IDX[int(K)][c_idx]
                w = EMB_NBR_W[int(K)][c_idx]
                return _score_from_weighted_rows(X_csr, idx, w, n_objects)

            def _score_knn_emb_smooth(c_idx: int, K: int) -> np.ndarray:
                assert SMOOTH_NBR_IDX is not None and SMOOTH_NBR_W is not None
                weights: Dict[int, float] = {}
                idx0 = EMB_NBR_IDX[int(K)][c_idx]
                w0 = EMB_NBR_W[int(K)][c_idx]
                base_scale = float(1.0 - smooth_lam)
                for j, wj in zip(idx0, w0):
                    jj = int(j)
                    weights[jj] = weights.get(jj, 0.0) + base_scale * float(wj)

                nbrs = SMOOTH_NBR_IDX[c_idx]
                w_s = SMOOTH_NBR_W[c_idx]
                for m, wm in zip(nbrs, w_s):
                    m = int(m)
                    idxm = EMB_NBR_IDX[int(K)][m]
                    wm_row = EMB_NBR_W[int(K)][m]
                    scale = float(smooth_lam) * float(wm)
                    for j, wj in zip(idxm, wm_row):
                        jj = int(j)
                        weights[jj] = weights.get(jj, 0.0) + scale * float(wj)

                idx_all = np.fromiter(weights.keys(), dtype=np.int32)
                w_all = np.fromiter(weights.values(), dtype=np.float32)
                w_all = w_all / (w_all.sum() + 1e-12)
                return _score_from_weighted_rows(X_csr, idx_all, w_all, n_objects)

            for K in KNN_KS:
                rows = []
                for cid, c_idx in zip(eligible, eligible_idx):
                    c_idx = int(c_idx)
                    scores = _score_knn_emb(c_idx, int(K))
                    seen_idx = X_mask[c_idx].indices
                    pos = test_pos.get(c_idx, set())
                    met = _eval_from_full_scores(scores, seen_idx, pos, K1=10, K2=ndcg_k)
                    met["label"] = int(cid)
                    rows.append(met)
                knnEmb = pd.DataFrame(rows)
                _add_run(all_runs, knnEmb, method=f"ConceptKNN_TextEmb_K{int(K)}", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)

                if smooth_enabled and add_smoothing_variants:
                    rows = []
                    for cid, c_idx in zip(eligible, eligible_idx):
                        c_idx = int(c_idx)
                        scores = _score_knn_emb_smooth(c_idx, int(K))
                        seen_idx = X_mask[c_idx].indices
                        pos = test_pos.get(c_idx, set())
                        met = _eval_from_full_scores(scores, seen_idx, pos, K1=10, K2=ndcg_k)
                        met["label"] = int(cid)
                        rows.append(met)
                    knnEmb_s = pd.DataFrame(rows)
                    _add_run(all_runs, knnEmb_s, method=f"ConceptKNN_TextEmb_K{int(K)}+SmoothTextEmb", cutoff=T, seed=-1, eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)

        # ---------------- ALS ----------------
        if run_als_flag:
            for seed in als_seeds:
                print(f"  ALS seed={seed} (GPU train={use_gpu_train})")
                als = fit_als(X_train, seed=int(seed), als_cfg=als_cfg, use_gpu_train=use_gpu_train)
                concept_factors, object_factors = extract_als_factors(als, n_concepts=n_concepts, n_objects=n_objects)

                rows = []
                for cid, c_idx in zip(eligible, eligible_idx):
                    c_idx = int(c_idx)
                    scores = object_factors @ concept_factors[c_idx]
                    seen_idx = X_mask[c_idx].indices
                    pos = test_pos.get(c_idx, set())
                    met = _eval_from_full_scores(scores, seen_idx, pos, K1=10, K2=ndcg_k)
                    met["label"] = int(cid)
                    rows.append(met)
                als_pc = pd.DataFrame(rows)
                _add_run(all_runs, als_pc, method="ALS", cutoff=T, seed=int(seed), eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)

                if smooth_enabled and add_smoothing_variants:
                    assert SMOOTH_NBR_IDX is not None and SMOOTH_NBR_W is not None

                    def _smooth_concept_factor(c_idx: int) -> np.ndarray:
                        nbrs = SMOOTH_NBR_IDX[c_idx]
                        w = SMOOTH_NBR_W[c_idx].astype(np.float32, copy=False)
                        u0 = concept_factors[c_idx]
                        if nbrs.size == 0:
                            return u0
                        uneigh = (w @ concept_factors[nbrs]).astype(np.float32, copy=False)
                        return (1.0 - smooth_lam) * u0 + smooth_lam * uneigh

                    rows = []
                    for cid, c_idx in zip(eligible, eligible_idx):
                        c_idx = int(c_idx)
                        u_s = _smooth_concept_factor(c_idx)
                        scores = object_factors @ u_s
                        seen_idx = X_mask[c_idx].indices
                        pos = test_pos.get(c_idx, set())
                        met = _eval_from_full_scores(scores, seen_idx, pos, K1=10, K2=ndcg_k)
                        met["label"] = int(cid)
                        rows.append(met)
                    als_s = pd.DataFrame(rows)
                    _add_run(all_runs, als_s, method="ALS+SmoothTextEmb", cutoff=T, seed=int(seed), eligible=eligible, strata=STRATA, ndcg_k=ndcg_k, strata_to_report=strata_to_report)

    results = pd.concat(all_runs, ignore_index=True)

    if f"ndcg@{ndcg_k}" in results.columns and f"ndcg@{ndcg_k}" != "ndcg@100":
        results = results.rename(columns={f"ndcg@{ndcg_k}": "ndcg@100"})

    cols = ["stratum", "n_concepts", "mrr", "recall@10", "recall@100", "ndcg@100", "method", "cutoff", "seed"]
    results = results[cols]

    out_path = OUT_DIR / str(cfg_get(cfg, "output.csv_path", f"{out_subdir}/eval_stratified_results.csv"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_path, index=False)
    print("[write]", out_path.resolve(), "| rows:", len(results))

    stable = OUT_DIR / out_subdir / "eval_stratified_results.csv"
    stable.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(stable, index=False)
    print("[write]", stable.resolve(), "| rows:", len(results))

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduce Table 1 raw CSV (Mode B) from YAML config.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    cfg, cfg_dir = load_yaml_config(args.config)
    run_from_config(cfg, cfg_dir=cfg_dir)


if __name__ == "__main__":
    main()
