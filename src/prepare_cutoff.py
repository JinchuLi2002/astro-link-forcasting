"""
prepare_cutoff.py

ArXiv release pipeline — data prep + Mode B split artifacts.

This file does:
1) Lock SIMBAD object resolution mapping (alias/name caches)
2) Construct mention-level (paper, concept, object) records with years + weights
3) Build global TRAIN vocab (concept_ids/object_ids) and temporal edge tables:
    - edges_train:  [label, object_id, year, raw_w]
    - edges_target: same schema, filtered to TRAIN vocab
    - pair_first_year_target: first appearance year per (label, object_id) in target
4) Implement Mode B split artifacts (as in Colab "FULL TEMPORAL EVAL … Mode B"):
    - train_pairs_from_edges_train(T)
    - seen_pairs_from_edges_target(T)
    - test_pairs_from_first_year_target(T)
    - build_sparse_train_matrix(train_pairs)
    - prepare_cutoff(T) returns:
        (train_pairs, test_pairs, eligible, X_train, train_seen, test_pos, mask_pairs)
    - writes artifacts to outputs/table1/T=YYYY/

No model training/eval is implemented here.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import unicodedata
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterator, Literal, Optional
from typing import Any, Mapping
from src.utils import cfg_get, resolve_dir, resolve_file


import numpy as np
import pandas as pd
import scipy.sparse as sp


# =============================================================================
# Repo-relative paths (override via env vars for convenience)
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("ASTRO_DATA_DIR", str(REPO_ROOT / "data")))
OUT_DIR = Path(os.environ.get("ASTRO_OUT_DIR", str(REPO_ROOT / "outputs")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Default filenames (match Colab notebook)
PAPERS_YEAR_PATH = DATA_DIR / "papers_year_mapping.csv"
CONCEPT_MAP_PATH = DATA_DIR / "papers_concepts_mapping.csv"
VOCAB_PATH = DATA_DIR / "concepts_vocabulary.csv"

PO_AGG_PATH = DATA_DIR / "paper_object_edges_llm_v2_agg.parquet"
PO_MENTIONS_PATH = DATA_DIR / "paper_object_edges_llm_v2_mentions.jsonl"

ALIAS_CACHE_PATH = DATA_DIR / "simbad_alias_cache.jsonl"  # optional
NAME_CACHE_PATH = DATA_DIR / "simbad_name_resolution_cache_llm_objects_with_otype_and_errors_clean.jsonl"
FAILURES_PATH = DATA_DIR / "simbad_resolution_failures.jsonl"  # optional

# =============================================================================
# Config override support (YAML-driven)
# =============================================================================
_PIPELINE_CFG: dict | None = None
_PIPELINE_CFG_DIR: Path | None = None


def apply_config(cfg: dict, *, cfg_dir: Path | None = None) -> None:
    """
    Apply YAML config overrides:
      - paths (data_dir/out_dir + filenames)
      - weights tables
      - alias-cache enable/disable
      - train/target EdgeConfig defaults via _default_cfgs()
    """
    global _PIPELINE_CFG, _PIPELINE_CFG_DIR
    global DATA_DIR, OUT_DIR
    global PAPERS_YEAR_PATH, CONCEPT_MAP_PATH, VOCAB_PATH
    global PO_AGG_PATH, PO_MENTIONS_PATH
    global ALIAS_CACHE_PATH, NAME_CACHE_PATH, FAILURES_PATH
    global ROLE_WEIGHT, STUDY_MODE_MULT, CONTEXT_ROLES
    global _GLOBAL_CACHE
    global alias2main, name2main, otype_by_main, _SIMBAD_READY

    _PIPELINE_CFG = cfg
    _PIPELINE_CFG_DIR = cfg_dir

    # --- dirs ---
    repo_root = Path(__file__).resolve().parents[1]

    data_dir = resolve_dir(
        cfg_get(cfg, "paths.data_dir", "data"),
        base_dir=repo_root,
        must_exist=False,
    )

    out_dir = resolve_dir(
        cfg_get(cfg, "paths.out_dir", "outputs"),
        base_dir=repo_root,
        create=True,
    )


    DATA_DIR = data_dir
    OUT_DIR = out_dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- files (resolve relative to data_dir unless absolute) ---
    PAPERS_YEAR_PATH = resolve_file(cfg_get(cfg, "paths.papers_year", "papers_year_mapping.csv"), base_dir=DATA_DIR)
    CONCEPT_MAP_PATH = resolve_file(cfg_get(cfg, "paths.concept_map", "papers_concepts_mapping.csv"), base_dir=DATA_DIR)
    VOCAB_PATH = resolve_file(cfg_get(cfg, "paths.concept_vocab", "concepts_vocabulary.csv"), base_dir=DATA_DIR)

    PO_AGG_PATH = resolve_file(cfg_get(cfg, "paths.po_agg", "paper_object_edges_llm_v2_agg.parquet"), base_dir=DATA_DIR)
    PO_MENTIONS_PATH = resolve_file(cfg_get(cfg, "paths.po_mentions", "paper_object_edges_llm_v2_mentions.jsonl"), base_dir=DATA_DIR)

    NAME_CACHE_PATH = resolve_file(
        cfg_get(cfg, "paths.simbad_name_cache", "simbad_name_resolution_cache_llm_objects_with_otype_and_errors_clean.jsonl"),
        base_dir=DATA_DIR,
    )

    # alias cache may be disabled (null) to reproduce Colab
    alias_raw = cfg_get(cfg, "paths.simbad_alias_cache", None)
    if alias_raw in (None, "", "null"):
        # point to a missing file so load_simbad_caches behaves like "no alias cache"
        ALIAS_CACHE_PATH = DATA_DIR / "__ALIAS_CACHE_DISABLED__"
    else:
        ALIAS_CACHE_PATH = resolve_file(alias_raw, base_dir=DATA_DIR)

    FAILURES_PATH = resolve_file(cfg_get(cfg, "paths.simbad_failures", "simbad_resolution_failures.jsonl"), base_dir=DATA_DIR)

    # --- weights tables (optional overrides) ---
    rw = cfg_get(cfg, "weights.role_weight", None)
    if isinstance(rw, dict) and rw:
        ROLE_WEIGHT = {str(k): float(v) for k, v in rw.items()}

    sm = cfg_get(cfg, "weights.study_mode_mult", None)
    if isinstance(sm, dict) and sm:
        STUDY_MODE_MULT = {str(k): float(v) for k, v in sm.items()}

    cr = cfg_get(cfg, "weights.context_roles", None)
    if isinstance(cr, (list, tuple, set)) and cr:
        CONTEXT_ROLES = set(str(x) for x in cr)

    # --- reset caches so new paths/weights take effect ---
    _GLOBAL_CACHE = {}
    alias2main = {}
    name2main = {}
    otype_by_main = {}
    _SIMBAD_READY = False

    print("[config] DATA_DIR =", DATA_DIR)
    print("[config] OUT_DIR  =", OUT_DIR)
    print("[config] alias_cache =", ALIAS_CACHE_PATH, "| exists =", ALIAS_CACHE_PATH.exists())
    print("[config] name_cache  =", NAME_CACHE_PATH, "| exists =", NAME_CACHE_PATH.exists())


def edge_config_from_dict(d: Mapping[str, Any], *, name: str) -> EdgeConfig:
    """
    Build an EdgeConfig from YAML mapping, with safe defaults.
    """
    if d is None:
        d = {}
    return EdgeConfig(
        name=str(d.get("name", name)),
        noreg=bool(d.get("noreg", True)),
        role_filter=str(d.get("role_filter", "all")),
        study_filter=str(d.get("study_filter", "all")),
        weighting=str(d.get("weighting", "role_x_mode")),
        paper_norm=str(d.get("paper_norm", "none")),
        paper_tau=float(d.get("paper_tau", 25.0)),
        force_mentions_jsonl=bool(d.get("force_mentions_jsonl", True)),
        cache=bool(d.get("cache", True)),
    )

CONTEXT_ROLES = {"comparison_or_reference", "calibration", "serendipitous_or_field_source"}


# =============================================================================
# Config tagger (used for mention-cache invalidation)
# =============================================================================
RoleFilter = Literal["all", "substantive", "primary_only"]
StudyFilter = Literal["all", "non_sim_only", "new_obs_only"]
PaperNorm = Literal["none", "cap_total", "l1"]


@dataclass(frozen=True)
class EdgeConfig:
    name: str = "baseline"
    noreg: bool = True

    # filters
    role_filter: RoleFilter = "all"
    study_filter: StudyFilter = "all"

    # weighting scheme
    weighting: Literal["role_x_mode", "binary"] = "role_x_mode"

    # per-paper normalization
    paper_norm: PaperNorm = "none"
    paper_tau: float = 25.0  # only used if paper_norm != "none"

    # data/caching
    force_mentions_jsonl: bool = True 
    cache: bool = True


def weights_hash() -> str:
    """Short stable hash of ROLE_WEIGHT + STUDY_MODE_MULT."""
    payload = {
        "ROLE_WEIGHT": dict(sorted(ROLE_WEIGHT.items())),
        "STUDY_MODE_MULT": dict(sorted(STUDY_MODE_MULT.items())),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.md5(blob.encode()).hexdigest()[:8]


def cfg_tag(cfg: EdgeConfig) -> str:
    core = asdict(cfg).copy()
    core_for_hash = {k: core[k] for k in core if k != "name"}
    blob = json.dumps(core_for_hash, sort_keys=True, separators=(",", ":"))
    h = hashlib.md5(blob.encode()).hexdigest()[:8]

    tau_part = f"{cfg.paper_tau:g}" if cfg.paper_norm != "none" else "none"
    wh = weights_hash() if cfg.weighting == "role_x_mode" else "binary"

    return (
        f"{cfg.name}"
        f"_rf={cfg.role_filter}"
        f"_sf={cfg.study_filter}"
        f"_w={cfg.weighting}"
        f"_pn={cfg.paper_norm}-{tau_part}"
        f"_noreg={cfg.noreg}"
        f"_wh={wh}"
        f"_{h}"
    )


# =============================================================================
# SIMBAD cache loaders (verbatim logic from Colab)
# =============================================================================
def read_jsonl(path: Path) -> Iterator[dict] | None:
    if not path.exists():
        return None
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def alias_norm(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    return " ".join(s.split()).lower()


# Very safe cleaning: unicode normalize + strip control chars + whitespace normalize
_CTRL_RE = re.compile(r"[\u0000-\u001F\u007F-\u009F]")


def safe_clean_name(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKC", s)
    s = _CTRL_RE.sub("", s)
    s = " ".join(s.strip().split())
    return s


def key_norm(s: str) -> str:
    return alias_norm(safe_clean_name(s))


alias2main: Dict[str, str] = {}
name2main: Dict[str, Optional[str]] = {}
otype_by_main: Dict[str, Optional[str]] = {}
_SIMBAD_READY: bool = False


def load_simbad_caches(
    alias_cache_path: Path = ALIAS_CACHE_PATH, name_cache_path: Path = NAME_CACHE_PATH
) -> None:
    """Populate alias2main/name2main/otype_by_main using v2-style caches."""
    global alias2main, name2main, otype_by_main, _SIMBAD_READY

    alias2main = {}
    name2main = {}
    otype_by_main = {}

    if alias_cache_path.exists():
        for row in read_jsonl(alias_cache_path) or []:
            main_id = row.get("main_id")
            if not isinstance(main_id, str) or not main_id.strip():
                continue
            aliases = row.get("aliases") or [main_id]
            if not isinstance(aliases, list):
                aliases = [main_id]
            for a in aliases:
                if not isinstance(a, str) or not a.strip():
                    continue
                k = key_norm(a)
                if k:
                    alias2main[k] = main_id
        print("Loaded alias cache entries:", len(alias2main))
    else:
        print("[WARN] No alias cache found at:", alias_cache_path)

    if name_cache_path.exists():
        for row in read_jsonl(name_cache_path) or []:
            # key: prefer alias_norm else query_name
            k = row.get("alias_norm")
            if not isinstance(k, str) or not k.strip():
                qn = row.get("query_name")
                if isinstance(qn, str) and qn.strip():
                    k = qn
                else:
                    continue
            k = key_norm(k)
            name2main[k] = row.get("main_id")

            # optional: object type by main_id (for noReg filtering)
            mid = row.get("main_id")
            if isinstance(mid, str) and mid.strip():
                otype_by_main.setdefault(mid, row.get("otype"))
        print("Loaded name-resolution cache entries:", len(name2main))
        print("Loaded otype_by_main entries:", len(otype_by_main))
    else:
        raise FileNotFoundError(f"Missing NAME_CACHE_PATH: {name_cache_path}")

    _SIMBAD_READY = True
    print("Caches ready.")


def map_main_id(name: str) -> Optional[str]:
    if not _SIMBAD_READY:
        load_simbad_caches()
    k = key_norm(name)
    mid = alias2main.get(k)
    if isinstance(mid, str) and mid.strip():
        return mid
    mid = name2main.get(k)
    if isinstance(mid, str) and mid.strip():
        return mid
    return None


# =============================================================================
# Paper-object evidence loaders + mention construction (verbatim logic from Colab)
# =============================================================================
def _uniq_sorted(xs):
    xs = [x for x in xs if isinstance(x, str) and x]
    return sorted(set(xs))


def load_po_mentions() -> pd.DataFrame:
    if not PO_MENTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Need {PO_MENTIONS_PATH.name} for mention-level filters; "
            f"got exists=False at {PO_MENTIONS_PATH}"
        )
    df = pd.read_json(PO_MENTIONS_PATH, lines=True, dtype={"arxiv_id": "string"})
    need = {"arxiv_id", "object_name_norm", "role", "study_mode"}
    missing = need - set(df.columns)
    if missing:
        raise ValueError(f"PO_MENTIONS missing columns: {missing} | cols={df.columns.tolist()}")
    df["arxiv_id"] = df["arxiv_id"].astype("string").str.strip()
    df["object_name_norm"] = df["object_name_norm"].astype("string")
    df["role"] = df["role"].astype("string")
    df["study_mode"] = df["study_mode"].astype("string")
    return df


def load_po_agg() -> pd.DataFrame:
    if not PO_AGG_PATH.exists():
        raise FileNotFoundError(f"Need {PO_AGG_PATH.name}; got exists=False at {PO_AGG_PATH}")
    po = pd.read_parquet(PO_AGG_PATH)
    need = {"arxiv_id", "object_name_norm", "obj_weight", "roles"}
    missing = need - set(po.columns)
    if missing:
        raise ValueError(f"PO_AGG missing columns: {missing} | cols={po.columns.tolist()}")
    po = po.copy()
    if len(po) > 0 and isinstance(po.iloc[0]["roles"], str):
        po["roles"] = po["roles"].map(lambda s: ast.literal_eval(s) if isinstance(s, str) else s)
    po["arxiv_id"] = po["arxiv_id"].astype("string").str.strip()
    po["object_name_norm"] = po["object_name_norm"].astype("string")
    po["obj_weight"] = po["obj_weight"].astype("float32")

    tot = po.groupby("arxiv_id")["obj_weight"].sum()
    print("po_agg total obj_weight per paper quantiles:")
    print(tot.quantile([0.5, 0.9, 0.95, 0.99, 0.999]))

    return po[["arxiv_id", "object_name_norm", "obj_weight", "roles"]]


def apply_paper_norm(po_resolved: pd.DataFrame, cfg: EdgeConfig) -> pd.DataFrame:
    if cfg.paper_norm == "none":
        return po_resolved

    po_resolved = po_resolved.copy()
    tot = po_resolved.groupby("arxiv_id")["obj_weight"].sum()

    if cfg.paper_norm == "cap_total":
        scale = (cfg.paper_tau / tot).clip(upper=1.0)
        po_resolved = po_resolved.join(scale.rename("scale"), on="arxiv_id")
        po_resolved["obj_weight"] = (po_resolved["obj_weight"] * po_resolved["scale"]).astype("float32")
        po_resolved = po_resolved.drop(columns=["scale"])
        return po_resolved

    if cfg.paper_norm == "l1":
        po_resolved = po_resolved.join(tot.rename("tot"), on="arxiv_id")
        po_resolved["obj_weight"] = (po_resolved["obj_weight"] / po_resolved["tot"]).astype("float32")
        po_resolved = po_resolved.drop(columns=["tot"])
        return po_resolved

    raise ValueError(f"Unknown paper_norm: {cfg.paper_norm}")


def is_region_like(main_id: str, otype: str | None) -> bool:
    s = (main_id or "").upper()
    t = (otype or "").upper()
    if "REG" in t or "FIELD" in t:
        return True
    if s.startswith("NAME ") and ("FIELD" in s or "REGION" in s):
        return True
    return False


def build_mentions(cfg: EdgeConfig) -> pd.DataFrame:
    tag = cfg_tag(cfg)
    cache_dir = OUT_DIR / "mentions_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"mentions__{tag}.parquet"

    if cfg.cache and cache_path.exists():
        print("[cache hit]", cache_path.name)
        return pd.read_parquet(cache_path)

    # ---- Load paper-year + paper->concept ----
    papers_year = pd.read_csv(PAPERS_YEAR_PATH, dtype={"arxiv_id": "string", "year": "int32"})
    papers_year["arxiv_id"] = papers_year["arxiv_id"].astype("string").str.strip()
    papers_year = papers_year.dropna(subset=["arxiv_id", "year"]).copy()
    year_map = dict(zip(papers_year["arxiv_id"].astype(str), papers_year["year"].astype(int)))

    concept_map = pd.read_csv(CONCEPT_MAP_PATH, dtype={"arxiv_id": "string", "label": "int32"})
    concept_map["arxiv_id"] = concept_map["arxiv_id"].astype("string").str.strip()
    concept_map = concept_map.dropna(subset=["arxiv_id", "label"]).copy()
    concept_map = concept_map.drop_duplicates(subset=["arxiv_id", "label"]).reset_index(drop=True)

    # ---- Build paper-object weights ----
    need_mentions = (
        cfg.force_mentions_jsonl
        or cfg.role_filter != "all"
        or cfg.study_filter != "all"
        or cfg.weighting != "role_x_mode"
    )

    if need_mentions:
        pm = load_po_mentions()

        # role filter
        if cfg.role_filter == "substantive":
            pm = pm[~pm["role"].isin(CONTEXT_ROLES)].copy()
        elif cfg.role_filter == "primary_only":
            pm = pm[pm["role"] == "primary_subject"].copy()
        elif cfg.role_filter != "all":
            raise ValueError(f"Unknown role_filter: {cfg.role_filter}")

        # study filter
        if cfg.study_filter == "non_sim_only":
            pm = pm[~pm["study_mode"].isin({"simulation_or_theory", "not_applicable"})].copy()
        elif cfg.study_filter == "new_obs_only":
            pm = pm[pm["study_mode"] == "new_observation"].copy()
        elif cfg.study_filter != "all":
            raise ValueError(f"Unknown study_filter: {cfg.study_filter}")

        # weighting
        if cfg.weighting == "binary":
            pm["mention_weight"] = np.float32(1.0)
        elif cfg.weighting == "role_x_mode":
            pm["role_weight"] = pm["role"].map(ROLE_WEIGHT).fillna(np.float32(0.75)).astype("float32")
            pm["mode_mult"] = pm["study_mode"].map(STUDY_MODE_MULT).fillna(np.float32(0.50)).astype("float32")
            pm["mention_weight"] = (pm["role_weight"] * pm["mode_mult"]).astype("float32")
        else:
            raise ValueError(f"Unknown weighting: {cfg.weighting}")

        pm = pm[pm["mention_weight"] > 0].copy()

        po_agg = (
            pm.groupby(["arxiv_id", "object_name_norm"], as_index=False)
            .agg(
                obj_weight=("mention_weight", "sum"),
                roles=("role", _uniq_sorted),
            )
        )
        po_agg["obj_weight"] = po_agg["obj_weight"].astype("float32")

    else:
        po_agg = load_po_agg()

    print("po_agg:", po_agg.shape, "| unique papers:", po_agg["arxiv_id"].nunique())

    # ---- Resolve to SIMBAD main_id ----
    po_agg = po_agg.copy()
    po_agg["object_id"] = po_agg["object_name_norm"].map(map_main_id)
    po_agg = po_agg.dropna(subset=["object_id"]).copy()
    po_agg["object_id"] = po_agg["object_id"].astype(str)

    # Deduplicate (paper, object_id) after resolution (multiple names can map to same object)
    def _merge_roles(series):
        out = set()
        for v in series:
            if isinstance(v, (list, tuple, set)):
                out.update(v)
        return sorted(out)

    po = (
        po_agg.groupby(["arxiv_id", "object_id"], as_index=False)
        .agg(obj_weight=("obj_weight", "sum"), roles=("roles", _merge_roles))
    )
    po["obj_weight"] = po["obj_weight"].astype("float32")

    # ---- Optional: noReg ----
    if cfg.noreg and isinstance(otype_by_main, dict) and len(otype_by_main) > 0:
        bad = po["object_id"].map(lambda mid: is_region_like(mid, otype_by_main.get(mid))).astype(bool)
        po = po[~bad].copy()

    # ---- Year ----
    po["year"] = po["arxiv_id"].astype(str).map(year_map).astype("Int32")
    po = po.dropna(subset=["year"]).copy()
    po["year"] = po["year"].astype(int)

    # ---- Optional: per-paper normalization ----
    po = apply_paper_norm(po, cfg)

    # ---- Join to concepts ----
    mentions = po.merge(concept_map, on="arxiv_id", how="inner")
    mentions["role_weight"] = mentions["obj_weight"].astype(np.float32)

    mentions = mentions.dropna(subset=["label", "object_id", "year"]).copy()
    mentions["label"] = mentions["label"].astype(int)
    mentions["object_id"] = mentions["object_id"].astype(str)
    mentions["year"] = mentions["year"].astype(int)

    # cache
    if cfg.cache:
        mentions.to_parquet(cache_path, index=False)
        print("[cache write]", cache_path.name)

    print(
        "mentions:",
        mentions.shape,
        "| years:",
        int(mentions["year"].min()),
        "→",
        int(mentions["year"].max()),
        "| concepts:",
        int(mentions["label"].nunique()),
        "| objects:",
        int(mentions["object_id"].nunique()),
    )
    return mentions

# =============================================================================
# STRATA construction (matches Colab: vocab CSV + NON_PHYS_CLASSES + regex heuristics)
# =============================================================================

NON_PHYS_CLASSES = {"Statistics & AI", "Numerical Simulation", "Instrumental Design"}

SURVEY_MEAS_RE = re.compile(
    r"\b("
    r"gaia|dr\s?[0-9]+|data\s+release|catalog|pipeline|survey|photometr|astrometr|parallax|proper\s+motion|psf|calibrat"
    r"|tess|kepler|k2|sdss|pan-?starrs|des|ztf|lsst|wise|2mass|hst|jwst|chandra|xmm|alma|vla|vlbi"
    r")\b",
    re.IGNORECASE,
)

def _is_survey_meas_row(row: pd.Series) -> bool:
    text = f"{row.get('concept','')} {row.get('description','')}"
    return bool(SURVEY_MEAS_RE.search(text))


def build_strata(
    *,
    vocab_path: Path = VOCAB_PATH,
    concept_ids: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """
    Build STRATA dict used for Table 1 filtering.

    Matches Colab logic:
      vocab = read_csv(VOCAB_PATH); rename label->concept_id
      NON_PHYS_CLASSES = {Statistics & AI, Numerical Simulation, Instrumental Design}
      SURVEY_MEAS_RE heuristic over (concept, description)
      STRATA = {all, physical_subset_excl_stats_sim_instr, nonphysical_only_stats_sim_instr, survey_or_measurement_keyword}

    We additionally intersect each stratum with the TRAIN universe `concept_ids`,
    so strata are immediately compatible with your evaluation universe.
    """
    if not vocab_path.exists():
        raise FileNotFoundError(
            f"Missing concept vocabulary CSV at {vocab_path}. "
            f"Expected something like data/concepts_vocabulary.csv"
        )

    vocab = pd.read_csv(vocab_path)
    if "label" in vocab.columns and "concept_id" not in vocab.columns:
        vocab = vocab.rename(columns={"label": "concept_id"})

    if "concept_id" not in vocab.columns:
        raise ValueError(
            f"VOCAB file missing concept id column. Expected 'label' or 'concept_id'. "
            f"Got columns: {vocab.columns.tolist()}"
        )

    # Required for NON_PHYS_CLASSES filter
    if "class" not in vocab.columns:
        raise ValueError(
            f"VOCAB file missing 'class' column required for NON_PHYS_CLASSES. "
            f"Got columns: {vocab.columns.tolist()}"
        )

    # Optional but expected by the regex heuristic
    for col in ["concept", "description"]:
        if col not in vocab.columns:
            vocab[col] = ""

    vocab = vocab.copy()
    vocab["concept_id"] = vocab["concept_id"].astype(int)

    vocab["is_nonphys_class"] = vocab["class"].isin(NON_PHYS_CLASSES)
    # Note: apply(axis=1) is fine here (runs once during global prep)
    vocab["is_survey_meas"] = vocab.apply(_is_survey_meas_row, axis=1)

    strata_sets = {
        "all": set(vocab["concept_id"].tolist()),
        "physical_subset_excl_stats_sim_instr": set(vocab.loc[~vocab["is_nonphys_class"], "concept_id"].tolist()),
        "nonphysical_only_stats_sim_instr": set(vocab.loc[vocab["is_nonphys_class"], "concept_id"].tolist()),
        "survey_or_measurement_keyword": set(vocab.loc[vocab["is_survey_meas"], "concept_id"].tolist()),
    }

    # Restrict to TRAIN universe (concept_ids)
    universe = set(int(x) for x in concept_ids.tolist())
    STRATA = {k: np.array(sorted(list(v & universe)), dtype=int) for k, v in strata_sets.items()}

    strata_meta = {
        "vocab_path": str(vocab_path),
        "n_vocab_rows": int(len(vocab)),
        "n_vocab_unique_concepts": int(vocab["concept_id"].nunique()),
        "non_phys_classes": sorted(list(NON_PHYS_CLASSES)),
        "survey_meas_regex": SURVEY_MEAS_RE.pattern,
        "sizes_in_universe": {k: int(len(v)) for k, v in STRATA.items()},
        "universe_n_concepts": int(len(concept_ids)),
    }

    return STRATA, strata_meta


def write_strata_artifacts(
    STRATA: dict[str, np.ndarray],
    strata_meta: dict,
    *,
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # json-friendly: lists
    strata_json = {k: v.astype(int).tolist() for k, v in STRATA.items()}
    (out_dir / "strata.json").write_text(json.dumps(strata_json, indent=2))
    (out_dir / "strata_meta.json").write_text(json.dumps(strata_meta, indent=2))
    print("[write] strata artifacts →", out_dir)


def load_strata_artifacts(*, out_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    spath = out_dir / "strata.json"
    mpath = out_dir / "strata_meta.json"
    if not spath.exists() or not mpath.exists():
        raise FileNotFoundError(f"Missing strata artifacts in {out_dir} (need strata.json + strata_meta.json)")
    strata_json = json.loads(spath.read_text())
    meta = json.loads(mpath.read_text())
    STRATA = {k: np.array(v, dtype=int) for k, v in strata_json.items()}
    return STRATA, meta


# =============================================================================
# Global vocab + temporal edge construction (matches Colab block)
# =============================================================================
def build_global_edges(cfg_train: EdgeConfig, cfg_target: EdgeConfig):
    """Construct edges_train/edges_target + TRAIN vocab + pair_first_year_target."""
    # Build train edges
    mentions_train = build_mentions(cfg_train)
    edges_train = mentions_train.rename(columns={"role_weight": "raw_w"})[
        ["label", "object_id", "year", "raw_w"]
    ].copy()

    # Build target edges
    mentions_target = build_mentions(cfg_target)
    edges_target = mentions_target.rename(columns={"role_weight": "raw_w"})[
        ["label", "object_id", "year", "raw_w"]
    ].copy()

    # Normalize dtypes (same as Colab)
    edges_train["label"] = edges_train["label"].astype(int)
    edges_train["object_id"] = edges_train["object_id"].astype(str)
    edges_train["year"] = edges_train["year"].astype(int)
    edges_train["raw_w"] = edges_train["raw_w"].astype(np.float32)

    edges_target["label"] = edges_target["label"].astype(int)
    edges_target["object_id"] = edges_target["object_id"].astype(str)
    edges_target["year"] = edges_target["year"].astype(int)
    edges_target["raw_w"] = edges_target["raw_w"].astype(np.float32)

    print(
        "edges_train:",
        edges_train.shape,
        "concepts:",
        edges_train["label"].nunique(),
        "objects:",
        edges_train["object_id"].nunique(),
    )
    print(
        "edges_target:",
        edges_target.shape,
        "concepts:",
        edges_target["label"].nunique(),
        "objects:",
        edges_target["object_id"].nunique(),
    )

    concept_ids = np.sort(edges_train["label"].unique()).astype(int)
    object_ids = np.sort(edges_train["object_id"].unique()).astype(str)

    concept_id_to_idx = {int(cid): i for i, cid in enumerate(concept_ids)}
    object_id_to_idx = {str(oid): i for i, oid in enumerate(object_ids)}

    print("Global vocab from TRAIN:", len(concept_ids), len(object_ids))

    # Restrict target edges to train vocab (important!)
    edges_target = edges_target[
        edges_target["label"].isin(concept_id_to_idx.keys())
        & edges_target["object_id"].isin(object_id_to_idx.keys())
    ].copy()

    print("edges_target after vocab filter:", edges_target.shape)

    # First-appearance year per (concept, object) pair (computed once)
    pair_first_year_target = (
        edges_target.groupby(["label", "object_id"], as_index=False)["year"]
        .min()
        .rename(columns={"year": "first_year"})
    )

    return edges_train, edges_target, concept_ids, object_ids, pair_first_year_target


def write_global_artifacts(
    edges_train: pd.DataFrame,
    edges_target: pd.DataFrame,
    concept_ids: np.ndarray,
    object_ids: np.ndarray,
    pair_first_year_target: pd.DataFrame,
    out_subdir: str = "table1/_global",
) -> Path:
    out = OUT_DIR / out_subdir
    out.mkdir(parents=True, exist_ok=True)

    edges_train.to_parquet(out / "edges_train.parquet", index=False)
    edges_target.to_parquet(out / "edges_target.parquet", index=False)
    pair_first_year_target.to_parquet(out / "pair_first_year_target.parquet", index=False)

    np.save(out / "concept_ids.npy", concept_ids)
    np.save(out / "object_ids.npy", object_ids)

    meta = {
        "data_dir": str(DATA_DIR),
        "papers_year_path": str(PAPERS_YEAR_PATH),
        "concept_map_path": str(CONCEPT_MAP_PATH),
        "po_agg_path": str(PO_AGG_PATH),
        "po_mentions_path": str(PO_MENTIONS_PATH),
        "alias_cache_path": str(ALIAS_CACHE_PATH),
        "name_cache_path": str(NAME_CACHE_PATH),
        "edges_train_rows": int(len(edges_train)),
        "edges_target_rows": int(len(edges_target)),
        "n_concepts": int(len(concept_ids)),
        "n_objects": int(len(object_ids)),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    STRATA, strata_meta = build_strata(vocab_path=VOCAB_PATH, concept_ids=concept_ids)
    write_strata_artifacts(STRATA, strata_meta, out_dir=out)

    print("[write] global artifacts →", out)
    return out


# =============================================================================
# Mode B temporal split builders + per-cutoff artifact writing (matches Colab)
# =============================================================================
_GLOBAL_CACHE: dict = {}

def _default_cfgs() -> tuple[EdgeConfig, EdgeConfig]:
    """
    Defaults used for global edge construction.
    If apply_config() was called and edge_configs exist in YAML, use those.
    """
    global _PIPELINE_CFG
    if isinstance(_PIPELINE_CFG, dict):
        cfg_train = edge_config_from_dict(cfg_get(_PIPELINE_CFG, "edge_configs.train", {}), name="train_all")
        cfg_target = edge_config_from_dict(cfg_get(_PIPELINE_CFG, "edge_configs.target", {}), name="target_all")
        return cfg_train, cfg_target

    # fallback (previous hardcoded defaults)
    cfg_train = EdgeConfig(
        name="train_all",
        noreg=True,
        role_filter="all",
        study_filter="all",
        weighting="role_x_mode",
        paper_norm="none",
        cache=False,
    )
    cfg_target = EdgeConfig(
        name="target_all",
        noreg=True,
        role_filter="all",
        study_filter="all",
        weighting="role_x_mode",
        paper_norm="none",
        cache=False,
    )
    return cfg_train, cfg_target



def load_global_artifacts(out_subdir: str = "table1/_global") -> dict:
    """Load global artifacts written by write_global_artifacts()."""
    out = OUT_DIR / out_subdir
    need = [
        out / "edges_train.parquet",
        out / "edges_target.parquet",
        out / "pair_first_year_target.parquet",
        out / "concept_ids.npy",
        out / "object_ids.npy",
    ]
    missing = [p for p in need if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing global artifacts: {[str(p) for p in missing]}")

    edges_train = pd.read_parquet(out / "edges_train.parquet")
    edges_target = pd.read_parquet(out / "edges_target.parquet")
    pair_first_year_target = pd.read_parquet(out / "pair_first_year_target.parquet")
    concept_ids = np.load(out / "concept_ids.npy")
    object_ids = np.load(out / "object_ids.npy")

    concept_ids = concept_ids.astype(int)
    object_ids = object_ids.astype(str)

    concept_id_to_idx = {int(cid): i for i, cid in enumerate(concept_ids)}
    object_id_to_idx = {str(oid): i for i, oid in enumerate(object_ids)}

    try:
        STRATA, strata_meta = load_strata_artifacts(out_dir=out)
    except FileNotFoundError:
        # Build + persist if missing (keeps older caches compatible)
        STRATA, strata_meta = build_strata(vocab_path=VOCAB_PATH, concept_ids=concept_ids)
        write_strata_artifacts(STRATA, strata_meta, out_dir=out)

    return dict(
        out_dir=out,
        edges_train=edges_train,
        edges_target=edges_target,
        pair_first_year_target=pair_first_year_target,
        concept_ids=concept_ids,
        object_ids=object_ids,
        concept_id_to_idx=concept_id_to_idx,
        object_id_to_idx=object_id_to_idx,
        n_concepts=int(len(concept_ids)),
        n_objects=int(len(object_ids)),
        STRATA=STRATA,
        strata_meta=strata_meta,
    )


def ensure_global_artifacts(
    cfg_train: EdgeConfig | None = None,
    cfg_target: EdgeConfig | None = None,
    *,
    out_subdir: str = "table1/_global",
    force_rebuild: bool = False,
) -> dict:
    """
    Ensure global artifacts exist on disk and are loaded into memory.
    This prevents re-running expensive joins when generating multiple cutoffs.
    """
    global _GLOBAL_CACHE
    if _GLOBAL_CACHE and not force_rebuild:
        return _GLOBAL_CACHE

    if cfg_train is None or cfg_target is None:
        cfg_train, cfg_target = _default_cfgs()

    if not force_rebuild:
        try:
            _GLOBAL_CACHE = load_global_artifacts(out_subdir=out_subdir)
            print("[global] loaded from disk:", _GLOBAL_CACHE["out_dir"])
            return _GLOBAL_CACHE
        except FileNotFoundError:
            pass

    # Build from scratch
    edges_train, edges_target, concept_ids, object_ids, pair_first_year_target = build_global_edges(
        cfg_train, cfg_target
    )
    write_global_artifacts(
        edges_train, edges_target, concept_ids, object_ids, pair_first_year_target, out_subdir=out_subdir
    )
    _GLOBAL_CACHE = load_global_artifacts(out_subdir=out_subdir)
    print("[global] rebuilt + loaded:", _GLOBAL_CACHE["out_dir"])
    return _GLOBAL_CACHE


def train_pairs_from_edges_train(T: int, *, edges_train: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Mode B: weights for ALS matrix come from TRAIN edges up to year T.

    Colab semantics:
      sub = edges_train[year<=T]
      total_w = sum(raw_w) per (label, object_id)
      value = log1p(total_w)
    """
    if edges_train is None:
        edges_train = ensure_global_artifacts()["edges_train"]
    sub = edges_train[edges_train["year"] <= int(T)]
    agg = (
        sub.groupby(["label", "object_id"], as_index=False)["raw_w"]
        .sum()
        .rename(columns={"raw_w": "total_w"})
    )
    agg["value"] = np.log1p(agg["total_w"].astype(np.float32))
    return agg[["label", "object_id", "value"]]


def seen_pairs_from_edges_target(T: int, *, edges_target: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Mode B: what counts as 'seen' for masking comes from TARGET edges up to year T.
    Binary mask is sufficient; ensure unique (label, object_id) rows.
    """
    if edges_target is None:
        edges_target = ensure_global_artifacts()["edges_target"]
    sub = edges_target[edges_target["year"] <= int(T)][["label", "object_id"]].drop_duplicates()
    sub = sub.copy()
    sub["value"] = np.float32(1.0)
    return sub


def test_pairs_from_first_year_target(
    T: int, *, pair_first_year_target: pd.DataFrame | None = None
) -> pd.DataFrame:
    """Pairs whose first appearance year in TARGET is strictly after T."""
    if pair_first_year_target is None:
        pair_first_year_target = ensure_global_artifacts()["pair_first_year_target"]
    return pair_first_year_target[pair_first_year_target["first_year"] > int(T)][["label", "object_id"]]


def build_sparse_train_matrix(
    train_pairs: pd.DataFrame, *, concept_id_to_idx=None, object_id_to_idx=None
) -> sp.csr_matrix:
    """Build CSR matrix X_train from (label, object_id, value) using global vocab ordering."""
    G = ensure_global_artifacts()
    if concept_id_to_idx is None:
        concept_id_to_idx = G["concept_id_to_idx"]
    if object_id_to_idx is None:
        object_id_to_idx = G["object_id_to_idx"]

    r = train_pairs["label"].map(concept_id_to_idx)
    c = train_pairs["object_id"].map(object_id_to_idx)
    if r.isna().any() or c.isna().any():
        bad_r = int(r.isna().sum())
        bad_c = int(c.isna().sum())
        raise ValueError(f"Index mapping failed: missing rows={bad_r}, missing cols={bad_c}. Check vocab filtering.")

    r = r.to_numpy(dtype=np.int32)
    c = c.to_numpy(dtype=np.int32)
    x = train_pairs["value"].to_numpy(dtype=np.float32)
    X = sp.coo_matrix((x, (r, c)), shape=(G["n_concepts"], G["n_objects"])).tocsr()
    return X


def _pairs_to_seen_sets(pairs_df: pd.DataFrame, *, concept_id_to_idx, object_id_to_idx) -> dict[int, set[int]]:
    """Concept_idx -> set(object_idx). Used by evaluation code (train_seen/test_pos)."""
    seen: dict[int, set[int]] = defaultdict(set)
    for r in pairs_df.itertuples(index=False):
        c = concept_id_to_idx[int(r.label)]
        o = object_id_to_idx[str(r.object_id)]
        seen[c].add(o)
    return seen


def write_cutoff_artifacts(
    T: int,
    train_pairs: pd.DataFrame,
    mask_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    eligible: np.ndarray,
    X_train: sp.csr_matrix,
    *,
    out_subdir: str = "table1",
    overwrite: bool = False,
    min_train_pos: int = 10,
) -> Path:
    out = OUT_DIR / out_subdir / f"T={int(T)}"
    if out.exists() and not overwrite:
        print("[cutoff] exists, skip (use --overwrite-cutoff):", out)
        return out
    out.mkdir(parents=True, exist_ok=True)

    train_pairs.to_parquet(out / "train_pairs.parquet", index=False)
    mask_pairs.to_parquet(out / "mask_pairs.parquet", index=False)
    test_pairs.to_parquet(out / "test_pairs.parquet", index=False)
    np.save(out / "eligible.npy", eligible.astype(int))
    sp.save_npz(out / "X_train.npz", X_train)

    meta = {
        "cutoff_year": int(T),
        "min_train_pos": int(min_train_pos),
        "train_pairs_rows": int(len(train_pairs)),
        "mask_pairs_rows": int(len(mask_pairs)),
        "test_pairs_rows": int(len(test_pairs)),
        "n_eligible": int(len(eligible)),
        "X_train_shape": [int(X_train.shape[0]), int(X_train.shape[1])],
        "X_train_nnz": int(X_train.nnz),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print("[write] cutoff artifacts →", out)
    return out


def prepare_cutoff(
    T: int,
    *,
    min_train_pos: int = 10,
    write: bool = True,
    out_subdir: str = "table1",
    overwrite: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, sp.csr_matrix, dict, dict, pd.DataFrame]:
    """
    Mode B prepare_cutoff(T) — verbatim Colab semantics.

    Returns:
      train_pairs, test_pairs, eligible, X_train, train_seen, test_pos, mask_pairs
    """
    G = ensure_global_artifacts()

    train_pairs = train_pairs_from_edges_train(T, edges_train=G["edges_train"])
    mask_pairs = seen_pairs_from_edges_target(T, edges_target=G["edges_target"])
    test_pairs = test_pairs_from_first_year_target(T, pair_first_year_target=G["pair_first_year_target"])

    X_train = build_sparse_train_matrix(
        train_pairs, concept_id_to_idx=G["concept_id_to_idx"], object_id_to_idx=G["object_id_to_idx"]
    ).tocsr()

    # Eligibility computed on TARGET sense (mask/test), not on train weights
    train_deg = mask_pairs.groupby("label")["object_id"].nunique()
    test_deg = test_pairs.groupby("label")["object_id"].nunique()

    eligible = np.array(
        [
            int(c)
            for c in G["concept_ids"]
            if (
                int(c) in train_deg.index
                and int(c) in test_deg.index
                and int(train_deg.loc[int(c)]) >= int(min_train_pos)
                and int(test_deg.loc[int(c)]) >= 1
            )
        ],
        dtype=int,
    )

    train_seen = _pairs_to_seen_sets(
        mask_pairs, concept_id_to_idx=G["concept_id_to_idx"], object_id_to_idx=G["object_id_to_idx"]
    )
    test_pos = _pairs_to_seen_sets(
        test_pairs, concept_id_to_idx=G["concept_id_to_idx"], object_id_to_idx=G["object_id_to_idx"]
    )

    if write:
        write_cutoff_artifacts(
            T,
            train_pairs=train_pairs,
            mask_pairs=mask_pairs,
            test_pairs=test_pairs,
            eligible=eligible,
            X_train=X_train,
            out_subdir=out_subdir,
            overwrite=overwrite,
            min_train_pos=min_train_pos,
        )

    return train_pairs, test_pairs, eligible, X_train, train_seen, test_pos, mask_pairs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Mode B cutoff artifacts (global + per-cutoff).")
    parser.add_argument("--min-train-pos", type=int, default=10, help="Eligibility threshold on TARGET-seen degree.")
    parser.add_argument("--force-rebuild-global", action="store_true", help="Rebuild global artifacts even if present.")
    parser.add_argument("--overwrite-cutoff", action="store_true", help="Overwrite per-cutoff artifacts if present.")
    parser.add_argument("--config", type=str, default=None, help="YAML config path (optional).")
    args = parser.parse_args()


    from src.utils import load_yaml_config
    cfg, cfg_dir = load_yaml_config(args.config)
    apply_config(cfg, cfg_dir=cfg_dir)

    cfg_train, cfg_target = _default_cfgs()
    ensure_global_artifacts(cfg_train, cfg_target, force_rebuild=args.force_rebuild_global)

    cutoffs: list[int] = []

    for T in cfg["cutoffs"]:
        prepare_cutoff(
            T,
            min_train_pos=args.min_train_pos,
            write=True,
            out_subdir="table1",
            overwrite=args.overwrite_cutoff,
        )
