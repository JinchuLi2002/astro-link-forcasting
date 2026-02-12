# src/utils.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Union

PathLike = Union[str, os.PathLike]


# -----------------------------
# Core filesystem helpers
# -----------------------------
def ensure_dir(path: PathLike) -> Path:
    """
    Ensure a directory exists (mkdir -p) and return it as a Path.
    """
    p = Path(path).expanduser()
    p.mkdir(parents=True, exist_ok=True)
    return p


def ensure_parent_dir(path: PathLike) -> Path:
    """
    Ensure the parent directory of a file path exists and return the file path as a Path.
    """
    p = Path(path).expanduser()
    ensure_dir(p.parent)
    return p


def find_repo_root(
    start: Optional[PathLike] = None,
    markers: Tuple[str, ...] = (".git", "pyproject.toml", "setup.cfg", "requirements.txt"),
) -> Path:
    """
    Walk upward from `start` (or CWD) to find a repo root by marker files/dirs.
    Falls back to CWD if no marker is found.
    """
    cur = Path(start or os.getcwd()).resolve()
    for parent in (cur, *cur.parents):
        for m in markers:
            if (parent / m).exists():
                return parent
    return cur


# -----------------------------
# YAML config loading
# -----------------------------
def load_yaml_config(path: PathLike) -> Tuple[Dict[str, Any], Path]:
    """
    Load a YAML config and return (cfg, cfg_dir).

    - Expands env vars in all string values (e.g., ${HOME})
    - Does NOT auto-resolve paths to Path objects; use resolve_* helpers below.
    """
    cfg_path = Path(path).expanduser().resolve()
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config YAML not found: {cfg_path}")

    try:
        import yaml  # PyYAML
    except ImportError as e:
        raise ImportError(
            "PyYAML is required for load_yaml_config(). Install with: pip install pyyaml"
        ) from e

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}
    if not isinstance(cfg, dict):
        raise ValueError(f"Top-level YAML must be a mapping/dict. Got: {type(cfg)}")

    cfg = _expand_env_vars(cfg)
    return cfg, cfg_path.parent


def _expand_env_vars(obj: Any) -> Any:
    """
    Recursively expand environment variables in string fields.
    """
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env_vars(x) for x in obj]
    if isinstance(obj, tuple):
        return tuple(_expand_env_vars(x) for x in obj)
    if isinstance(obj, dict):
        return {k: _expand_env_vars(v) for k, v in obj.items()}
    return obj


# -----------------------------
# Path resolution helpers
# -----------------------------
def resolve_path(
    raw: PathLike,
    *,
    base_dir: Optional[Path] = None,
    must_exist: bool = False,
) -> Path:
    """
    Resolve a path-like value into an absolute Path.

    - Expands ~ and env vars
    - If relative and base_dir is provided, resolves relative to base_dir
    - If must_exist=True, raises if path does not exist
    """
    if raw is None:  # type: ignore[comparison-overlap]
        raise ValueError("resolve_path() got None")

    s = raw
    # Expand env vars only if raw is a string (load_yaml_config already expands, but safe)
    if isinstance(raw, str):
        s = os.path.expandvars(raw)

    p = Path(s).expanduser()

    if base_dir is not None and not p.is_absolute():
        p = base_dir / p

    p = p.resolve(strict=False)

    if must_exist and not p.exists():
        raise FileNotFoundError(f"Path does not exist: {p}")

    return p


def resolve_dir(
    raw: PathLike,
    *,
    base_dir: Optional[Path] = None,
    create: bool = False,
    must_exist: bool = False,
) -> Path:
    """
    Resolve a directory path. Optionally create it.
    """
    p = resolve_path(raw, base_dir=base_dir, must_exist=False)
    if create:
        ensure_dir(p)
    if must_exist and (not p.exists() or not p.is_dir()):
        raise NotADirectoryError(f"Directory does not exist: {p}")
    return p


def resolve_file(
    raw: PathLike,
    *,
    base_dir: Optional[Path] = None,
    must_exist: bool = False,
    ensure_parent: bool = False,
) -> Path:
    """
    Resolve a file path. Optionally ensure parent dir exists.
    """
    p = resolve_path(raw, base_dir=base_dir, must_exist=False)
    if ensure_parent:
        ensure_parent_dir(p)
    if must_exist and (not p.exists() or not p.is_file()):
        raise FileNotFoundError(f"File does not exist: {p}")
    return p


def cfg_get(cfg: Mapping[str, Any], key: str, default: Any = None) -> Any:
    """
    Support dotted key access: cfg_get(cfg, 'paths.data_dir').
    """
    cur: Any = cfg
    for part in key.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur
