# Astro Link Forecasting

This repository accompanies the arXiv preprint:

**Predicting New Concept–Object Associations in Astronomy by Mining the Literature**  
Jinchu Li, Yuan‑Sen Ting, Alberto Accomazzi, Tirthankar Ghosal, Nesar Ramachandra  
arXiv: **2602.14335** (astro-ph.IM)

It provides the end-to-end experimental pipeline to construct a large-scale, literature-derived **concept–object** graph and to **forecast future concept–object associations** under a strict temporal evaluation protocol.

---

## Core task

> Given a cutoff year **T**, train on all concept–object associations observed up to **T**,  
> and evaluate how well different methods rank objects whose association with the concept first appears **after T**.

The default configuration reproduces:

- Results **with inference-time concept smoothing** (main paper results)
- Results **without smoothing** (ablation)

---

## Contents

- [1. Repository overview](#1-repository-overview)
- [2. Required external data](#2-required-external-data)
- [3. Installation](#3-installation)
- [4. Configuration](#4-configuration)
- [5. Sample workflow](#5-sample-workflow)
- [6. Reproducibility notes](#6-reproducibility-notes)
- [7. Citation](#7-citation)

---

# 1. Repository overview

This repository implements the complete forecasting workflow:

1. **Strict temporal split construction** (train/target cutoff protocol)
2. **Concept–object graph assembly** from literature-derived inputs
3. **Concept-neighbor construction** (for smoothing and embedding-based baselines)
4. **Training + evaluation** of forecasting methods
5. **Stratified metric aggregation** over concept subsets

---

# 2. Required external data

## 2.1 Concept data (AstroMLab 5)

Paper–concept associations and concept embeddings are sourced from:

**Ting et al. (2025), AstroMLab 5: Structured Summaries and Concept Extraction for ~400,000 Astrophysics Papers**

Place the following files in `data/`:

- `concepts_embeddings.npz`
- `concepts_vocabulary.csv`
- `papers_concepts_mapping.csv`
- `papers_year_mapping.csv`

> This repository does **not** redistribute AstroMLab 5 data.

## 2.2 Object extraction data (this work)

This repository expects mention-level LLM object extraction data:

- `paper_object_edges_llm_mentions.jsonl`
- SIMBAD name resolution cache: `simbad_name_resolution_cache_*.jsonl`

Each JSONL row corresponds to a single object mention in a paper and includes (at minimum):

- normalized object name
- semantic role
- study mode
- resolved SIMBAD identifier

All **concept–object edges and weights** are generated dynamically from these mention-level inputs.  
No precomputed weighted graph is required.

---

# 3. Installation

## Python version

Tested with Python **3.10+**.

## Install dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Configuration

All experiments are controlled via:

```text
config/table1.yaml
```

## 4.1 Edge weight construction

Edge weights are computed as:

```text
w(c,o) = log(1 + Σ_m  ρ_r(m) × γ_σ(m))
```

where:

- `ρ_r(m)` is the **role weight**
- `γ_σ(m)` is the **study-mode multiplier**

Weights are configurable under:

```yaml
weights:
  role_weight:
  study_mode_mult:
```

Changing these values changes the underlying graph and therefore the scientific question being evaluated.

## 4.2 Edge configuration (important)

Edge construction is controlled by:

```yaml
edge_configs:
  train:
  target:
```

These control:

- role filtering (`role_filter`)
- study filtering (`study_filter`)
- weighting scheme (`weighting`)
- per-paper normalization (`paper_norm`)
- region exclusion (`noreg`)
- mention-level reconstruction (`force_mentions_jsonl`)

### Recommended setting (reproduces the paper)

To reproduce the published results:

```yaml
role_filter: all
study_filter: all
weighting: role_x_mode
paper_norm: none
noreg: true
```

The evaluation assumes:

- Train and target graphs are built under **identical edge semantics**
- Only the temporal cutoff defines the split
- Stratification is applied **after** graph construction

## 4.3 Train vs. target configuration

The pipeline allows different configs for `train` and `target`, but this is **not recommended** for standard forecasting experiments.

Using different filters may:

- change which edges count as “seen”
- alter eligibility criteria
- introduce distribution shift
- create evaluation artifacts

For clarity and reproducibility, keep:

```yaml
edge_configs.train == edge_configs.target
```

## 4.4 Role and study filtering

Edge construction can optionally filter object mentions before aggregation.

### `role_filter`

Controls which semantic roles are retained:

- `all` — keep all object mentions (used in the paper)
- `substantive` — exclude context roles
- `primary_only` — retain only primary scientific targets

Context roles are defined under:

```yaml
weights:
  context_roles:
```

Default context roles:

```yaml
context_roles:
  - comparison_or_reference
  - calibration
  - serendipitous_or_field_source
```

In the main experiments, `role_filter: all` is used, so context roles are **included** but typically **downweighted** via smaller role weights.

### `study_filter`

Controls filtering by study type:

- `all` — retain all study modes (used in the paper)
- `non_sim_only` — exclude theory/simulation-only mentions
- `new_obs_only` — retain only new observational studies

Main results use:

```yaml
study_filter: all
```

### `noreg`

When enabled, objects classified as sky regions or fields (based on SIMBAD object type metadata) are excluded.

This prevents non-physical spatial regions (e.g., survey fields) from behaving like astrophysical objects.

Main results use:

```yaml
noreg: true
```

## 4.5 Stratified evaluation

Stratification (via `output.strata_to_report`) determines which **concepts** contribute to reported evaluation metrics.

Example:

```yaml
output:
  strata_to_report:
    - physical_subset_excl_stats_sim_instr
```

Stratification is applied **after** graph construction, meaning:

- the graph is built over the full concept universe
- temporal splits are computed on the full graph
- stratification only filters which concepts contribute to reported metrics
- no held-out information is used during graph construction

Training on all concepts and reporting on a subset (e.g., physical concepts) is valid and used in the paper.

### Available strata

| Stratum name | Definition |
|---|---|
| `all` | All concepts in the training universe |
| `physical_subset_excl_stats_sim_instr` | Concepts whose high-level class is **not** in {Statistics & AI, Numerical Simulation, Instrumental Design} |
| `nonphysical_only_stats_sim_instr` | Concepts whose class **is** in {Statistics & AI, Numerical Simulation, Instrumental Design} |
| `survey_or_measurement_keyword` | Concepts whose name/description matches a survey/instrument/measurement keyword regex |

### Notes on `survey_or_measurement_keyword`

This subset is defined using a heuristic regex applied to concept names and descriptions (e.g., Gaia, SDSS, photometry, calibration).

Important considerations:

- heuristic, crude text matching
- overlaps substantially with `nonphysical_only_stats_sim_instr`
- not a headline result in the paper
- included primarily for diagnostics/exploration

### Best practice

To reproduce the paper:

```yaml
output:
  strata_to_report:
    - physical_subset_excl_stats_sim_instr
```

Altering strata changes only **what is reported**, not how the graph is constructed.

## 4.6 Other key config fields

### `cutoffs`

```yaml
cutoffs: [2017, 2019, 2021, 2023]
```

Temporal evaluation years.

### `min_train_pos`

Minimum number of prior associations required for a concept to be evaluated.

### `smoothing`

Inference-time concept smoothing parameters.

### `als`

Implicit ALS hyperparameters:

- latent factors
- regularization
- iterations
- alpha
- seeds

To reproduce paper averages, use multiple seeds.

---

# 5. Sample workflow

From the repository root:

```bash
bash scripts/reproduce_table1.sh config/table1.yaml
```

This runs:

1. `prepare_cutoff.py`
2. `smoothing.py`
3. `train_eval.py`

Outputs:

```text
OUT_DIR/
  table1/
    _global/
    T=2017/
    T=2019/
    T=2021/
    T=2023/
    eval_stratified_results.csv
    table1.tex
```

---

# 6. Reproducibility notes

- Graph construction is deterministic given the configuration and input JSONL files.
- No pre-aggregated graph artifacts are required.

**Important:** altering edge construction changes the scientific object of study and should be clearly documented in derived experiments.

---

# 7. Citation

If you use this repository, please cite:

```bibtex
@misc{li2026predictingnewconceptobjectassociations,
      title={Predicting New Concept-Object Associations in Astronomy by Mining the Literature},
      author={Jinchu Li and Yuan-Sen Ting and Alberto Accomazzi and Tirthankar Ghosal and Nesar Ramachandra},
      year={2026},
      eprint={2602.14335},
      archivePrefix={arXiv},
      primaryClass={astro-ph.IM},
      url={https://arxiv.org/abs/2602.14335},
}
```
