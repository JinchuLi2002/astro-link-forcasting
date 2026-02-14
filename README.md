# Astro Link Forecasting

This repository accompanies the paper:

**Predicting New Concept–Object Associations in Astronomy by Mining the Literature**

It provides the full experimental pipeline used to construct a large-scale literature-derived concept–object graph and to forecast future associations under a temporal evaluation protocol.

The core scientific task is:

> Given a cutoff year *T*, train on all concept–object associations observed up to *T*,  
> and evaluate how well different methods rank objects whose association first appears after *T*.

The default configuration reproduces both:

- Results **with inference-time concept smoothing** (main paper results)
- Results **without smoothing**

---

# 1. Repository Overview

This repository implements the complete forecasting workflow:

1. Construction of temporal Mode B train/test splits  
2. Concept–object graph assembly from literature-derived data  
3. Concept-embedding neighbor construction (for smoothing and embedding-based baselines)  
4. Training and evaluation of forecasting methods  
5. Stratified metric aggregation  

---

# 2. Required External Data

## Concept Data (AstroMLab 5)

Paper–concept associations and concept embeddings are sourced from:

**Ting et al. (2025), AstroMLab 5: Structured Summaries and Concept Extraction for 400,000 Astrophysics Papers**

Required files (place in `data/`):

- `concepts_embeddings.npz`
- `concepts_vocabulary.csv`
- `papers_concepts_mapping.csv`
- `papers_year_mapping.csv`

This repository does not redistribute AstroMLab 5 data.

---

## Object Extraction Data (This Work)

This repository expects mention-level LLM object extraction data:

- `paper_object_edges_llm_mentions.jsonl`
- SIMBAD name resolution cache (`simbad_name_resolution_cache_*.jsonl`)

Each JSONL row represents a single object mention in a paper, including:

- normalized object name
- semantic role
- study mode
- resolved SIMBAD identifier

All concept–object edges and weights are generated dynamically from these mention-level inputs.  
No precomputed weighted graph is required.

---

# 3. Installation

### Python Version

Tested with Python 3.10+.

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 4. Configuration

All experiments are controlled via:

```
config/table1.yaml
```

---

## Edge Weight Construction

Edge weights are computed as:

w(c,o) = log(1 + Σ_m ρ_r(m) × γ_σ(m))

where:

- ρ_r(m) = role weight  
- γ_σ(m) = study-mode multiplier  

Weights are configurable under:

```yaml
weights:
  role_weight:
  study_mode_mult:
```

Changing these values changes the underlying graph and therefore the scientific question being evaluated.

---

## Edge Configuration (Important)

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

### Recommended Setting (Reproduces the Paper)

To reproduce the published results:

```yaml
role_filter: all
study_filter: all
weighting: role_x_mode
paper_norm: none
noreg: true
```

The evaluation assumes:

- Train and target graphs are built under identical edge semantics
- Only temporal cutoff defines the split
- Stratification is applied after graph construction

---

## Train vs. Target Configuration

The pipeline allows different configs for `train` and `target`, but this is **not recommended** for standard forecasting experiments.

Using different filters may:

- Change which edges count as "seen"
- Alter eligibility criteria
- Introduce distribution shift
- Create evaluation artifacts

For scientific clarity and reproducibility, keep:

```yaml
edge_configs.train == edge_configs.target
```

---

## Stratified Evaluation

Stratification (via `output.strata_to_report`) determines which *concepts* are included when reporting evaluation metrics.

Example:

```yaml
output:
  strata_to_report:
    - physical_subset_excl_stats_sim_instr
```

Stratification is applied **after graph construction**.

This means:

- The concept–object graph is built over the full concept universe.
- Temporal splits are computed on the full graph.
- Stratification only filters which concepts contribute to reported metrics.
- No held-out information is used during graph construction.

Training on all concepts and reporting on a subset (e.g., physical concepts) is valid and used in the paper.

---

### Available Strata

The following concept subsets are constructed from the concept vocabulary:

| Stratum name | Definition |
|--------------|------------|
| `all` | All concepts in the training universe |
| `physical_subset_excl_stats_sim_instr` | Concepts whose high-level class is **not** in {Statistics & AI, Numerical Simulation, Instrumental Design} |
| `nonphysical_only_stats_sim_instr` | Concepts whose class is in {Statistics & AI, Numerical Simulation, Instrumental Design} |
| `survey_or_measurement_keyword` | Concepts whose name or description matches a survey/instrument/measurement keyword regex |

---

### Notes on `survey_or_measurement_keyword`

The `survey_or_measurement_keyword` subset is defined using a heuristic regular expression applied to concept names and descriptions (e.g., Gaia, SDSS, photometry, calibration, etc.).

Important considerations:

- This subset is heuristic and based on crude text matching.
- It overlaps substantially with the `nonphysical_only_stats_sim_instr` subset.
- It is not reported as a headline result in the paper.
- Empirically, performance on this subset is similar to or weaker than the physical subset.

It is included primarily for diagnostic and exploratory analysis rather than as a primary evaluation target.

---

### Best Practice

For reproducing the paper:

```yaml
output:
  strata_to_report:
    - physical_subset_excl_stats_sim_instr
```

Altering strata changes only which concepts are reported, not how the graph is constructed.

If reporting on alternative strata, this should be clearly documented in derived experiments.


---

## Other Key Config Fields

### cutoffs

```yaml
cutoffs: [2017, 2019, 2021, 2023]
```

Temporal evaluation years.

### min_train_pos

Minimum number of prior associations required for a concept to be evaluated.

### smoothing

Inference-time concept smoothing parameters.

### als

Implicit ALS hyperparameters:

- latent factors
- regularization
- iterations
- alpha
- seeds

To reproduce paper averages, use multiple seeds.

---

# 5. Sample Workflow

From the repository root:

```bash
bash scripts/reproduce_table1.sh config/table1.yaml
```

This runs:

1. `prepare_cutoff.py`
2. `smoothing.py`
3. `train_eval.py`

Outputs:

```
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

# 6. Reproducibility Guarantees

- Graph construction is deterministic given config.
- Temporal splits follow strict Mode B semantics.
- Weighting is config-defined.
- No pre-aggregated graph artifacts are required.

Altering edge construction changes the scientific object of study and should be clearly documented in derived experiments.

---

# 7. Citation

(placeholder — add citation after review period)
