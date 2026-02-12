# Astro Link Forecasting

TODO: 
1. add citations

This repository accompanies the paper:

**Predicting New Concept–Object Associations in Astronomy by Mining the Literature**

It provides the full experimental pipeline used to construct a large-scale literature-derived concept–object graph and to forecast future associations under a temporal evaluation protocol.

The core scientific task is:

> Given a cutoff year *T*, train on all concept–object associations observed up to *T*,  
> and evaluate how well different methods rank objects whose association first appears after *T*.

The default configuration produces both:
- Results **with inference-time concept smoothing** (main results in the paper), and  
- Results **without smoothing**.

---

# 1. Repository Overview

This repository implements the complete forecasting workflow described in the paper:

1. Construction of temporal train/test splits
2. Concept–object graph assembly from literature-derived data
3. Concept embedding neighbor construction for smoothing and embedding-based baselines
4. Training and evaluation of forecasting methods
5. Stratified metric aggregation (i.e. how does the performance differ if we only predict physical concepts vs. survey-related concepts?)

---

# 2. Required External Data

This repository depends on data released as part of:

**Ting et al. (2025), AstroMLab 5: Structured Summaries and Concept Extraction for 400,000 Astrophysics Papers**

The following files are required and should be placed in `data/`:

- `concepts_embeddings.npz`
- `concepts_vocabulary.csv`
- `papers_concepts_mapping.csv`
- `papers_year_mapping.csv`

This repository does not redistribute AstroMLab 5 data.

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

Experiment settings can be defined in:

```
config/table1.yaml
```

Key configuration fields include:

- `DATA_DIR`: path to input data files
- `OUT_DIR`: directory for experiment outputs
- `cutoffs`: e.g. `[2017, 2019, 2021, 2023]`
- ALS hyperparameters (rank, regularization, iterations)
- Smoothing hyperparameters (`k`, `lambda`)
- K values for KNN baselines

All experiments are driven by this configuration file.

---

# 5. Sample Workflow

From the repository root:

```bash
bash scripts/reproduce_table1.sh config/table1.yaml
```

This executes the full forecasting workflow:

1. `prepare_cutoff.py`
   - Builds the global concept–object universe
   - Constructs temporal train/test splits for each cutoff

2. `smoothing.py`
   - Builds concept embedding k-nearest-neighbor tables
   - Caches smoothing weights for inference-time propagation

3. `train_eval.py`
   - Trains and evaluates forecasting methods
   - Applies inference-time concept smoothing
   - Computes stratified metrics
   - Writes:
     - `eval_stratified_results.csv`
     - `table1.tex`

---

## Output Structure

Outputs are written to:

```
OUT_DIR/
  table1/
    _global/                  # cached vocab and smoothing tables
    T=2017/                   # per-cutoff artifacts
    T=2019/
    T=2021/
    T=2023/
    eval_stratified_results.csv
    table1.tex
```

The CSV file contains stratified metrics for all methods and cutoffs.  
The LaTeX file corresponds to the main results table in the paper.

# 6. Reproducibility and Experimental Control

- Random seeds are fixed where applicable.

---

# 7. Citation

