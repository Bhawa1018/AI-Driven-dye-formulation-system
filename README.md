# AI-Driven Dye Formulation Platform for Textile Coloration

**Multi-Stage Scale-Up Optimization and Cost-Benefit Analysis**
Open University of Sri Lanka — Textile Engineering & Industrial Studies

A laboratory dye recipe approved at 0.01 kg does not reproduce at 100–500 kg.
Liquor ratio contracts, exhaustion rises, mechanical shear changes, and the
dyer compensates by trial and error. This project builds and evaluates a
platform that predicts the recipe required at each stage of the scale-up
ladder directly from the approved lab dip, and quantifies what that is worth.

---

## What this repository contains

| Step | Deliverable | File |
|---|---|---|
| 1 | Physics-grounded synthetic data generator | `src/data_generator.py` |
| 2 | Exploratory analysis and publication figures | `src/eda.py` |
| 3–4 | ML benchmark, validation, feature attribution | `src/ml_pipeline.py` |
| — | Per-stage production models for the app | `src/train_stage_models.py` |
| 5 | Techno-economic and environmental analysis | `src/cost_benefit.py` |
| 6 | Interactive prototype | `app.py` |
| 7 | Research paper / thesis draft | `RESEARCH_PAPER.md` |

Supporting modules: `src/config.py` (every physical constant, price and process
window in one auditable place), `src/shade_master.py` (the shade dictionary),
`src/viz_theme.py` (one validated visual system for all figures).

---

## Quick start

```bash
pip install -r requirements.txt

python run_all.py            # full study, with hyper-parameter search
python run_all.py --fast     # same pipeline, no search (minutes)

streamlit run app.py         # the interactive prototype
```

Steps can also be run individually, in this order:

```bash
python src/data_generator.py
python src/eda.py
python src/ml_pipeline.py
python src/train_stage_models.py
python src/cost_benefit.py
```

### Outputs

```
textile_dyeing_scaleup_master.csv      8,000 jobs x 155 columns (the dataset)
data/                                  shade master, mill-format export
outputs/figures/                       11 publication figures (300 dpi)
outputs/tables/                        every number quoted in the paper
outputs/models/                        champion model + per-stage models
```

---

## The five scale-up stages

| Stage | Batch | Liquor ratio | Typical machine |
|---|---|---|---|
| Lab Dip | 0.01 kg | 1:20 | Lab beaker dyer |
| 1 kg Sample | 1 kg | 1:15 – 1:20 | Sample Aventec / jet |
| 7 kg Sample | 7 kg | 1:12 – 1:15 | Sample Tonello / soft-flow |
| Mini Bulk | 20–75 kg | 1:10 – 1:12 | Bataji / soft-flow jet |
| Bulk | 100–500 kg | 1:6 – 1:10 | Industrial soft-flow jet |

---

## How the data is generated, and why that is defensible

Dye-house recipe archives are commercially sensitive, and the job-card workbook
supplied with this project holds two illustrative jobs — not enough to train
anything. The dataset is therefore synthetic, but its **generative mechanism is
stated explicitly** rather than being a draw from convenient distributions.

The recipe required at stage *s* to reproduce the lab-dip shade is

```
f_s = f_LR · f_shear · f_depth · f_substrate · f_machine · ε
```

where the leading term comes from partition theory. For a dye of partition
constant *K*, equilibrium exhaustion at liquor ratio *L* is `E = K/(K+L)`;
holding the shade fixed therefore requires `c_s/c_lab = (K+L_s)/(K+L_lab)`,
damped by an exponent γ because the dyer simultaneously re-optimises the
electrolyte.

**The generator is verified, not assumed.** `validate_dataset()` runs twelve
checks, including a structural-recovery test: regressing `log f` on its
mechanisms recovers the generative exponent γ = 0.300 as **0.290**, with
residual noise 0.0222 against the declared 0.0224. The physics is provably in
the data, with the intended elasticity.

The 499-shade master dictionary, the stage ladder, the machine names and the
87-column job-card layout all come from the mill's own workbooks.

---

## Three things worth knowing before reading the results

**1. The trial-count units are a correction, not a transcription.**
The brief quotes "3.8 → 1.2 trials per shade" and, separately, "reduce trials
from 4 to 1 at each stage". These cannot both hold: 3.8 attempts per stage
implies 19 dyeings per job, which contradicts the brief's own "3–5 iterative
trials". The platform adopts the coherent reading — **3.8 corrective trials per
shade, summed across the five stages** — and `config.py` documents it. The
arithmetic is in `config.py` §6.

**2. Calibrating to both industrial baselines forces bulk to be the hardest
stage.** A 40 % bulk right-first-time rate alone consumes 0.92 of the 3.8
corrective-trial budget. That is not an artefact; it is the premise of the
research — a lab recipe does not translate to bulk.

**3. The savings are derived, not asserted.** It would be easy and worthless to
assume RFT rises to 85 % and compute the consequences. Instead `cost_benefit.py`
measures the probability of landing outside the colour tolerance under three
regimes on held-out data — lab dip carried over, the mill's standing bulk
factor, and the model — and scales the rework rate by the measured ratio. If
the model were poor, that ratio would approach 1 and the savings would vanish.

---

## Methodological notes

**Leakage control.** Features are selected by an explicit whitelist and then
audited: any column belonging to a stage at or beyond the target, any
adjustment factor, any cost or outcome column, and the latent difficulty score
raise an error. A model that can see `bulk_adj_factor` reconstructs the answer
exactly.

**Target scaling is mandatory, including for trees.** The target vector mixes
dye concentrations (0.1–7 %owf), electrolyte (5–95 g/L) and a trial count
(1–7). A multi-output tree selects splits on summed variance reduction across
targets, so it is scale-sensitive in the target. Benchmarking without this
understated Random Forest at R² 0.57 instead of 0.96.

**The selection metric matches the application.** Hyper-parameter search and
champion selection both optimise concentration-weighted relative recipe error —
the quantity that decides whether a batch is accepted. Selecting on R² instead
measurably degraded it, because R² rewards getting large concentrations right
rather than keeping every concentration inside a relative tolerance band.

**The trial count is near-unpredictable, and the analysis says so.** It is
generated as `1 + Poisson(λ)`, so only the variance of the conditional mean is
reducible: the attainable R² ceiling is ≈ 0.054. Quoting the achieved R² of
≈ 0.03 without that bound would misreport an irreducible property of the
process as a modelling failure. What the platform shifts is the *rate*, which
Step 5 derives from recipe accuracy.

---

## Limitations

- The dataset is synthetic. Every result characterises the **method**, not a
  deployed system. Before production use the models must be re-fitted on real
  job-card history.
- No spectrophotometric data. The platform predicts recipes, not reflectance
  curves; ΔE is approximated by a relative-concentration tolerance.
- Costs are indicative 2024–25 South-Asian mill prices and should be replaced
  with the mill's actual contract prices.
- The economic model assumes avoided trials release capacity that is used.

---

## Environment notes

Built and verified on Windows 10, Python 3.13, with numpy 2.5 / pandas 3.0 /
scikit-learn 1.9.

Two platform-specific issues are handled in code rather than left to bite:

- **OneDrive file locks.** Files in a syncing folder are intermittently
  lock-protected. `config.write_csv()` and `viz_theme.save()` retry with
  backoff and fall back to a suffixed filename rather than losing a run.
- **Windows long paths.** `streamlit >= 1.42` ships a deeply nested template
  asset that exceeds the 260-character limit when installed under the Windows
  Store Python prefix, and the install fails midway. `requirements.txt` pins
  `streamlit < 1.42` for this reason.
