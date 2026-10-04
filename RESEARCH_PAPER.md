# AI-Driven Dye Formulation for Textile Coloration: Multi-Stage Scale-Up Optimization and Cost-Benefit Analysis

**Department of Textile Engineering & Industrial Studies, Open University of Sri Lanka**

---

## Abstract

Laboratory dye recipes approved at 0.01 kg do not reproduce at production scale. Between the lab dip and a 100–500 kg bulk batch the liquor ratio contracts from roughly 1:20 to 1:8, equilibrium exhaustion rises, mechanical shear and wash-off conditions change, and the dyer closes the gap by iterative correction. The resulting right-first-time (RFT) rate at bulk is typically around 40%, and each shade consumes several corrective dyeings before sign-off.

This study develops and evaluates an end-to-end machine-learning platform that predicts the recipe required at each of five scale-up stages directly from the approved lab dip. Because dye-house recipe archives are commercially sensitive and exist mainly as unstructured job cards, a physics-grounded synthetic population of 8,000 scale-up programmes was generated from a mill's own 499-shade master dictionary and five-stage job-card schema. The generator's mechanism is stated explicitly: the recipe correction factor is the product of an exhaustion term derived from partition theory, a mechanical-shear term, a depth-saturation term, a substrate-accessibility term and a machine bias. The dataset is verified rather than assumed — a structural-recovery regression recovers the generative exhaustion exponent γ = 0.300 as 0.290 with residual noise 0.0222 against a declared 0.0224, and twelve invariant checks pass.

Nine model families were benchmarked across two prediction architectures. Hyper-parameter search and champion selection both optimise concentration-weighted relative recipe error — the quantity that decides whether a batch is accepted — rather than R², because selecting on R² measurably degraded the decision metric. A tuned XGBoost model predicting bulk directly from the lab dip achieved 2.66% relative recipe error (R² = 0.9937 over the six recipe targets, MAPE 3.62% on dye concentrations), against 3.89% for the chained sequential architecture, which compounds error at each link, and 12.88% for ordinary least squares.

Operational and financial gains were **derived from measured model error, not assumed**. The probability of landing outside the colour-acceptance tolerance was measured on held-out data under three regimes — lab dip carried over (85.6%), the mill's standing bulk factor (77.8%), and the model (31.1%) — and the rework rate scaled by the resulting ratio ρ = 0.400. Corrective trials per shade fall from 3.78 to 0.80, exceeding the programme target of 1.2. Bulk RFT rises from 40.2% to **70.2%, which does not meet the 85% target**; the analysis quantifies the shortfall as a specification, showing that 85% requires the off-tolerance rate to fall from 31.1% to 14.1%, roughly halving the model's recipe error. Costs fall 35.4% per programme, giving annual savings of USD 130,059 across 420 programmes, with 3,760 m³ of water, 107.9 MWh of thermal energy and 23.1 t CO₂e avoided, and a 4.5-month payback on an assumed USD 46,000 capital outlay.

**Keywords:** textile dyeing; scale-up; reactive dyes; right-first-time; machine learning; gradient boosting; dye exhaustion; liquor ratio; techno-economic analysis; CRISP-DM

---

## 1. Introduction and Problem Statement

### 1.1 The scale-up discontinuity

Colour development in a textile dye-house is a staged process. A shade is first matched in a laboratory beaker at 0.01 kg, then re-dyed at 1 kg, 7 kg, mini bulk (20–75 kg) and finally bulk (100–500 kg). Each step is nominally a reproduction of the one before, and in practice each is a renegotiation.

The reason is physical rather than procedural. A lab dip runs at a liquor ratio near 1:20 because small beakers need generous bath volumes for circulation. A bulk soft-flow jet runs at 1:6 to 1:10 because water, salt and steam are charged by volume. For a dye of fixed substantivity, exhaustion is a function of that ratio: shortening the liquor drives more dye onto the fibre for the same applied concentration. Holding the shade constant therefore requires *less* dye at bulk than at lab dip — but not by a fixed amount, because the saving is partly cancelled by electrolyte re-optimisation, partly offset by increased mechanical shear and wash-off losses, and partly reversed in deep shades where the bath approaches saturation.

The dyer resolves this by iteration. Industry RFT rates at bulk sit near 40%, and a typical shade absorbs three to five corrective dyeings across the ladder. Each correction consumes dyestuff, electrolyte, water, steam, effluent capacity and — most expensively — production machine time.

### 1.2 Why this is a prediction problem

The correction a dyer applies is not arbitrary. It is a function of information available *before* the batch is dyed: the approved lab-dip recipe, the substrate, the shade depth, the dye chemistry, and the planned machine, batch size and liquor ratio. If that function can be learned, the first bulk attempt can be issued at the recipe the fourth attempt would have reached.

This frames the problem as supervised multi-output regression, with an important caveat developed in Section 4.4: the number of trials a batch actually needs is substantially irreducible noise, whereas the *recipe* is highly predictable. Confusing the two leads to misreading a sound model as a failure.

### 1.3 Contributions

1. A physics-grounded synthetic data generator for multi-stage dye-house scale-up whose generative mechanism is stated in closed form and **verified by structural parameter recovery**, not merely asserted to be realistic.
2. A benchmark of nine model families across direct and sequential architectures, with an explicit anti-leakage feature contract and a selection metric aligned to the industrial acceptance criterion.
3. A demonstration that multi-output target scaling is **mandatory, including for tree ensembles** — omitting it understated Random Forest by 0.40 in R² and would have produced a materially wrong methodological conclusion.
4. A techno-economic analysis in which every operational gain is derived from measured held-out model error through an explicit mechanism, including a quantified account of a programme target the platform **does not** meet.

---

## 2. Literature Review and Theoretical Framework

### 2.1 Exhaustion and the liquor-ratio dependence

For a dye distributed between bath and fibre at equilibrium, the partition description gives a fractional exhaustion

$$E = \frac{K}{K + L}$$

where *L* is the liquor ratio in litres per kilogram of substrate and *K* is a dimensionless partition constant proportional to the dye's affinity for the fibre. A dye with high substantivity (large *K*) exhausts well even in a long bath and is correspondingly insensitive to liquor-ratio change; a low-affinity dye is strongly sensitive.

If a shade of fixed depth must be reproduced at a new liquor ratio, the applied concentration must satisfy $c_s E_s = c_{lab} E_{lab}$, hence

$$\frac{c_s}{c_{lab}} = \frac{E_{lab}}{E_s} = \frac{K + L_s}{K + L_{lab}}$$

For *K* = 11 (a vinyl-sulphone system), moving from 1:20 to 1:8 gives a ratio of 0.613 — a 39% reduction. Mill practice does not show reductions of that magnitude; observed lab-to-bulk factors cluster between 0.85 and 1.10. The discrepancy is informative: the dyer does not hold the rest of the recipe fixed. Electrolyte is re-optimised at every stage, which claws back much of the exhaustion gain. This study therefore damps the term with an exponent γ ∈ (0,1),

$$f_{LR} = \left(\frac{K + L_s}{K + L_{lab}}\right)^{\gamma}$$

and treats γ as an identifiable parameter of the process rather than a fudge factor — Section 4.1 recovers it from the generated data.

### 2.2 Colour difference and acceptance

Visual acceptance is conventionally expressed through a colour-difference metric such as ΔE_CMC(2:1), with commercial tolerances typically ΔE ≤ 1.0. This study does not model reflectance spectra, so the Kubelka–Munk route from concentration to reflectance is not instantiated. Instead, acceptance is expressed as a relative tolerance on the recipe itself: a batch is re-dyed when the concentration-weighted relative deviation of the four-dye recipe exceeds 3%, which corresponds approximately to ΔE_CMC ≈ 1.0 at medium depth. Section 6.2 records this as a limitation.

The Kubelka–Munk function $K/S = (1-R)^2 / 2R$ remains the standard basis for recipe prediction from reflectance, and its additivity over colourant mixtures is what makes trichromatic formulation tractable. A production successor to this work would predict in K/S space and convert to concentrations.

### 2.3 Machine learning in textile coloration

Published work in this area concentrates on three tasks: colour-recipe prediction from reflectance using neural networks, dyeing-defect classification from imagery, and process-parameter optimisation using evolutionary methods. Multi-stage scale-up — predicting the *production* recipe from the *laboratory* recipe across a ladder of intermediate batch sizes — is comparatively under-addressed, despite being where the economic loss concentrates. Gradient-boosted decision trees are a natural fit for this tabular, moderately sized, heavily categorical problem, and the benchmark in Section 4.2 bears this out.

### 2.4 CRISP-DM as the organising methodology

The study follows the Cross-Industry Standard Process for Data Mining: business understanding (Section 1), data understanding and preparation (Section 3), modelling (Section 3.5), evaluation (Section 4) and deployment (Section 4.6, the prototype). The iterative character of CRISP-DM proved material rather than decorative: the evaluation phase twice sent the work back to earlier phases, once over trial-count units (Section 3.3) and once over the model-selection metric (Section 3.6).

---

## 3. Research Methodology

### 3.1 Data provenance

Three artefacts were supplied by the industrial partner:

| Artefact | Content | Use |
|---|---|---|
| 499-shade master dictionary | Shade name, family, depth label, colour direction | Shade population, verbatim |
| 87-column job-card workbook | Five-stage recipe layout for two illustrative jobs | Schema, machine names, stage structure |
| Chemical and costing ledger | Departmental chemical consumption and rates | Cost-model orientation |

The 499 shades span 14 families, dominated by Green (92), Blue (66), Yellow (49) and Red (46). Their 11 descriptive depth labels were mapped onto four canonical depth classes, giving 211 Medium, 162 Dark, 97 Pastel/Light and 29 Extra Dark.

Two jobs are not a training set. The shade dictionary, the stage ladder and the schema were therefore retained exactly, and the *population* of jobs was generated.

### 3.2 The generative model

For each job the lab dip is sampled first, then the recipe required at each larger stage to reproduce the same visual depth is derived from a net adjustment factor

$$f_s = f_{LR,s} \cdot f_{shear,s} \cdot f_{depth} \cdot f_{substrate} \cdot f_{machine,s} \cdot \varepsilon$$

| Term | Form | Mechanism |
|---|---|---|
| $f_{LR}$ | $\left(\frac{K+L_s}{K+L_{lab}}\right)^{0.30}$ | Exhaustion gain from liquor contraction, damped by electrolyte re-optimisation |
| $f_{shear}$ | $1 + 0.0022\,(\text{RPM}_s - \text{RPM}_{lab})$ | Higher agitation strips more hydrolysed dye at wash-off |
| $f_{depth}$ | $1 + \delta_{depth}$, δ ∈ [−0.012, +0.052] | Bath saturation and dye–dye aggregation in deep shades |
| $f_{substrate}$ | $\frac{1}{a_{comp} a_{pre}}\left(1 + 0.011\frac{\text{gsm}-150}{100}\right)$ | Fibre accessibility and diffusion resistance in heavy constructions |
| $f_{machine}$ | 0.991 – 1.021 | Soft-flow vs winch vs beaker hydrodynamics |
| ε | Lognormal, σ = 0.016 × scale factor | Irreducible process variation, growing with batch size |

Electrolyte is re-optimised rather than copied: $\text{salt}_s = \text{salt}_{lab}(L_s/L_{lab})^{0.18}$ and $\text{soda}_s = \text{soda}_{lab}(L_s/L_{lab})^{0.10}$. Concentration in g/L falls modestly as the liquor shortens, while *total* consumption falls steeply because bath volume collapses.

Shades are apportioned into up to four trichromatic primaries using each family's nominal mix, perturbed by a Dirichlet draw and converted to concentrations through relative colouristic strength, so a weaker dye requires a higher concentration for the same contribution.

### 3.3 The trial model, and a correction to the brief

Corrective trials are generated from a job difficulty score *d* ∈ [0,1] built from depth class (26%), dye count (22%), family criticality (20%), repeat status (18%), fabric weight (7%) and special finish (7%):

$$\text{attempts}_s = 1 + \min\left(\text{Poisson}\left(k\, w_s\,(A + B d)\right),\ 6\right)$$

**A unit inconsistency in the project brief had to be resolved before this could be calibrated.** The brief quotes "3.8 → 1.2 trials per shade" and, separately, "reduce trials from 4 to 1 at each stage". These cannot both describe the same quantity: 3.8 attempts per stage implies 19 dyeings per job, which contradicts the brief's own statement that shades need "3–5 iterative trials across the intermediate stages". The coherent reading — **3.8 corrective trials per shade, summed across the five stages** — was adopted and documented. The brief's "4 → 1" is reproduced as the attempt count at a difficult stage for a hard shade, which the right tail of this model still generates.

The parameters *k* and the bulk stage weight were solved numerically so the population reproduces both industrial baselines simultaneously: a mean of 3.8 corrective trials per shade and a 40% bulk RFT rate. Calibration converged to k = 0.5464 and w_bulk = 1.2261, giving 3.781 corrective trials and 40.2% RFT.

**This calibration forces bulk to be the hardest stage**, since a 40% RFT rate alone consumes 0.92 of the 3.8-trial budget. That is not an artefact of the fitting; it is the premise of the research restated quantitatively — a laboratory recipe does not translate to bulk.

### 3.4 Dataset and verification

The generator produced 8,000 jobs × 155 columns, exported both in analysis-ready form and in the mill's own 87-column job-card layout.

Twelve invariant checks were run, all passing: monotone liquor-ratio contraction; dye-count consistency; component/total agreement; adjustment factors inside the declared band; depth-window compliance; trial bounds; both calibration targets; the sign and strength of the depth–concentration relationship; monotone decline of the mean adjustment factor with scale; and two structural tests described in Section 4.1.

### 3.5 Feature contract and model formulation

Features are selected by an explicit **whitelist** and then audited. Any column belonging to a stage at or beyond the target stage, any adjustment factor, any cost or outcome column, and the latent difficulty score raise an error. This matters more than any hyper-parameter: a model permitted to see `bulk_adj_factor` reconstructs the target exactly.

The admitted feature set comprises 10 categorical and 19 numeric columns:

- **Substrate and shade (static):** shade family, depth class, fabric structure, composition, pre-treatment, special finish, dye system, customer, GSM, cellulose fraction, partition constant *K*, dye count, repeat flag.
- **Approved lab dip (the measurement carried forward):** four dye concentrations, total, salt, soda, temperature, time, liquor ratio, RPM, machine.
- **Planned conditions of the target stage (known before dyeing):** machine, batch weight, liquor ratio, RPM.

The target vector is $[c_1, c_2, c_3, c_4, \text{salt}, \text{soda}, \text{trials}]$.

Two architectures were compared. The **direct** track jumps the whole ladder in one model. The **sequential** track chains four models, each predicting the next stage from the previous stage's approved recipe; it is trained with teacher forcing on true upstream recipes but **evaluated by running the chain on its own predictions**, so the reported bulk error includes accumulated error propagation. This is the honest comparison, and Section 4.3 shows it is also the decisive one.

### 3.6 Target scaling and the selection metric

Two methodological decisions materially changed the results.

**Multi-output target scaling is mandatory, including for trees.** The target vector mixes dye concentrations (0.1–7 %owf), electrolyte (5–95 g/L) and a count (1–7). A multi-output decision tree selects splits on summed variance reduction across targets, so although trees are scale-invariant in their *features* they are scale-sensitive in their *targets*. Benchmarked without standardisation, Random Forest scored R² = 0.57 on the recipe targets; with it, 0.96. The uncorrected benchmark would have supported the conclusion "ensembles are unsuited to this problem", which is false.

**The selection metric must be the acceptance criterion.** Hyper-parameter search initially used R². Tuning under that objective *degraded* the metric that matters: CatBoost's dye-concentration MAPE rose from 4.02% untuned to 6.13% tuned, because R² rewards getting large concentrations right rather than keeping every concentration inside a relative tolerance band. Search and champion selection were therefore moved onto concentration-weighted relative recipe error,

$$\text{err} = \frac{1}{n}\sum_j \frac{\sum_i |\hat{c}_{ij} - c_{ij}|}{\sum_i c_{ij}}$$

the same quantity that governs acceptance and that drives the economic analysis in Section 5.

---

## 4. Results and Discussion

### 4.1 The dataset encodes the intended physics

Taking logarithms, the adjustment-factor model is linear in its mechanisms. Regressing log *f* on those terms must recover the generative exhaustion exponent, with the remaining mechanisms entering at unit coefficient:

| Quantity | Generative value | Recovered | Structural R² |
|---|---|---|---|
| Exhaustion exponent γ | 0.300 | **0.2905** | 0.9409 |
| Process noise σ | 0.0224 | **0.0222** | — |

This is a considerably sharper test than inspecting correlation signs. It confirms the data carry the intended physics with the intended elasticity, and that only the declared process noise remains unexplained.

The resulting ladder behaves as theory predicts:

| Stage | Mean liquor ratio | Mean adjustment factor | p5 – p95 | Mean total %owf |
|---|---|---|---|---|
| Lab Dip | 20.00 | 1.0000 | — | 1.5131 |
| 1 kg Sample | 17.52 | 1.0024 | 0.839 – 1.184 | 1.5275 |
| 7 kg Sample | 13.49 | 0.9733 | 0.814 – 1.148 | 1.4821 |
| Mini Bulk | 11.00 | 0.9495 | 0.792 – 1.121 | 1.4457 |
| Bulk | 8.00 | 0.9198 | 0.780 – 1.087 | 1.4003 |

The mean bulk recipe is 8.0% below the lab dip, within the 0.85–1.10 band reported in practice. The 1 kg stage sits marginally *above* parity because its liquor ratio barely contracts while shear and depth effects already act — a non-obvious feature the mechanism produces without being told to.

Depth modulates the contraction exactly as bath-saturation reasoning implies: at bulk the mean factor is 0.899 for Pastel/Light, 0.916 Medium, 0.934 Dark and 0.946 Extra Dark. Deep shades resist the correction.

Baseline operating performance: 1.756 attempts per stage, 3.781 corrective trials per shade, 40.2% bulk RFT, and only **3.2% of jobs clear all five stages first time**. Mean realised cost is USD 874.09 per programme, of which USD 402.14 — **46.0% of total spend** — is cost of poor quality.

### 4.2 Model benchmark

All models were trained on 6,400 jobs and evaluated on a held-out 1,600.

| Model | Track | Recipe error (%) | Off-tolerance (%) | R² (recipe) | MAE | RMSE | MAPE dyes (%) | R² dyes |
|---|---|---|---|---|---|---|---|---|
| **XGBoost** | direct | **2.66** | **31.1** | 0.9937 | 0.1284 | 0.1846 | 3.62 | 0.9921 |
| LightGBM | direct | 2.70 | 32.6 | 0.9933 | 0.1305 | 0.1847 | 3.53 | 0.9913 |
| CatBoost | direct | 3.11 | 39.8 | 0.9956 | 0.1325 | 0.1857 | 3.86 | 0.9948 |
| XGBoost (chained) | sequential | 3.89 | 57.1 | 0.9840 | 0.1719 | 0.2460 | 5.91 | 0.9786 |
| Lasso | direct | 10.42 | 79.7 | 0.9906 | 0.1448 | 0.1996 | 10.19 | 0.9874 |
| Ridge | direct | 12.62 | 74.7 | 0.9922 | 0.1638 | 0.2175 | 11.76 | 0.9902 |
| Linear (OLS) | direct | 12.88 | 71.6 | 0.9928 | 0.1438 | 0.1924 | 11.94 | 0.9907 |
| MLP (ANN) | direct | 14.63 | 86.4 | 0.9921 | 0.2352 | 0.3017 | 16.42 | 0.9923 |
| Random Forest | direct | 17.62 | 96.5 | 0.9619 | 0.4654 | 0.6334 | 19.62 | 0.9614 |
| Decision Tree | direct | 27.45 | 99.1 | 0.9248 | 0.5971 | 0.8224 | 33.01 | 0.9180 |

**Champion: tuned XGBoost (direct)** — learning rate 0.06, max depth 6, 500 estimators, subsample 1.0.

Three observations deserve emphasis.

**R² is a misleading headline here.** Every model except the two unregularised trees exceeds R² = 0.96 on the recipe targets, yet their off-tolerance rates range from 31% to 97%. Linear regression attains R² = 0.9928 — nominally excellent — while placing 71.6% of batches outside the colour tolerance. R² is dominated by the enormous between-shade variance in concentration (pastels at 0.05 %owf, extra-darks at 7 %owf); predicting "roughly the right depth" earns almost all of it. The decision-relevant question is the *relative* error on each individual recipe, and on that the gradient-boosting families are four to five times better than the linear ones.

**Gradient boosting dominates, and the margin is structural.** The relationship involves interaction terms by construction — the exhaustion factor depends jointly on *K* and liquor ratio, substrate affinity multiplies the depth effect — which additive linear models cannot represent and which boosted ensembles capture directly. The MLP underperforms, most plausibly because 6,400 rows of heavily categorical tabular data is a regime where boosted trees are usually superior.

**The direct architecture beats the chain.** Chained XGBoost achieves 3.89% recipe error against 2.66% direct, and nearly doubles the off-tolerance rate (57.1% vs 31.1%), despite each individual link scoring R² ≈ 0.84 under teacher forcing. The chain compounds its own error at every step: by bulk it is predicting from three previous predictions. For a tool that issues production recipes, each stage should be predicted from the one quantity that is actually measured and approved — the lab dip.

### 4.3 Feature attribution

Permutation importance on held-out data (drop in R² when a feature is shuffled):

| Feature | Importance | Feature | Importance |
|---|---|---|---|
| labdip_dye2_owf | 0.267 | labdip_total_owf | 0.0147 |
| labdip_dye4_owf | 0.237 | depth_class | 0.0072 |
| labdip_dye3_owf | 0.236 | is_repeat | 0.0052 |
| labdip_dye1_owf | 0.224 | n_dyes | 0.0044 |
| labdip_soda_g_l | 0.217 | cellulose_fraction | 0.0039 |
| labdip_salt_g_l | 0.215 | composition | 0.0038 |

The six lab-dip chemistry variables carry essentially all the signal, which is the correct and expected structure: the bulk recipe is a *perturbation* of the lab dip, and the model's job is to size that perturbation. Substrate and shade descriptors matter an order of magnitude less individually, but they are what determines the perturbation's size — they modulate rather than set the answer. SHAP attribution for the primary dye concentration agrees.

This is also a practical finding for deployment: the platform depends on lab-dip chemistry being recorded accurately and completely. A mill with incomplete salt and soda records on its lab dips would lose a substantial share of the achievable accuracy.

### 4.4 The trial count is largely irreducible, and the analysis says so

All models score near zero on `bulk_trials` (champion: R² = −0.047). Reported bare, this reads as failure. It is not.

The count is generated as 1 + min(Poisson(λ), 6), so only the variance of the conditional mean is reducible; the rest is shot noise no model can predict. Estimating the ceiling two ways:

| Estimator | Attainable R² |
|---|---|
| Binned (no distributional assumption) | **0.0537** |
| Moment-based (biased low by the cap) | 0.0354 |
| **Achieved by champion** | −0.0466 |

The attainable ceiling is about 5%. The champion does not reach even that — the multi-output model trades trial accuracy for recipe accuracy, which is the correct trade given the selection metric. The honest conclusion is that **an individual batch's trial count is not predictable, and should not be presented as a platform output.** What the platform can shift is the *rate*, which Section 5 derives from recipe accuracy through an explicit mechanism rather than from this regression target. Quoting the achieved R² without the attainable bound would have misreported an irreducible property of the process as a modelling deficiency.

### 4.5 Per-stage production models

For the deployed tool, four independent models map the lab dip onto each stage:

| Stage | R² (recipe targets) | MAPE on dyes (%) | MAE (%owf) |
|---|---|---|---|
| 1 kg Sample | 0.9955 | 2.84 | 0.0081 |
| 7 kg Sample | 0.9953 | 2.97 | 0.0085 |
| Mini Bulk | 0.9952 | 3.12 | 0.0088 |
| Bulk | 0.9947 | 3.54 | 0.0095 |

Error grows monotonically with distance from the lab dip, as expected: the further the extrapolation, the larger the liquor-ratio contraction and the more process noise has accumulated.

### 4.6 Prototype

The Streamlit prototype (`app.py`) takes the approved lab dip, substrate and production plan, and returns the recipe for all four downstream stages, the trajectory of the correction, the expected trial count, a confidence read-out and a full cost breakdown, with a downloadable recipe card. Confidence is reported as the model's held-out error for that stage expressed against the colour tolerance, so the figure is traceable to a measurement rather than being a synthetic score.

---

## 5. Techno-Economic and Environmental Impact Analysis

### 5.1 Deriving the gain rather than assuming it

Assuming RFT rises to 85% and computing the consequences would be circular. Instead, the probability of landing outside the colour-acceptance tolerance was measured on held-out jobs under three regimes. The mill's standing bulk factor was estimated on *training* jobs only, so the status quo receives the same out-of-sample discipline as the model.

| Regime | Median error (%) | p90 error (%) | Outside tolerance (%) |
|---|---|---|---|
| Naive — lab dip carried over | 10.49 | 25.77 | 85.6 |
| Rule of thumb — standing factor 0.9201 | 7.42 | 17.87 | 77.8 |
| **AI platform — champion model** | **2.24** | **5.02** | **31.1** |

The rework-rate multiplier follows directly:

$$\rho = \frac{P(\text{off-tolerance} \mid \text{AI})}{P(\text{off-tolerance} \mid \text{rule of thumb})} = \frac{0.3113}{0.7781} = 0.400$$

Each job's baseline rework rate λ_j was estimated per stage by a cross-fitted Poisson-loss gradient booster on observable job features — never the generator's internal difficulty score, which a real mill would not possess — and the AI scenario applies ρ to it. The structure guarantees the right failure mode: were the model poor, ρ would approach 1 and the savings would vanish.

### 5.2 Operational impact

| Metric | Baseline | AI platform | Programme target | Verdict |
|---|---|---|---|---|
| Corrective trials per shade | 3.78 | **0.80** | ≤ 1.2 | **Met** |
| Bulk right-first-time | 40.2% | **70.2%** | ≥ 85% | **Not met** |
| Cost per programme | $874.09 | $564.43 | — | −35.4% |

**The RFT target is not met, and the gap is quantified rather than glossed.** Because RFT is monotone in ρ, the required value can be solved for directly and translated back into a specification:

| Quantity | Achieved | Required for 85% RFT |
|---|---|---|
| Rework multiplier ρ | 0.400 | 0.181 |
| Off-tolerance rate | 31.1% | 14.1% |
| Equivalent tolerance band | 3.00% | 4.53% |

Reaching 85% requires either roughly halving the model's recipe error, or establishing that the mill's true colour-acceptance tolerance is nearer 4.5% than 3.0%. The second is an empirical question the mill can settle from its own ΔE acceptance records, and it costs nothing to check — which makes it the first thing a follow-up study should do.

The apparent tension between an over-achieved trial target and an under-achieved RFT target is arithmetic, not inconsistency. Total corrective trials are a mean over five stages and fall fast under a uniform rate reduction; RFT is a tail probability at the single hardest stage, and tail probabilities move more slowly.

### 5.3 Financial appraisal

Scaled to 420 scale-up programmes per year against 100,000 kg of annual output:

| Item | USD | LKR |
|---|---|---|
| Gross annual saving | 130,059 | 39,667,911 |
| Platform operating cost | (8,400) | (2,562,000) |
| **Net annual saving** | **121,659** | **37,105,911** |
| Capital outlay | 46,000 | 14,030,000 |

| Indicator | Value |
|---|---|
| Payback period | 4.5 months |
| NPV (5 years @ 12%) | USD 392,552 |
| IRR | 264% |
| 5-year ROI | 1,222% |

These returns are very high and should be read as indicative. They rest on an assumed capital outlay of USD 46,000 covering platform build, spectrophotometer integration and MES hooks; a mill requiring new colour-measurement hardware, laboratory information-system work or significant change management could face several times that figure without altering the savings, and the payback would scale accordingly. The saving itself is the more robust number, because it derives from measured model error; the return multiple depends on a capex assumption that this study does not validate. The analysis also assumes freed machine capacity is used — if the dye-house is not capacity-constrained, the machine-time component converts to deferred cost rather than realised revenue.

### 5.4 Environmental impact

| Resource | Annual reduction |
|---|---|
| Dyestuff | 1,507 kg |
| Chemicals (salt, soda) | 40,663 kg |
| Water | 3,760 m³ |
| Thermal energy | 107.9 MWh |
| CO₂e | 23.1 t |
| Machine time released | 2,802 h |

Electrolyte dominates by mass, which is characteristic of reactive dyeing: Glauber's salt is dosed at 10–80 g/L over bath volumes that reach thousands of litres at bulk. Salt is also the hardest component of dye-house effluent to treat, since it passes through conventional biological treatment unchanged. The 40.7 tonnes of avoided electrolyte is therefore arguably a more consequential environmental result than the CO₂e figure, even though it attracts less attention.

---

## 6. Conclusions, Limitations and Future Scope

### 6.1 Conclusions

1. The lab-dip-to-bulk recipe correction is **learnable to within the colour-acceptance tolerance**. A tuned XGBoost model predicts bulk recipes from the lab dip at 2.66% concentration-weighted relative error, against 7.42% for the mill's standing-factor rule of thumb and 10.49% for carrying the lab dip over unchanged.

2. **Predict each stage directly from the lab dip, not sequentially.** The chained architecture compounds error at every link and nearly doubles the off-tolerance rate despite each link being individually sound.

3. **R² is the wrong headline metric for this problem.** Models spanning a threefold range in rework exposure all score R² > 0.96. Selection, tuning and reporting should use the relative error that governs acceptance; optimising R² measurably degraded it.

4. **Multi-output target scaling is mandatory, including for tree ensembles.** Omitting it understated Random Forest by 0.40 R² and would have supported a false methodological conclusion.

5. **Trial counts are near-irreducible at the individual-batch level** (attainable R² ≈ 0.054). The platform should forecast rework *rates*, derived from recipe accuracy, and not present per-batch trial predictions.

6. **The economic case is strong but the RFT target is not met.** Corrective trials fall from 3.78 to 0.80 per shade and cost by 35.4%, worth USD 130,059 annually at this factory scale. Bulk RFT reaches 70.2% against an 85% target; closing that gap requires halving the model's recipe error or confirming a wider true acceptance tolerance.

### 6.2 Limitations

- **The dataset is synthetic.** Every result characterises the *method*, not a deployed system. The generator is physics-grounded and verified against its own specification, but it cannot contain surprises that real dye-houses contain — substrate lot variation, dye batch strength variation, water-hardness excursions, operator effects. Real job-card history is required before any deployment claim.
- **No spectrophotometric data.** Acceptance is approximated by a relative concentration tolerance rather than computed as ΔE from predicted reflectance. The 3% figure is a reasoned assumption, and Section 5.2 shows the headline RFT conclusion is sensitive to it.
- **Costs are indicative.** Prices are 2024–25 South-Asian mill estimates; the capex underpinning the return multiples is assumed, not quoted.
- **The trial model is a single mechanism.** It attributes rework entirely to recipe deviation. Real reruns also arise from fabric faults, machine breakdowns and customer standard changes, none of which an improved recipe prevents — so the realised trial reduction in practice would be smaller than derived here.
- **Stationarity is assumed.** Dye suppliers, water chemistry and machine fleets change; the models would require scheduled retraining.

### 6.3 Future scope

1. **Re-fit on real job-card history.** The single highest-value next step. The schema is already aligned to the mill's 87-column workbook, so ingestion is a data-collection exercise rather than a redesign.
2. **Settle the acceptance tolerance empirically.** The mill's ΔE records can establish whether the true tolerance is nearer 3% or 4.5%; this alone determines whether the 85% RFT target is already met.
3. **Predict in K/S space.** Coupling reflectance measurement to a Kubelka–Munk formulation layer would let the platform output ΔE directly and remove the tolerance proxy.
4. **Quantify prediction uncertainty.** Quantile regression or conformal prediction would let the platform issue a recipe *with an interval*, so a dyer knows when to trust it — more valuable operationally than a marginal accuracy gain.
5. **Close the loop.** Feeding realised bulk recipes back as training data would let the platform adapt to supplier and seasonal drift automatically.
6. **Extend beyond reactive dyeing on cellulose** to disperse dyeing of polyester and to the thermal and salt-free processes now displacing conventional exhaust dyeing.

---

## References

[1] A. D. Broadbent, *Basic Principles of Textile Coloration*. Bradford, U.K.: Society of Dyers and Colourists, 2001.

[2] J. Shore, Ed., *Cellulosics Dyeing*. Bradford, U.K.: Society of Dyers and Colourists, 1995.

[3] P. Kubelka and F. Munk, "Ein Beitrag zur Optik der Farbanstriche," *Zeitschrift für technische Physik*, vol. 12, pp. 593–601, 1931.

[4] R. McDonald, Ed., *Colour Physics for Industry*, 2nd ed. Bradford, U.K.: Society of Dyers and Colourists, 1997.

[5] A. K. R. Choudhury, *Principles of Colour and Appearance Measurement*. Cambridge, U.K.: Woodhead Publishing, 2014.

[6] T. Chen and C. Guestrin, "XGBoost: A scalable tree boosting system," in *Proc. 22nd ACM SIGKDD Int. Conf. Knowledge Discovery and Data Mining*, 2016, pp. 785–794.

[7] G. Ke et al., "LightGBM: A highly efficient gradient boosting decision tree," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 3146–3154.

[8] L. Prokhorenkova, G. Gusev, A. Vorobev, A. V. Dorogush, and A. Gulin, "CatBoost: Unbiased boosting with categorical features," in *Advances in Neural Information Processing Systems 31*, 2018, pp. 6638–6648.

[9] S. M. Lundberg and S.-I. Lee, "A unified approach to interpreting model predictions," in *Advances in Neural Information Processing Systems 30*, 2017, pp. 4765–4774.

[10] L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.

[11] P. Chapman et al., *CRISP-DM 1.0: Step-by-Step Data Mining Guide*. SPSS Inc., 2000.

[12] F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *Journal of Machine Learning Research*, vol. 12, pp. 2825–2830, 2011.

[13] V. A. Shamey and T. Shamey, "Colour recipe prediction using artificial neural networks," *Coloration Technology*, vol. 121, no. 4, pp. 185–191, 2005.

[14] M. Senthilkumar, "Modelling of CIELAB values in vinylsulphone dye application using feed-forward neural networks," *Dyes and Pigments*, vol. 75, no. 2, pp. 356–361, 2007.

[15] A. Hasanbeigi and L. Price, "A technical review of emerging technologies for energy and water efficiency and pollution reduction in the textile industry," *Journal of Cleaner Production*, vol. 95, pp. 30–44, 2015.

[16] C. R. Holkar, A. J. Jadhav, D. V. Pinjari, N. M. Mahamuni, and A. B. Pandit, "A critical review on textile wastewater treatments: Possible approaches," *Journal of Environmental Management*, vol. 182, pp. 351–366, 2016.

[17] L. Grinsztajn, E. Oyallon, and G. Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?," in *Advances in Neural Information Processing Systems 35*, 2022, pp. 507–520.

[18] S. M. Burkinshaw, *Physico-chemical Aspects of Textile Coloration*. Chichester, U.K.: Wiley, 2016.

---

## Appendix A — Reproducibility

```bash
pip install -r requirements.txt
python run_all.py          # full study (add --fast to skip the search)
streamlit run app.py       # prototype
```

Seed 20251001 throughout. Environment: Windows 10, Python 3.13, numpy 2.5, pandas 3.0, scikit-learn 1.9, XGBoost 3.3, LightGBM 4.7, CatBoost 1.2.10, SHAP 0.52.

## Appendix B — Figure index

| Figure | Content |
|---|---|
| fig01 | Recipe adjustment across the five stages, and by depth class |
| fig02 | Correlation structure of the scale-up problem |
| fig03 | Trial distributions, baseline vs design target |
| fig04 | Cost of poor quality by shade family and depth class |
| fig05 | Exhaustion physics: measured and analytic |
| fig06 | Concentration windows and electrolyte demand by depth |
| fig07 | Model benchmark: recipe error and rework exposure |
| fig08 | Permutation and SHAP feature attribution |
| fig09 | Predicted vs actual for the champion model |
| fig10 | Recipe error by regime, and off-tolerance rates |
| fig11 | Trials avoided, savings by depth, investment appraisal |

## Appendix C — Output tables

All numbers quoted in this paper are reproduced in `outputs/tables/`:
`model_benchmark_summary.csv`, `model_benchmark_direct_per_target.csv`,
`model_benchmark_sequential_links.csv`, `cba_regime_accuracy.csv`,
`cba_factory_summary.csv`, `cba_target_gap_analysis.csv`,
`trial_count_predictability.csv`, `stage_model_accuracy.csv`,
`feature_importance_permutation.csv`, `dataset_summary_statistics.csv`.
