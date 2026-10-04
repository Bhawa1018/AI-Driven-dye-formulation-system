"""
data_generator.py  --  STEP 1 of the research pipeline
======================================================
Physics-grounded synthetic data generator for multi-stage textile dye-house
scale-up.

Why synthetic data
------------------
Dye-house recipe archives are commercially sensitive and, where they exist, are
stored as unstructured job cards (see the 87-column workbook supplied with this
project, which holds two illustrative jobs). To develop and benchmark the
modelling pipeline we therefore generate a statistically sound surrogate
population whose *generative mechanism is stated explicitly* and is grounded in
dyeing physics, rather than drawing numbers from arbitrary distributions.

The generative model
--------------------
For every job the lab dip is sampled first, then the recipe required at each
larger stage to reproduce the SAME visual depth is derived from a net
adjustment factor

    f_s = f_LR,s  x  f_shear,s  x  f_depth  x  f_substrate  x  f_machine,s  x  eps

  f_LR     exhaustion / partition term. For a dye of partition constant K the
           equilibrium exhaustion at liquor ratio L is  E = K / (K + L).
           Holding fixed depth requires  c_s / c_lab = E_lab / E_s
                                                     = (K + L_s) / (K + L_lab).
           Because the dyer simultaneously re-optimises the electrolyte, which
           partly cancels the exhaustion gain, the term is damped by an
           exponent gamma in (0, 1):
                f_LR = ((K + L_s) / (K + L_lab)) ** gamma
  f_shear  bulk machines run at higher agitation; more hydrolysed dye is
           stripped at wash-off, so slightly more dye must be applied.
  f_depth  in deep shades the bath nears saturation, dye-dye aggregation rises
           and marginal exhaustion falls, so deep shades need more compensation.
  f_subst  substrate accessibility (composition, pre-treatment) and a diffusion
           penalty for heavy constructions.
  f_mach   machine-family hydrodynamic bias.
  eps      lognormal irreducible process noise.

Electrolyte is re-optimised per stage rather than copied:
    salt_s = salt_lab * (L_s / L_lab) ** 0.18
    soda_s = soda_lab * (L_s / L_lab) ** 0.10
i.e. the concentration in g/L falls modestly as the liquor shortens (short
liquor is intrinsically more exhausting), while the *total* salt consumed falls
steeply because the bath volume collapses.

Trials are generated from a job difficulty score d in [0, 1]:
    attempts_s = 1 + min(Poisson(k * w_s * (A + B * d)), TRIAL_MAX - 1)
with (k, w_bulk) solved numerically so that the synthetic population reproduces
the two industrial baselines declared in config: a mean of 3.8 attempts per
stage and a 40% bulk right-first-time rate.

Outputs
-------
  textile_dyeing_scaleup_master.csv              analysis-ready, one row per job
  data/textile_dyeing_scaleup_factory_format.csv same data in the mill's own
                                                 87-column job-card layout
  outputs/tables/dataset_summary_statistics.csv
  outputs/tables/dataset_correlation_matrix.csv
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd

try:
    from . import config as C
    from .shade_master import build_shade_master
except ImportError:
    import config as C
    from shade_master import build_shade_master

warnings.filterwarnings("ignore", category=FutureWarning)

CUSTOMERS = ["GAP", "H&M", "Next", "Decathlon", "Tesco", "Primark", "M&S",
             "Uniqlo", "C&A", "Zara"]
CUSTOMER_P = [0.16, 0.14, 0.11, 0.10, 0.10, 0.09, 0.08, 0.08, 0.07, 0.07]


# ===========================================================================
# 1. Helper pieces of the generative model
# ===========================================================================
def _dye_trade_name(dye_class: str, primary: str, rng: np.random.Generator) -> str:
    """Build a plausible commercial dye name, e.g. 'REMAZOL BLUE RGB 3R'."""
    prefix = C.DYE_CLASSES[dye_class]["trade_prefix"]
    suffix = rng.choice(["RGB", "ED", "RR", "BB", "3R", "4GL", "HE-B", "CT"])
    return f"{prefix} {primary.upper()} {suffix}"


def _split_into_primaries(family: str, n_dyes: int, total_owf: float,
                          rng: np.random.Generator):
    """
    Apportion a total %owf across up to 4 trichromatic primaries.

    The nominal family mix is truncated to n_dyes components, perturbed with a
    Dirichlet draw (so tonal balance varies realistically between jobs), then
    converted to concentrations using relative colouristic strength: a weaker
    dye needs a higher concentration for the same contribution.
    """
    mix = C.FAMILY_PRIMARY_MIX.get(family, [("Black", 1.0)])
    mix = mix[:n_dyes]
    names = [m[0] for m in mix]
    weights = np.array([m[1] for m in mix], dtype=float)
    weights = weights / weights.sum()

    # Dirichlet concentration 60 => tight but non-degenerate tonal variation.
    weights = rng.dirichlet(weights * 60.0)

    strengths = np.array([C.DYE_PRIMARIES[n]["strength"] for n in names])
    raw = weights / strengths
    conc = total_owf * raw / raw.sum()

    out_names = names + [None] * (4 - len(names))
    out_conc = list(conc) + [0.0] * (4 - len(conc))
    return out_names, out_conc


def _difficulty(depth_class: str, n_dyes: int, family: str, is_repeat: bool,
                gsm: float, finish: str) -> float:
    """Job difficulty score in [0, 1] driving the trial-count distribution."""
    w = C.DIFFICULTY_WEIGHTS
    depth_s = {"Pastel/Light": 0.10, "Medium": 0.40,
               "Dark": 0.75, "Extra Dark": 1.00}[depth_class]
    n_s = (n_dyes - 1) / 3.0
    fam_s = C.FAMILY_CRITICALITY.get(family, 0.60)
    rep_s = 0.0 if is_repeat else 1.0
    gsm_s = float(np.clip((gsm - 120.0) / 200.0, 0.0, 1.0))
    fin_s = 0.0 if finish == "None" else 0.75

    d = (w["depth"] * depth_s + w["n_dyes"] * n_s + w["family"] * fam_s
         + w["repeat"] * rep_s + w["gsm"] * gsm_s + w["finish"] * fin_s)
    return float(np.clip(d, 0.0, 1.0))


def _adjustment_factor(stage: str, lr_stage: float, lr_lab: float, K: float,
                       rpm_stage: float, rpm_lab: float, depth_class: str,
                       composition: str, pretreatment: str, gsm: float,
                       machine: str, rng: np.random.Generator) -> float:
    """Net lab-dip -> stage recipe correction factor (see module docstring)."""
    if stage == "labdip":
        return 1.0

    # (a) exhaustion / partition term, damped by electrolyte re-optimisation
    f_lr = ((K + lr_stage) / (K + lr_lab)) ** C.GAMMA_LR_DAMPING

    # (b) mechanical shear / wash-off loss
    f_shear = 1.0 + C.C_SHEAR * (rpm_stage - rpm_lab)

    # (c) depth-dependent saturation / aggregation compensation
    f_depth = 1.0 + C.DEPTH_COMPENSATION[depth_class]

    # (d) substrate accessibility and diffusion penalty for heavy fabric
    affinity = (C.COMPOSITION_AFFINITY[composition]
                * C.PRETREATMENT_AFFINITY[pretreatment])
    f_subst = (1.0 / affinity) * (1.0 + C.C_GSM * (gsm - 150.0) / 100.0)

    # (e) machine-family hydrodynamic bias
    f_mach = C.MACHINE_BIAS.get(machine, 1.0)

    # (f) irreducible process noise, growing slightly with scale
    scale_noise = {"s1kg": 0.8, "s7kg": 1.0, "minibulk": 1.2, "bulk": 1.4}[stage]
    eps = float(rng.lognormal(0.0, C.SIGMA_PROCESS * scale_noise))

    f = f_lr * f_shear * f_depth * f_subst * f_mach * eps
    return float(np.clip(f, *C.ADJ_FACTOR_CLAMP))


# ===========================================================================
# 2. Trial-model calibration
# ===========================================================================
def calibrate_trial_model(difficulties: np.ndarray, seed: int,
                          n_iter: int = 60) -> tuple:
    """
    Solve for (k, w_bulk) so the synthetic population reproduces both
    industrial baselines in config: BASELINE_CORRECTIVE_TRIALS_PER_JOB (extra
    attempts beyond the first, summed over the five stages) and
    BASELINE_RFT_BULK (P(attempts_bulk == 1)).

    Uses a damped fixed-point iteration on the two analytic moment conditions

        E[corrective per job] = k * sum_s(w_s) * mean_j(A + B d_j)
        P(bulk == 1)          = mean_j exp(-k * w_bulk * (A + B d_j))

    which is stable because each target is monotone in its own parameter.
    """
    A, B = C.TRIAL_DIFFICULTY_A, C.TRIAL_DIFFICULTY_B
    lam_shape = A + B * difficulties                      # per-job, per unit k*w
    fixed_w = {s: C.STAGE_TRIAL_WEIGHT[s] for s in C.STAGES if s != "bulk"}

    k = 2.0
    w_bulk = C.STAGE_TRIAL_WEIGHT["bulk"]
    target_corrective = C.BASELINE_CORRECTIVE_TRIALS_PER_JOB   # per job, all stages
    target_rft = C.BASELINE_RFT_BULK

    for _ in range(n_iter):
        # (i) w_bulk so that bulk RFT matches, given current k
        lo, hi = 1e-4, 20.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            rft = float(np.mean(np.exp(-k * mid * lam_shape)))
            if rft > target_rft:      # too easy -> need a larger weight
                lo = mid
            else:
                hi = mid
        w_bulk = 0.5 * (lo + hi)

        # (ii) k so that the per-job corrective total matches, given w_bulk
        w_sum = sum(fixed_w.values()) + w_bulk
        k_new = target_corrective / (w_sum * float(np.mean(lam_shape)))
        k = 0.5 * k + 0.5 * k_new                         # damped update

    # The analytic total-corrective condition ignores the cap at TRIAL_MAX, which truncates
    # the upper tail and pulls the realised mean down. Refine k by Monte Carlo
    # against the capped sampler actually used by the generator. The RFT
    # condition needs no correction: P(attempts == 1) = P(Poisson == 0) exactly,
    # which the cap leaves untouched.
    mc = np.random.default_rng(seed + 991)
    for _ in range(8):
        draws = []
        for s in C.STAGES:
            w_s = w_bulk if s == "bulk" else fixed_w[s]
            lam = k * w_s * lam_shape
            draws.append(np.minimum(mc.poisson(lam), C.TRIAL_MAX - 1))
        realised = float(np.sum([d.mean() for d in draws]))
        if abs(realised - C.BASELINE_CORRECTIVE_TRIALS_PER_JOB) < 0.01:
            break
        k *= C.BASELINE_CORRECTIVE_TRIALS_PER_JOB / max(realised, 1e-6)
        # re-solve w_bulk for the updated k so RFT stays on target
        lo, hi = 1e-4, 20.0
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if float(np.mean(np.exp(-k * mid * lam_shape))) > target_rft:
                lo = mid
            else:
                hi = mid
        w_bulk = 0.5 * (lo + hi)

    weights = dict(fixed_w)
    weights["bulk"] = w_bulk
    return k, weights


# ===========================================================================
# 3. Main generator
# ===========================================================================
def generate_dataset(n_records: int = C.N_RECORDS, seed: int = C.RANDOM_SEED,
                     verbose: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shades = build_shade_master()

    n_shades = len(shades)
    shade_idx = rng.integers(0, n_shades, size=n_records)

    # ---- static / job-level attributes -----------------------------------
    structures = rng.choice(C.FABRIC_STRUCTURES, size=n_records,
                            p=C.FABRIC_STRUCTURE_P)
    gsm = np.array([
        rng.integers(*C.STRUCTURE_GSM_RANGE[s], endpoint=True) for s in structures
    ], dtype=float)
    compositions = rng.choice(C.COMPOSITIONS, size=n_records, p=C.COMPOSITION_P)
    pretreatments = rng.choice(C.PRETREATMENTS, size=n_records, p=C.PRETREATMENT_P)
    finishes = rng.choice(C.SPECIAL_FINISHES, size=n_records, p=C.SPECIAL_FINISH_P)
    customers = rng.choice(CUSTOMERS, size=n_records, p=CUSTOMER_P)
    is_repeat = rng.random(n_records) < C.P_REPEAT_SHADE

    dye_class_names = list(C.DYE_CLASSES.keys())
    dye_class_p = [C.DYE_CLASSES[k]["p"] for k in dye_class_names]
    dye_classes = rng.choice(dye_class_names, size=n_records, p=dye_class_p)

    dates = (pd.Timestamp("2023-01-02")
             + pd.to_timedelta(rng.integers(0, 1000, size=n_records), unit="D"))

    # ---- difficulty, then calibrate the trial model -----------------------
    fam = shades["shade_family"].to_numpy()[shade_idx]
    dep = shades["depth_class"].to_numpy()[shade_idx]
    n_typ = shades["n_dyes_typical"].to_numpy()[shade_idx]

    n_dyes = np.clip(n_typ + rng.integers(-1, 1, endpoint=True, size=n_records),
                     1, 4)
    # A shade can only be built from the primaries its family recipe defines, so
    # cap the dye count at the number of primaries available for that family.
    fam_n_primaries = np.array([
        len(C.FAMILY_PRIMARY_MIX.get(f, [("Black", 1.0)])) for f in fam
    ])
    n_dyes = np.minimum(n_dyes, fam_n_primaries)

    difficulty = np.array([
        _difficulty(dep[i], int(n_dyes[i]), fam[i], bool(is_repeat[i]),
                    gsm[i], finishes[i])
        for i in range(n_records)
    ])

    k_cal, trial_w = calibrate_trial_model(difficulty, seed)
    if verbose:
        print(f"[generator] trial model calibrated: k={k_cal:.4f}, "
              f"w_bulk={trial_w['bulk']:.4f}")

    # ---- build records ----------------------------------------------------
    records = []
    for i in range(n_records):
        srow = shades.iloc[int(shade_idx[i])]
        family = srow["shade_family"]
        depth_class = srow["depth_class"]
        dcls = dye_classes[i]
        K = C.DYE_CLASSES[dcls]["K"]
        nd = int(n_dyes[i])

        rec = {
            "job_id": f"JOB-{i + 1:06d}",
            "date": dates[i].date().isoformat(),
            "shade_no": int(srow["shade_no"]),
            "shade_code": srow["shade_code"],
            "shade_name": srow["shade_name"],
            "shade_family": family,
            "depth_label": srow["depth_label"],
            "depth_class": depth_class,
            "colour_direction": srow["colour_direction"],
            "customer": customers[i],
            "is_repeat": bool(is_repeat[i]),
            "fabric_structure": structures[i],
            "composition": compositions[i],
            "cellulose_fraction": C.COMPOSITION_CELLULOSE_FRACTION[compositions[i]],
            "gsm": float(gsm[i]),
            "pretreatment": pretreatments[i],
            "special_finish": finishes[i],
            "dye_class": dcls,
            "dye_partition_K": K,
            "n_dyes": nd,
            "difficulty_score": round(float(difficulty[i]), 4),
        }

        # ---- lab dip recipe ------------------------------------------------
        lo, hi = C.DEPTH_OWF_RANGE[depth_class]
        # Beta(2,2) keeps the draw inside the class window with a central mode.
        total_owf_lab = lo + (hi - lo) * float(rng.beta(2.0, 2.0))
        # Polyester blends: the recipe is expressed on total fabric weight, so a
        # lower cellulose fraction needs proportionally less reactive dye.
        total_owf_lab *= (0.55 + 0.45 * rec["cellulose_fraction"])

        names, concs = _split_into_primaries(family, nd, total_owf_lab, rng)
        for j in range(4):
            rec[f"dye{j + 1}_name"] = (
                _dye_trade_name(dcls, names[j], rng) if names[j] else ""
            )
            rec[f"dye{j + 1}_primary"] = names[j] if names[j] else ""

        lr_lab = C.STAGE_LIQUOR_RATIO["labdip"][0]
        rpm_lab = float(rng.integers(*C.STAGE_RPM["labdip"], endpoint=True))
        salt_lab = float(rng.uniform(*C.DEPTH_SALT_G_L[depth_class]))
        soda_lab = float(rng.uniform(*C.DEPTH_SODA_G_L[depth_class]))
        temp_lab = float(rng.uniform(*C.DYE_CLASSES[dcls]["temp_c"]))
        time_lab = float(rng.integers(*C.DYE_CLASSES[dcls]["time_min"], endpoint=True))

        # ---- walk the scale-up ladder --------------------------------------
        for stage in C.STAGES:
            if stage == "labdip":
                batch = C.STAGE_BATCH_KG["labdip"][0]
                lr = lr_lab
                rpm = rpm_lab
                machine = str(rng.choice(C.STAGE_MACHINES["labdip"]))
                f = 1.0
                salt, soda = salt_lab, soda_lab
                temp, dtime = temp_lab, time_lab
            else:
                b_lo, b_hi = C.STAGE_BATCH_KG[stage]
                batch = (b_lo if b_lo == b_hi
                         else float(np.round(rng.uniform(b_lo, b_hi), 1)))
                lr = float(np.round(rng.uniform(*C.STAGE_LIQUOR_RATIO[stage]), 2))
                rpm = float(rng.integers(*C.STAGE_RPM[stage], endpoint=True))
                machine = str(rng.choice(C.STAGE_MACHINES[stage]))
                f = _adjustment_factor(stage, lr, lr_lab, K, rpm, rpm_lab,
                                       depth_class, compositions[i],
                                       pretreatments[i], gsm[i], machine, rng)
                # electrolyte re-optimisation
                salt = salt_lab * (lr / lr_lab) ** 0.18 * float(rng.normal(1.0, 0.02))
                soda = soda_lab * (lr / lr_lab) ** 0.10 * float(rng.normal(1.0, 0.015))
                # thermal profile drifts slightly on larger machines
                temp = temp_lab + float(rng.normal(0.0, 1.1))
                # larger batches need longer to level out
                dtime = time_lab * (1.0 + 0.04 * np.log(batch / 0.01) / np.log(10)) \
                    * float(rng.normal(1.0, 0.03))

            total_owf = total_owf_lab * f
            stage_concs = [c * f for c in concs]

            rec[f"{stage}_machine"] = machine
            rec[f"{stage}_batch_kg"] = round(batch, 3)
            rec[f"{stage}_liquor_ratio"] = round(lr, 2)
            rec[f"{stage}_rpm"] = rpm
            rec[f"{stage}_adj_factor"] = round(f, 4)
            for j in range(4):
                rec[f"{stage}_dye{j + 1}_owf"] = round(stage_concs[j], 5)
            rec[f"{stage}_total_owf"] = round(total_owf, 5)
            rec[f"{stage}_salt_g_l"] = round(float(np.clip(salt, 5, 95)), 2)
            rec[f"{stage}_soda_g_l"] = round(float(np.clip(soda, 3, 30)), 2)
            rec[f"{stage}_temp_c"] = round(temp, 1)
            rec[f"{stage}_time_min"] = int(round(np.clip(dtime, 30, 120)))

            # ---- trials at this stage --------------------------------------
            lam = k_cal * trial_w[stage] * (C.TRIAL_DIFFICULTY_A
                                            + C.TRIAL_DIFFICULTY_B * difficulty[i])
            extra = int(min(rng.poisson(lam), C.TRIAL_MAX - 1))
            rec[f"{stage}_trials"] = 1 + extra

            # ---- derived physical quantities -------------------------------
            bath_l = batch * lr
            rec[f"{stage}_bath_volume_l"] = round(bath_l, 2)
            rec[f"{stage}_dye_mass_g"] = round(total_owf / 100.0 * batch * 1000.0, 3)
            rec[f"{stage}_salt_mass_kg"] = round(rec[f"{stage}_salt_g_l"] * bath_l / 1000.0, 4)
            rec[f"{stage}_soda_mass_kg"] = round(rec[f"{stage}_soda_g_l"] * bath_l / 1000.0, 4)

        rec["total_trials"] = sum(rec[f"{s}_trials"] for s in C.STAGES)
        rec["corrective_trials"] = rec["total_trials"] - len(C.STAGES)
        rec["rft_bulk"] = int(rec["bulk_trials"] == 1)
        rec["rft_all_stages"] = int(rec["total_trials"] == len(C.STAGES))
        rec["lab_to_bulk_adj_factor"] = rec["bulk_adj_factor"]
        rec["approval_status"] = "Approved"
        records.append(rec)

    df = pd.DataFrame(records)
    df = _attach_costs(df)
    return df


# ===========================================================================
# 4. Cost columns (used by EDA and by the techno-economic analysis)
# ===========================================================================
def attempt_cost(stage: str, row) -> float:
    """
    Fully-absorbed cost of ONE dyeing attempt at a given stage, in USD:
    dyes + electrolyte + auxiliaries + water + thermal energy + effluent
    + machine occupancy + fixed stage overhead.
    """
    U = C.UTILITIES
    batch = row[f"{stage}_batch_kg"]
    bath_l = row[f"{stage}_bath_volume_l"]

    # --- dyestuff ---------------------------------------------------------
    dye_usd = 0.0
    for j in range(1, 5):
        prim = row.get(f"dye{j}_primary", "")
        conc = row.get(f"{stage}_dye{j}_owf", 0.0) or 0.0
        if prim and conc > 0:
            kg = conc / 100.0 * batch
            dye_usd += kg * C.DYE_PRIMARIES[prim]["price_usd_kg"]

    # --- electrolyte + alkali ---------------------------------------------
    chem_usd = (row[f"{stage}_salt_mass_kg"] * C.CHEMICAL_PRICES_USD_KG["Glauber Salt"]
                + row[f"{stage}_soda_mass_kg"] * C.CHEMICAL_PRICES_USD_KG["Soda Ash"])

    # --- auxiliaries dosed per litre of bath ------------------------------
    for chem, g_l in C.AUX_DOSING_G_L.items():
        chem_usd += (g_l * bath_l / 1000.0) * C.CHEMICAL_PRICES_USD_KG[chem]

    # --- water: dye bath plus rinse / soaping / neutralisation baths -------
    total_water_l = bath_l * (1.0 + U["n_rinse_baths"] * U["rinse_volume_factor"])
    water_usd = total_water_l * U["water_usd_per_l"]

    # --- thermal energy: heat the dye bath, plus one hot soaping bath ------
    dT = max(row[f"{stage}_temp_c"] - U["ambient_temp_c"], 0.0)
    kwh = (bath_l * U["water_cp_kj_kg_k"] * dT / 3600.0) / U["boiler_efficiency"]
    kwh += (bath_l * U["water_cp_kj_kg_k"] * 50.0 / 3600.0) / U["boiler_efficiency"]
    energy_usd = kwh * U["steam_usd_per_kwh_thermal"]

    # --- effluent treatment ------------------------------------------------
    effluent_usd = (total_water_l / 1000.0) * U["effluent_usd_per_m3"]

    # --- machine occupancy + fixed overhead --------------------------------
    occupancy_hr = (row[f"{stage}_time_min"] + C.STAGE_HANDLING_MIN[stage]) / 60.0
    machine_usd = occupancy_hr * C.STAGE_MACHINE_USD_PER_HR[stage]

    return float(dye_usd + chem_usd + water_usd + energy_usd + effluent_usd
                 + machine_usd + C.STAGE_FIXED_OVERHEAD_USD[stage])


def _attach_costs(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-attempt, per-stage and total realised cost columns."""
    for stage in C.STAGES:
        per_attempt = df.apply(lambda r, s=stage: attempt_cost(s, r), axis=1)
        df[f"{stage}_cost_per_attempt_usd"] = per_attempt.round(4)
        df[f"{stage}_cost_actual_usd"] = (per_attempt * df[f"{stage}_trials"]).round(4)
        occ = (df[f"{stage}_time_min"] + C.STAGE_HANDLING_MIN[stage]) / 60.0
        df[f"{stage}_machine_hours_actual"] = (occ * df[f"{stage}_trials"]).round(3)
        water = (df[f"{stage}_bath_volume_l"]
                 * (1.0 + C.UTILITIES["n_rinse_baths"] * C.UTILITIES["rinse_volume_factor"]))
        df[f"{stage}_water_l_actual"] = (water * df[f"{stage}_trials"]).round(2)

    df["total_cost_actual_usd"] = df[[f"{s}_cost_actual_usd" for s in C.STAGES]].sum(axis=1).round(4)
    df["total_cost_ideal_usd"] = df[[f"{s}_cost_per_attempt_usd" for s in C.STAGES]].sum(axis=1).round(4)
    df["cost_of_poor_quality_usd"] = (df["total_cost_actual_usd"]
                                      - df["total_cost_ideal_usd"]).round(4)
    df["total_machine_hours_actual"] = df[[f"{s}_machine_hours_actual" for s in C.STAGES]].sum(axis=1).round(3)
    df["total_water_l_actual"] = df[[f"{s}_water_l_actual" for s in C.STAGES]].sum(axis=1).round(2)
    return df


# ===========================================================================
# 5. Export in the mill's own 87-column job-card layout
# ===========================================================================
FACTORY_STAGE_SUFFIX = {"labdip": "", "s1kg": ".1", "s7kg": ".2",
                        "minibulk": ".3", "bulk": ".4"}


def export_factory_format(df: pd.DataFrame, path: str = C.FACTORY_FORMAT_CSV) -> pd.DataFrame:
    """
    Re-express the master dataset in the column layout of the supplied mill
    job-card workbook, so the synthetic data can be dropped straight into the
    existing reporting sheets. Dye quantities are given as approved grams per
    batch (the unit the mill works in) rather than %owf.
    """
    out = pd.DataFrame({
        "job_id": df["job_id"],
        "date": df["date"],
        "shade_code": df["shade_name"],
        "shade_family": df["shade_family"],
        "depth_class": df["depth_class"],
        "customer": df["customer"],
        "is_repeat": np.where(df["is_repeat"], "YES", "NO"),
        "fabric_type": df["fabric_structure"].str.upper(),
        "composition": df["composition"],
        "gsm": df["gsm"].astype(int),
        "PRE TREATMENT": df["pretreatment"],
        "special_finish": df["special_finish"],
        "dye_name": df["dye_class"],
        "dye_class": df["dye_class"],
        "dye_hue": df["shade_family"].str.upper(),
        "n_dyes": df["n_dyes"],
    })
    for j in range(1, 5):
        out[f"NAME OF DYE {j}"] = df[f"dye{j}_name"]
    out["approval_status"] = df["approval_status"]
    out["process_remark"] = ""

    for stage in C.STAGES:
        sfx = FACTORY_STAGE_SUFFIX[stage]
        if stage == "labdip":
            out["batch_kg_labdip"] = df["labdip_batch_kg"]
            out["liquor_ratio_labdip"] = df["labdip_liquor_ratio"]
        else:
            out[f"machine_{stage}"] = df[f"{stage}_machine"]
            out[f"batch_kg_{stage}"] = df[f"{stage}_batch_kg"]
            out[f"liquor_ratio_{stage}"] = df[f"{stage}_liquor_ratio"]
        for j in range(1, 5):
            grams = df[f"{stage}_dye{j}_owf"] / 100.0 * df[f"{stage}_batch_kg"] * 1000.0
            out[f"APPROVED DYE (G) - DYE {j}{sfx}"] = grams.round(4).replace(0.0, np.nan)
        out[f"RPM{sfx}"] = df[f"{stage}_rpm"]
        out[f"salt_g_l_{stage}"] = df[f"{stage}_salt_g_l"]
        out[f"soda_ash_g_l_{stage}"] = df[f"{stage}_soda_g_l"]
        out[f"temperature_c_{stage}"] = df[f"{stage}_temp_c"]
        out[f"dyeing_time_min_{stage}"] = df[f"{stage}_time_min"]
        out[f"trials_{stage}"] = df[f"{stage}_trials"]

    out["machine_labdip"] = df["labdip_machine"]
    out.to_csv(path, index=False)
    return out


# ===========================================================================
# 6. Validation: summary statistics & correlation integrity
# ===========================================================================
def validate_dataset(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Verify that the generated population satisfies the physical and statistical
    invariants it is supposed to satisfy. Returns a dict of check -> bool.
    """
    checks = {}

    # (1) liquor ratio must contract monotonically along the ladder
    lrs = [df[f"{s}_liquor_ratio"] for s in C.STAGES]
    mono = np.ones(len(df), dtype=bool)
    for a, b in zip(lrs, lrs[1:]):
        mono &= (b <= a + 1e-9)
    checks["liquor_ratio_monotone_decreasing"] = bool(mono.all())

    # (2) dye concentrations are non-negative and respect n_dyes
    ok = True
    for stage in C.STAGES:
        cols = [f"{stage}_dye{j}_owf" for j in range(1, 5)]
        ok &= bool((df[cols] >= -1e-12).all().all())
        active = (df[cols] > 0).sum(axis=1)
        ok &= bool((active == df["n_dyes"]).all())
    checks["dye_concentrations_consistent_with_n_dyes"] = ok

    # (3) per-stage totals equal the sum of the components
    tot_ok = True
    for stage in C.STAGES:
        cols = [f"{stage}_dye{j}_owf" for j in range(1, 5)]
        tot_ok &= bool(np.allclose(df[cols].sum(axis=1), df[f"{stage}_total_owf"],
                                   atol=2e-4))
    checks["stage_totals_match_components"] = tot_ok

    # (4) adjustment factor inside the declared band
    adj = df[[f"{s}_adj_factor" for s in C.STAGES]]
    lo_c, hi_c = C.ADJ_FACTOR_CLAMP
    checks["adj_factor_within_clamp"] = bool(
        ((adj >= lo_c) & (adj <= hi_c)).all().all()
    )

    # (5) depth class windows respected at lab dip
    win_ok = True
    for dc, (lo, hi) in C.DEPTH_OWF_RANGE.items():
        sub = df.loc[df["depth_class"] == dc, "labdip_total_owf"]
        if len(sub):
            win_ok &= bool(sub.min() >= lo * 0.5 and sub.max() <= hi * 1.001)
    checks["labdip_depth_windows_respected"] = win_ok

    # (6) trials in [1, TRIAL_MAX]
    tcols = [f"{s}_trials" for s in C.STAGES]
    checks["trials_within_bounds"] = bool(
        df[tcols].ge(1).all().all() and df[tcols].le(C.TRIAL_MAX).all().all()
    )

    # (7) calibration targets reproduced
    mean_trials = float(df[tcols].to_numpy().mean())
    corrective = float(df["corrective_trials"].mean())
    rft = float(df["rft_bulk"].mean())
    checks["corrective_trials_on_target"] = abs(
        corrective - C.BASELINE_CORRECTIVE_TRIALS_PER_JOB) < 0.15
    checks["rft_bulk_on_target"] = abs(rft - C.BASELINE_RFT_BULK) < 0.03

    # (8) expected sign and strength of the key structural relationships.
    #
    # The raw correlation between bulk liquor ratio and the bulk adjustment
    # factor is diluted because the factor is a product of five independent
    # effects. The exhaustion mechanism is therefore tested as a PARTIAL
    # correlation, holding the other mechanisms (depth, substrate affinity,
    # construction, machine bias) fixed by residualising both variables on them.
    depth_ord = df["depth_class"].map(
        {"Pastel/Light": 0, "Medium": 1, "Dark": 2, "Extra Dark": 3}).to_numpy(float)
    controls = np.column_stack([
        np.ones(len(df)),
        depth_ord,
        df["composition"].map(C.COMPOSITION_AFFINITY).to_numpy(float),
        df["pretreatment"].map(C.PRETREATMENT_AFFINITY).to_numpy(float),
        df["gsm"].to_numpy(float),
        df["bulk_rpm"].to_numpy(float),
        df["bulk_machine"].map(C.MACHINE_BIAS).fillna(1.0).to_numpy(float),
        df["dye_partition_K"].to_numpy(float),
    ])

    def _resid(y):
        y = np.asarray(y, dtype=float)
        beta, *_ = np.linalg.lstsq(controls, y, rcond=None)
        return y - controls @ beta

    r_lr = _resid(df["bulk_liquor_ratio"])
    r_adj = _resid(df["bulk_adj_factor"])
    corr_lr_partial = float(np.corrcoef(r_lr, r_adj)[0, 1])
    corr_lr = float(np.corrcoef(df["bulk_liquor_ratio"], df["bulk_adj_factor"])[0, 1])
    corr_depth = float(np.corrcoef(depth_ord, df["bulk_total_owf"])[0, 1])
    checks["owf_rises_with_depth_class"] = corr_depth > 0.70

    # (8b) STRUCTURAL RECOVERY TEST.
    # Taking logs, the adjustment-factor model is linear in its mechanisms:
    #   log f = gamma*log((K+L_s)/(K+L_lab)) + log f_shear + log f_depth
    #           + log f_subst + log f_mach + log eps
    # Regressing log f on those terms must recover the generative exhaustion
    # exponent GAMMA_LR_DAMPING, with the remaining mechanisms entering at unit
    # coefficient. This is a far sharper test than a correlation sign: it
    # confirms the dataset encodes the intended physics with the intended
    # elasticity, and that only the declared process noise is left over.
    K_arr = df["dye_partition_K"].to_numpy(float)
    lr_lab = C.STAGE_LIQUOR_RATIO["labdip"][0]
    x1 = np.log((K_arr + df["bulk_liquor_ratio"].to_numpy(float)) / (K_arr + lr_lab))
    x2 = np.log(1.0 + C.C_SHEAR * (df["bulk_rpm"].to_numpy(float)
                                   - df["labdip_rpm"].to_numpy(float)))
    x3 = np.log(1.0 + df["depth_class"].map(C.DEPTH_COMPENSATION).to_numpy(float))
    affinity = (df["composition"].map(C.COMPOSITION_AFFINITY).to_numpy(float)
                * df["pretreatment"].map(C.PRETREATMENT_AFFINITY).to_numpy(float))
    x4 = np.log((1.0 / affinity)
                * (1.0 + C.C_GSM * (df["gsm"].to_numpy(float) - 150.0) / 100.0))
    x5 = np.log(df["bulk_machine"].map(C.MACHINE_BIAS).fillna(1.0).to_numpy(float))

    X = np.column_stack([np.ones(len(df)), x1, x2, x3, x4, x5])
    y = np.log(df["bulk_adj_factor"].to_numpy(float))
    # Exclude the few records where the sanity clamp censored the factor.
    keep = ((df["bulk_adj_factor"] > C.ADJ_FACTOR_CLAMP[0] + 1e-9)
            & (df["bulk_adj_factor"] < C.ADJ_FACTOR_CLAMP[1] - 1e-9)).to_numpy()
    beta, *_ = np.linalg.lstsq(X[keep], y[keep], rcond=None)
    gamma_hat = float(beta[1])
    resid = y[keep] - X[keep] @ beta
    r2_struct = 1.0 - float(resid.var() / y[keep].var())
    sigma_hat = float(resid.std())

    checks["recovers_exhaustion_exponent_gamma"] = (
        abs(gamma_hat - C.GAMMA_LR_DAMPING) < 0.03)
    checks["residual_noise_matches_declared_sigma"] = (
        abs(sigma_hat - C.SIGMA_PROCESS * 1.4) < 0.006)

    # (9) the ladder must contract the recipe on average: mean adjustment factor
    # is monotonically non-increasing from the 1 kg sample down to bulk.
    means = [float(df[f"{s}_adj_factor"].mean()) for s in C.STAGES[1:]]
    checks["mean_adj_factor_declines_with_scale"] = all(
        b <= a + 1e-6 for a, b in zip(means, means[1:]))

    if verbose:
        print("\n[generator] dataset validation")
        for key, val in checks.items():
            print(f"   {'PASS' if val else 'FAIL'}  {key}")
        print(f"\n   corrective trials/job    = {corrective:.3f} "
              f"(target {C.BASELINE_CORRECTIVE_TRIALS_PER_JOB})")
        print(f"   mean attempts/stage      = {mean_trials:.3f}")
        print(f"   bulk right-first-time    = {rft * 100:.1f}% "
              f"(target {C.BASELINE_RFT_BULK * 100:.0f}%)")
        print(f"   corr(bulk LR, adj factor) = {corr_lr:+.3f} raw, "
              f"{corr_lr_partial:+.3f} partial")
        print(f"   corr(depth, bulk %owf)    = {corr_depth:+.3f}")
        print(f"   recovered gamma           = {gamma_hat:.4f} "
              f"(generative {C.GAMMA_LR_DAMPING})")
        print(f"   structural R^2            = {r2_struct:.4f}, "
              f"residual sigma = {sigma_hat:.4f} "
              f"(declared {C.SIGMA_PROCESS * 1.4:.4f})")
    return checks


def write_summary_tables(df: pd.DataFrame) -> None:
    num = df.select_dtypes(include=[np.number])
    desc = num.describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desc["missing"] = num.isna().sum()
    desc.to_csv(os.path.join(C.TBL_DIR, "dataset_summary_statistics.csv"))

    key_cols = (["gsm", "cellulose_fraction", "difficulty_score", "n_dyes",
                 "dye_partition_K"]
                + [f"{s}_liquor_ratio" for s in C.STAGES]
                + [f"{s}_total_owf" for s in C.STAGES]
                + [f"{s}_adj_factor" for s in C.STAGES]
                + [f"{s}_trials" for s in C.STAGES]
                + ["bulk_salt_g_l", "bulk_soda_g_l", "total_cost_actual_usd",
                   "cost_of_poor_quality_usd"])
    key_cols = [c for c in key_cols if c in df.columns]
    df[key_cols].corr(numeric_only=True).to_csv(
        os.path.join(C.TBL_DIR, "dataset_correlation_matrix.csv"))


def main() -> pd.DataFrame:
    print("=" * 78)
    print("STEP 1  Synthetic dataset generation engine")
    print("=" * 78)
    df = generate_dataset()
    C.write_csv(df, C.MASTER_CSV)
    print(f"\n[generator] wrote {C.MASTER_CSV}")
    print(f"[generator] shape = {df.shape[0]} records x {df.shape[1]} columns")

    export_factory_format(df)
    print(f"[generator] wrote {C.FACTORY_FORMAT_CSV} (mill job-card layout)")

    validate_dataset(df)
    write_summary_tables(df)
    print(f"[generator] wrote summary statistics + correlation matrix to "
          f"{os.path.relpath(C.TBL_DIR, C.ROOT)}")

    print("\n[generator] mean adjustment factor by stage (lab dip = 1.000)")
    for s in C.STAGES:
        col = df[f"{s}_adj_factor"]
        print(f"   {C.STAGE_LABELS[s]:<24s} {col.mean():.4f} "
              f"[p5 {col.quantile(0.05):.3f}, p95 {col.quantile(0.95):.3f}]")

    print("\n[generator] mean attempts by stage")
    for s in C.STAGES:
        print(f"   {C.STAGE_LABELS[s]:<24s} {df[f'{s}_trials'].mean():.2f}")

    print("\n[generator] mean total %owf by depth class")
    print(df.groupby("depth_class")[["labdip_total_owf", "bulk_total_owf"]]
          .mean().round(4).to_string())
    return df


if __name__ == "__main__":
    main()
