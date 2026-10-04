"""
cost_benefit.py  --  STEP 5 of the research pipeline
====================================================
Techno-economic feasibility, environmental impact and cost-benefit analysis.

The point of this module
------------------------
It would be easy, and worthless, to assert that the platform lifts right-first-
time from 40% to 85% and then compute the savings that follow. Instead every
operational gain here is DERIVED from the champion model's measured error on
held-out data, through an explicit mechanism:

  1. A stage needs another dyeing when the issued recipe sits outside the
     colour-acceptance tolerance (config.RECIPE_TOLERANCE_REL, about dE 1.0).

  2. Measure, on the test set, the probability of being outside tolerance
     under each regime:
        naive        carry the lab-dip recipe to bulk unchanged
        rule-of-thumb  apply the mill's single standing bulk factor (the status
                       quo this platform must actually beat)
        AI           the champion model's prediction

  3. The rework RATE scales with that probability:
        rho = P(off-tolerance | AI) / P(off-tolerance | rule-of-thumb)

  4. Fit a Poisson model of corrective trials per stage on observable job
     features to get each job's baseline rate lambda_j, then apply rho:
        trials_AI = 1 + rho * lambda_j        RFT_AI = E[exp(-rho * lambda_j)]

  5. Re-cost every job under both regimes and aggregate to the factory.

So the headline RFT and trial numbers are outputs of the measurement, not
inputs. If the model were poor, rho would approach 1 and the savings would
collapse - which is the property that makes this analysis worth running.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split

try:
    from . import config as C
    from . import viz_theme as T
    from .data_generator import attempt_cost
except ImportError:
    import config as C
    import viz_theme as T
    from data_generator import attempt_cost

# Emission factor for boiler steam from natural gas, kg CO2e per kWh thermal.
CO2E_PER_KWH_THERMAL = 0.202
# Emission factor for treated effluent, kg CO2e per m3 (aeration + sludge).
CO2E_PER_M3_EFFLUENT = 0.35


# ===========================================================================
# 1. Off-tolerance probability under each regime
# ===========================================================================
def relative_recipe_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    """
    Concentration-weighted relative error of a 4-dye recipe, per job.

    Weighting by the true concentration is deliberate: being 20% out on a dye
    present at 0.02 %owf barely moves the shade, while 3% out on the primary
    does. An unweighted mean over the four slots would be dominated by the
    trace components and would not correspond to a visible colour difference.
    """
    pred = np.asarray(pred, float)
    true = np.asarray(true, float)
    denom = true.sum(axis=1)
    denom[denom < 1e-9] = np.nan
    return np.abs(pred - true).sum(axis=1) / denom


def off_tolerance_rate(rel_err: np.ndarray,
                       tol: float = C.RECIPE_TOLERANCE_REL) -> float:
    r = rel_err[np.isfinite(rel_err)]
    return float((r > tol).mean()) if len(r) else np.nan


def measure_regimes(df: pd.DataFrame, preds: pd.DataFrame) -> dict:
    """
    Compare the three regimes on the SAME held-out jobs.

    'preds' is outputs/tables/champion_test_predictions.csv, which carries the
    original row index of every test job so the lab-dip recipe can be joined.
    """
    idx = preds["job_index"].to_numpy()
    test = df.iloc[idx].reset_index(drop=True)

    dye_cols = [f"bulk_dye{j}_owf" for j in range(1, 5)]
    lab_cols = [f"labdip_dye{j}_owf" for j in range(1, 5)]

    true = test[dye_cols].to_numpy(float)
    lab = test[lab_cols].to_numpy(float)
    ai = preds[[f"{c}_pred" for c in dye_cols]].to_numpy(float)

    # The mill's standing bulk factor, estimated on TRAINING jobs only so the
    # status-quo baseline gets the same held-out discipline as the model.
    train_mask = np.ones(len(df), dtype=bool)
    train_mask[idx] = False
    standing_factor = float(df.loc[train_mask, "bulk_adj_factor"].mean())
    rot = lab * standing_factor

    regimes = {
        "naive (lab dip carried over)": relative_recipe_error(lab, true),
        "rule of thumb (standing factor)": relative_recipe_error(rot, true),
        "AI platform (champion model)": relative_recipe_error(ai, true),
    }
    out = {}
    for name, err in regimes.items():
        out[name] = {
            "median_rel_error_pct": float(np.nanmedian(err) * 100),
            "mean_rel_error_pct": float(np.nanmean(err) * 100),
            "p90_rel_error_pct": float(np.nanpercentile(err, 90) * 100),
            "off_tolerance_rate": off_tolerance_rate(err),
            "errors": err,
        }
    out["_standing_factor"] = standing_factor
    out["_test_index"] = idx
    return out


# ===========================================================================
# 2. Baseline rework rate per job, per stage
# ===========================================================================
POISSON_FEATURES_CAT = ["shade_family", "depth_class", "fabric_structure",
                        "composition", "pretreatment", "special_finish",
                        "dye_class"]
POISSON_FEATURES_NUM = ["gsm", "n_dyes", "is_repeat_num", "cellulose_fraction",
                        "labdip_total_owf"]


def fit_baseline_rates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimate each job's baseline corrective-trial rate lambda_j at every stage.

    A single observed count is a noisy draw from its own rate, so the rates are
    estimated with a Poisson-loss gradient booster on observable job features -
    never from the generator's internal difficulty score, which a real mill
    would not have.
    """
    work = df.copy()
    work["is_repeat_num"] = work["is_repeat"].astype(int)
    for c in POISSON_FEATURES_CAT:
        work[c] = work[c].astype("category")
    X = work[POISSON_FEATURES_CAT + POISSON_FEATURES_NUM]

    rates = pd.DataFrame(index=df.index)
    for stage in C.STAGES:
        y = (work[f"{stage}_trials"] - 1).clip(lower=0).to_numpy(float)
        model = HistGradientBoostingRegressor(
            loss="poisson", max_iter=250, learning_rate=0.06,
            max_depth=6, categorical_features="from_dtype",
            random_state=C.RANDOM_SEED)
        # Honest out-of-sample rates: 2-fold cross-fitting.
        a, b = train_test_split(np.arange(len(work)), test_size=0.5,
                                random_state=C.RANDOM_SEED, shuffle=True)
        lam = np.zeros(len(work))
        for tr, te in ((a, b), (b, a)):
            model.fit(X.iloc[tr], y[tr])
            lam[te] = np.clip(model.predict(X.iloc[te]), 1e-4, None)
        rates[f"{stage}_lambda"] = lam
    return rates


# ===========================================================================
# 3. Scenario costing
# ===========================================================================
def cost_scenarios(df: pd.DataFrame, rates: pd.DataFrame, rho: float) -> pd.DataFrame:
    """
    Cost every job under the baseline and the AI regime.

    Trial counts under the AI regime are EXPECTED values (1 + rho*lambda), so
    they are fractional; that is correct for an annualised economic comparison
    and avoids pretending a single batch can take 1.3 dyeings.
    """
    out = pd.DataFrame(index=df.index)
    out["job_id"] = df["job_id"]
    out["shade_family"] = df["shade_family"]
    out["depth_class"] = df["depth_class"]

    tot_base_cost = np.zeros(len(df))
    tot_ai_cost = np.zeros(len(df))
    tot_base_trials = np.zeros(len(df))
    tot_ai_trials = np.zeros(len(df))
    saved_water = np.zeros(len(df))
    saved_hours = np.zeros(len(df))
    saved_dye_kg = np.zeros(len(df))
    saved_salt_kg = np.zeros(len(df))
    saved_kwh = np.zeros(len(df))

    U = C.UTILITIES
    for stage in C.STAGES:
        per_attempt = df[f"{stage}_cost_per_attempt_usd"].to_numpy(float)
        base_tr = df[f"{stage}_trials"].to_numpy(float)
        ai_tr = 1.0 + rho * rates[f"{stage}_lambda"].to_numpy(float)
        ai_tr = np.minimum(ai_tr, base_tr)   # the platform never adds trials

        tot_base_cost += per_attempt * base_tr
        tot_ai_cost += per_attempt * ai_tr
        tot_base_trials += base_tr
        tot_ai_trials += ai_tr

        avoided = base_tr - ai_tr
        bath = df[f"{stage}_bath_volume_l"].to_numpy(float)
        water = bath * (1.0 + U["n_rinse_baths"] * U["rinse_volume_factor"])
        saved_water += avoided * water
        occ = (df[f"{stage}_time_min"].to_numpy(float)
               + C.STAGE_HANDLING_MIN[stage]) / 60.0
        saved_hours += avoided * occ
        saved_dye_kg += avoided * (df[f"{stage}_dye_mass_g"].to_numpy(float) / 1000.0)
        saved_salt_kg += avoided * (df[f"{stage}_salt_mass_kg"].to_numpy(float)
                                    + df[f"{stage}_soda_mass_kg"].to_numpy(float))
        dT = np.clip(df[f"{stage}_temp_c"].to_numpy(float) - U["ambient_temp_c"], 0, None)
        kwh = ((bath * U["water_cp_kj_kg_k"] * (dT + 50.0) / 3600.0)
               / U["boiler_efficiency"])
        saved_kwh += avoided * kwh

        out[f"{stage}_trials_baseline"] = base_tr
        out[f"{stage}_trials_ai"] = ai_tr

    out["cost_baseline_usd"] = tot_base_cost
    out["cost_ai_usd"] = tot_ai_cost
    out["saving_usd"] = tot_base_cost - tot_ai_cost
    out["corrective_trials_baseline"] = tot_base_trials - len(C.STAGES)
    out["corrective_trials_ai"] = tot_ai_trials - len(C.STAGES)
    out["water_saved_l"] = saved_water
    out["machine_hours_saved"] = saved_hours
    out["dye_saved_kg"] = saved_dye_kg
    out["chemicals_saved_kg"] = saved_salt_kg
    out["thermal_kwh_saved"] = saved_kwh
    out["co2e_saved_kg"] = (saved_kwh * CO2E_PER_KWH_THERMAL
                            + saved_water / 1000.0 * CO2E_PER_M3_EFFLUENT)
    return out


# ===========================================================================
# 4. Factory roll-up and investment appraisal
# ===========================================================================
def factory_rollup(scen: pd.DataFrame, rft_base: float, rft_ai: float) -> dict:
    """Scale per-job results to the factory and appraise the investment."""
    F, I = C.FACTORY, C.INVESTMENT
    programmes = F["distinct_shades_per_year"]      # one scale-up ladder per shade

    per_job_saving = float(scen["saving_usd"].mean())
    annual_gross = per_job_saving * programmes
    annual_net = annual_gross - I["annual_opex_usd"]

    payback_years = (I["capex_usd"] / annual_net) if annual_net > 0 else np.inf
    r, n = I["discount_rate"], I["horizon_years"]
    npv = -I["capex_usd"] + sum(annual_net / (1 + r) ** t for t in range(1, n + 1))

    # IRR by bisection on the NPV of a level annuity.
    def _npv(rate):
        return -I["capex_usd"] + sum(annual_net / (1 + rate) ** t
                                     for t in range(1, n + 1))
    irr = np.nan
    if annual_net > 0:
        lo, hi = 0.0, 10.0
        if _npv(hi) < 0:
            for _ in range(200):
                mid = 0.5 * (lo + hi)
                if _npv(mid) > 0:
                    lo = mid
                else:
                    hi = mid
            irr = 0.5 * (lo + hi)

    return {
        "programmes_per_year": programmes,
        "annual_output_kg": F["annual_output_kg"],
        "mean_cost_baseline_usd": float(scen["cost_baseline_usd"].mean()),
        "mean_cost_ai_usd": float(scen["cost_ai_usd"].mean()),
        "mean_saving_per_programme_usd": per_job_saving,
        "saving_pct_of_spend": 100.0 * per_job_saving / float(scen["cost_baseline_usd"].mean()),
        "annual_gross_saving_usd": annual_gross,
        "annual_gross_saving_lkr": annual_gross * C.LKR_PER_USD,
        "annual_opex_usd": I["annual_opex_usd"],
        "annual_net_saving_usd": annual_net,
        "capex_usd": I["capex_usd"],
        "payback_years": payback_years,
        "payback_months": payback_years * 12.0,
        "npv_5yr_usd": npv,
        "irr": irr,
        "roi_5yr_pct": 100.0 * (annual_net * n - I["capex_usd"]) / I["capex_usd"],
        "corrective_trials_baseline": float(scen["corrective_trials_baseline"].mean()),
        "corrective_trials_ai": float(scen["corrective_trials_ai"].mean()),
        "rft_bulk_baseline": rft_base,
        "rft_bulk_ai": rft_ai,
        "annual_water_saved_m3": float(scen["water_saved_l"].mean()) * programmes / 1000.0,
        "annual_machine_hours_saved": float(scen["machine_hours_saved"].mean()) * programmes,
        "annual_dye_saved_kg": float(scen["dye_saved_kg"].mean()) * programmes,
        "annual_chemicals_saved_kg": float(scen["chemicals_saved_kg"].mean()) * programmes,
        "annual_thermal_mwh_saved": float(scen["thermal_kwh_saved"].mean()) * programmes / 1000.0,
        "annual_co2e_saved_t": float(scen["co2e_saved_kg"].mean()) * programmes / 1000.0,
    }


# ===========================================================================
# 5. Figures
# ===========================================================================
def fig10_regime_errors(regimes: dict) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2))
    names = [k for k in regimes if not k.startswith("_")]
    cols = [T.STATUS["critical"], T.SERIES[1], T.STATUS["good"]]

    for name, col in zip(names, cols):
        err = regimes[name]["errors"] * 100
        err = err[np.isfinite(err)]
        grid = np.linspace(0, 30, 400)
        cdf = [(err <= g).mean() * 100 for g in grid]
        ax1.plot(grid, cdf, color=col, linewidth=2.2, label=name)
    ax1.axvline(C.RECIPE_TOLERANCE_REL * 100, color=T.INK_MUTED, linewidth=1.4,
                linestyle=(0, (4, 3)))
    ax1.annotate(f"colour tolerance\n{C.RECIPE_TOLERANCE_REL * 100:.1f}%",
                 (C.RECIPE_TOLERANCE_REL * 100, 52), textcoords="offset points",
                 xytext=(10, 0), fontsize=8.5, color=T.INK_MUTED)
    ax1.set_xlabel("Relative recipe error (%)")
    ax1.set_ylabel("Share of jobs within this error (%)")
    ax1.set_xlim(0, 30)
    ax1.set_ylim(0, 102)
    ax1.legend(loc="lower right")
    T.titled(ax1, "How close does each regime get?",
             "Cumulative distribution of concentration-weighted recipe error")

    rates = [regimes[n]["off_tolerance_rate"] * 100 for n in names]
    x = np.arange(len(names))
    ax2.bar(x, rates, color=cols, edgecolor=T.SURFACE, linewidth=2.0, width=0.62)
    for xi, v in zip(x, rates):
        ax2.text(xi, v + 1.5, f"{v:.1f}%", ha="center", fontsize=10,
                 color=T.INK_2, fontweight="bold")
    ax2.set_xticks(x, [n.replace(" (", "\n(") for n in names], fontsize=8.5)
    ax2.set_ylabel("Jobs outside colour tolerance (%)")
    ax2.set_ylim(0, 108)
    T.titled(ax2, "The quantity that drives every saving",
             "Rework rate is scaled by the ratio of these probabilities")

    T.source_note(fig, "Held-out test jobs only. The standing-factor baseline is "
                       "estimated on training jobs, so both regimes are judged "
                       "out of sample.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig10_regime_errors.png"))


def fig11_savings(scen: pd.DataFrame, roll: dict) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16.2, 5.0))

    # --- per-stage trial reduction ----------------------------------------
    base = [scen[f"{s}_trials_baseline"].mean() for s in C.STAGES]
    ai = [scen[f"{s}_trials_ai"].mean() for s in C.STAGES]
    x = np.arange(len(C.STAGES))
    w = 0.38
    ax1.bar(x - w / 2 - 0.01, base, w, color=T.SERIES[0], edgecolor=T.SURFACE,
            linewidth=2.0, label="Current practice")
    ax1.bar(x + w / 2 + 0.01, ai, w, color=T.SERIES[1], edgecolor=T.SURFACE,
            linewidth=2.0, label="AI platform (derived)")
    for xi, (b, a) in enumerate(zip(base, ai)):
        ax1.text(xi - w / 2 - 0.01, b + 0.04, f"{b:.2f}", ha="center",
                 fontsize=8.5, color=T.INK_2, fontweight="bold")
        ax1.text(xi + w / 2 + 0.01, a + 0.04, f"{a:.2f}", ha="center",
                 fontsize=8.5, color=T.INK_2, fontweight="bold")
    ax1.set_xticks(x, [C.STAGE_LABELS[s].split(" (")[0] for s in C.STAGES],
                   rotation=20, ha="right", fontsize=8.5)
    ax1.set_ylabel("Mean dyeing attempts")
    ax1.legend(loc="upper center")
    ax1.set_ylim(0, max(base) * 1.25)
    T.titled(ax1, "Trials avoided",
             f"{roll['corrective_trials_baseline']:.2f} to "
             f"{roll['corrective_trials_ai']:.2f} corrective trials per shade")

    # --- savings by depth class -------------------------------------------
    dcol = T.depth_colors()
    g = scen.groupby("depth_class")["saving_usd"].mean().reindex(T.DEPTH_ORDER)
    xb = np.arange(len(g))
    ax2.bar(xb, g.values, color=[dcol[d] for d in g.index],
            edgecolor=T.SURFACE, linewidth=2.0, width=0.66)
    for xi, v in zip(xb, g.values):
        ax2.text(xi, v + g.max() * 0.02, f"${v:,.0f}", ha="center", fontsize=9,
                 color=T.INK_2, fontweight="bold")
    ax2.set_xticks(xb, [d.replace("/", "/\n") for d in g.index], fontsize=8.5)
    ax2.set_ylabel("Mean saving per scale-up programme (USD)")
    ax2.set_ylim(0, g.max() * 1.18)
    T.titled(ax2, "Where the money is",
             "Deep shades cost the most to get wrong, and save the most")

    # --- cumulative cash flow ---------------------------------------------
    yrs = np.arange(0, C.INVESTMENT["horizon_years"] + 1)
    cash = [-C.INVESTMENT["capex_usd"]
            + roll["annual_net_saving_usd"] * t for t in yrs]
    ax3.plot(yrs, cash, color=T.SERIES[0], marker="o", linewidth=2.2,
             markeredgecolor=T.SURFACE, markeredgewidth=2.0)
    ax3.axhline(0, color=T.INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    pb = roll["payback_years"]
    if np.isfinite(pb) and pb <= yrs[-1]:
        ax3.axvline(pb, color=T.STATUS["good"], linewidth=2.0,
                    linestyle=(0, (5, 3)))
        ax3.annotate(f"payback\n{pb * 12:.1f} months", (pb, 0),
                     textcoords="offset points", xytext=(10, 28), fontsize=9,
                     color=T.STATUS["good"], fontweight="bold")
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Cumulative net cash flow (USD)")
    ax3.yaxis.set_major_formatter(
        plt.FuncFormatter(lambda v, _: f"-${abs(v) / 1000:,.0f}k" if v < 0
                          else f"${v / 1000:,.0f}k"))
    T.titled(ax3, "Investment appraisal",
             f"NPV ${roll['npv_5yr_usd']:,.0f} at "
             f"{C.INVESTMENT['discount_rate'] * 100:.0f}% over "
             f"{C.INVESTMENT['horizon_years']} years")

    T.source_note(fig, "Savings are derived from measured model error, not assumed. "
                       "Undiscounted cash line; NPV quoted in the panel title.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig11_cost_benefit.png"))


# ===========================================================================
# 6. Main
# ===========================================================================
def main() -> dict:
    print("=" * 78)
    print("STEP 5  Techno-economic feasibility & cost-benefit analysis")
    print("=" * 78)
    T.apply_theme()

    df = pd.read_csv(C.master_path())
    pred_path = os.path.join(C.TBL_DIR, "champion_test_predictions.csv")
    if not os.path.exists(pred_path):
        raise SystemExit("Champion predictions not found. Run: python src/ml_pipeline.py")
    preds = pd.read_csv(pred_path)

    # ---- 1. measure the three regimes ------------------------------------
    regimes = measure_regimes(df, preds)
    print(f"\n--- Recipe accuracy by regime (held-out, n={len(preds):,}) --------")
    print(f"mill standing bulk factor (from training jobs): "
          f"{regimes['_standing_factor']:.4f}")
    rows = []
    for name, m in regimes.items():
        if name.startswith("_"):
            continue
        print(f"  {name:<34s} median {m['median_rel_error_pct']:6.2f}%   "
              f"p90 {m['p90_rel_error_pct']:6.2f}%   "
              f"off-tolerance {m['off_tolerance_rate'] * 100:5.1f}%")
        rows.append({"regime": name,
                     "median_rel_error_pct": m["median_rel_error_pct"],
                     "mean_rel_error_pct": m["mean_rel_error_pct"],
                     "p90_rel_error_pct": m["p90_rel_error_pct"],
                     "off_tolerance_rate_pct": m["off_tolerance_rate"] * 100})
    C.write_csv(pd.DataFrame(rows).round(4),
                os.path.join(C.TBL_DIR, "cba_regime_accuracy.csv"))

    p_rot = regimes["rule of thumb (standing factor)"]["off_tolerance_rate"]
    p_ai = regimes["AI platform (champion model)"]["off_tolerance_rate"]
    rho = float(np.clip(p_ai / p_rot, 0.0, 1.0)) if p_rot > 0 else 1.0
    print(f"\n  rework-rate multiplier  rho = P(off-tol | AI) / P(off-tol | rule of thumb)")
    print(f"                              = {p_ai:.4f} / {p_rot:.4f} = {rho:.4f}")

    # ---- 2. baseline rates ------------------------------------------------
    print("\n[cba] fitting Poisson rework-rate models (cross-fitted)...")
    rates = fit_baseline_rates(df)

    # ---- 3. RFT under each regime ----------------------------------------
    lam_bulk = rates["bulk_lambda"].to_numpy(float)
    rft_base_model = float(np.mean(np.exp(-lam_bulk)))
    rft_base_obs = float(df["rft_bulk"].mean())
    rft_ai = float(np.mean(np.exp(-rho * lam_bulk)))
    print(f"\n--- Right-first-time at bulk --------------------------------------")
    print(f"  observed baseline            {rft_base_obs * 100:5.1f}%")
    print(f"  Poisson-model baseline       {rft_base_model * 100:5.1f}%  (sanity check)")
    print(f"  derived under the AI regime  {rft_ai * 100:5.1f}%"
          f"   (target {C.TARGET_RFT_BULK * 100:.0f}%)")

    # ---- 4. cost both scenarios ------------------------------------------
    scen = cost_scenarios(df, rates, rho)
    C.write_csv(scen.round(4), os.path.join(C.TBL_DIR, "cba_per_job_scenarios.csv"))

    roll = factory_rollup(scen, rft_base_obs, rft_ai)

    print("\n--- Operational impact per scale-up programme ---------------------")
    print(f"  corrective trials   {roll['corrective_trials_baseline']:.2f}  ->  "
          f"{roll['corrective_trials_ai']:.2f}   "
          f"(target {C.TARGET_CORRECTIVE_TRIALS_PER_JOB})")
    print(f"  cost                ${roll['mean_cost_baseline_usd']:,.2f}  ->  "
          f"${roll['mean_cost_ai_usd']:,.2f}")
    print(f"  saving              ${roll['mean_saving_per_programme_usd']:,.2f} "
          f"({roll['saving_pct_of_spend']:.1f}% of spend)")

    print(f"\n--- Annual factory impact ({roll['programmes_per_year']} programmes, "
          f"{roll['annual_output_kg']:,.0f} kg output) ---")
    print(f"  gross saving        ${roll['annual_gross_saving_usd']:,.0f}"
          f"   (LKR {roll['annual_gross_saving_lkr']:,.0f})")
    print(f"  net of platform opex ${roll['annual_net_saving_usd']:,.0f}")
    print(f"  dyestuff saved      {roll['annual_dye_saved_kg']:,.0f} kg")
    print(f"  chemicals saved     {roll['annual_chemicals_saved_kg']:,.0f} kg")
    print(f"  water saved         {roll['annual_water_saved_m3']:,.0f} m3")
    print(f"  thermal energy      {roll['annual_thermal_mwh_saved']:,.1f} MWh")
    print(f"  CO2e avoided        {roll['annual_co2e_saved_t']:,.1f} t")
    print(f"  machine time freed  {roll['annual_machine_hours_saved']:,.0f} h")

    print("\n--- Investment appraisal -----------------------------------------")
    print(f"  capex               ${roll['capex_usd']:,.0f}")
    print(f"  payback             {roll['payback_months']:.1f} months")
    print(f"  NPV ({C.INVESTMENT['horizon_years']} yr @ "
          f"{C.INVESTMENT['discount_rate'] * 100:.0f}%)  ${roll['npv_5yr_usd']:,.0f}")
    print(f"  IRR                 {roll['irr'] * 100:.1f}%" if np.isfinite(roll["irr"])
          else "  IRR                 n/a")
    print(f"  5-year ROI          {roll['roi_5yr_pct']:,.0f}%")

    C.write_csv(pd.DataFrame([roll]).round(4),
                os.path.join(C.TBL_DIR, "cba_factory_summary.csv"))

    fig10_regime_errors(regimes)
    fig11_savings(scen, roll)
    print(f"\n[cba] figures and tables written")

    # ---- what accuracy would the stated RFT target actually require? ------
    #
    # Reporting "target not met" is only half an answer. Because RFT is
    # monotone in rho, the rho that would hit the target can be solved for
    # directly, then translated back into the off-tolerance rate and the
    # recipe error the model would have to achieve. That turns a missed target
    # into a specification.
    gap = None
    if rft_ai < C.TARGET_RFT_BULK:
        lo, hi = 0.0, 1.0
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            if float(np.mean(np.exp(-mid * lam_bulk))) > C.TARGET_RFT_BULK:
                lo = mid
            else:
                hi = mid
        rho_req = 0.5 * (lo + hi)
        off_tol_req = rho_req * p_rot
        ai_err = regimes["AI platform (champion model)"]["errors"]
        ai_err = ai_err[np.isfinite(ai_err)]
        # The recipe error whose exceedance probability equals off_tol_req.
        err_req = float(np.quantile(ai_err, 1.0 - off_tol_req))
        gap = {
            "rft_target": C.TARGET_RFT_BULK,
            "rft_achieved": rft_ai,
            "rho_achieved": rho,
            "rho_required": rho_req,
            "off_tolerance_achieved_pct": p_ai * 100,
            "off_tolerance_required_pct": off_tol_req * 100,
            "current_p90_error_pct":
                regimes["AI platform (champion model)"]["p90_rel_error_pct"],
            "tolerance_band_required_pct": err_req * 100,
        }
        print("\n--- What the 85% RFT target would require -------------------------")
        print(f"  achieved RFT {rft_ai * 100:.1f}% against a {C.TARGET_RFT_BULK * 100:.0f}% target.")
        print(f"  rework multiplier rho must fall from {rho:.3f} to {rho_req:.3f},")
        print(f"  i.e. the off-tolerance rate from {p_ai * 100:.1f}% to "
              f"{off_tol_req * 100:.1f}%.")
        print(f"  With this model's error distribution that means holding the")
        print(f"  colour tolerance at {err_req * 100:.2f}% relative error rather")
        print(f"  than {C.RECIPE_TOLERANCE_REL * 100:.1f}% - equivalently, roughly halving")
        print(f"  the model's recipe error, or widening the accepted tolerance.")
        C.write_csv(pd.DataFrame([gap]).round(4),
                    os.path.join(C.TBL_DIR, "cba_target_gap_analysis.csv"))

    # ---- target reconciliation -------------------------------------------
    print("\n--- Reconciliation against the stated programme targets -----------")
    def _verdict(actual, target, higher_is_better=True):
        ok = actual >= target if higher_is_better else actual <= target
        return "MET" if ok else "NOT MET"
    print(f"  RFT at bulk >= {C.TARGET_RFT_BULK * 100:.0f}%        "
          f"achieved {rft_ai * 100:.1f}%   "
          f"{_verdict(rft_ai, C.TARGET_RFT_BULK)}")
    print(f"  corrective trials <= {C.TARGET_CORRECTIVE_TRIALS_PER_JOB}    "
          f"achieved {roll['corrective_trials_ai']:.2f}   "
          f"{_verdict(roll['corrective_trials_ai'], C.TARGET_CORRECTIVE_TRIALS_PER_JOB, False)}")
    return dict(regimes=regimes, rho=rho, rates=rates, scenarios=scen, rollup=roll,
                rft_ai=rft_ai, rft_base=rft_base_obs, gap=gap)


if __name__ == "__main__":
    main()
