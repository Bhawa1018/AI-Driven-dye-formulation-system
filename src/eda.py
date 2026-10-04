"""
eda.py  --  STEP 2 of the research pipeline
===========================================
Exploratory data analysis and publication-ready figures.

Produces, in outputs/figures/:
  fig01_recipe_shift_across_stages.png   recipe adjustment along the ladder
  fig02_correlation_heatmap.png          GSM / depth / liquor ratio / bulk conc
  fig03_trial_distributions.png          baseline vs AI design target
  fig04_cost_impact.png                  cost of poor quality by family & depth
  fig05_exhaustion_physics.png           adjustment factor vs liquor ratio
  fig06_depth_concentration_windows.png  %owf windows by depth class

and, in outputs/tables/, the numeric companions to each figure.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from . import config as C
    from . import viz_theme as T
except ImportError:
    import config as C
    import viz_theme as T


# ===========================================================================
def load_master() -> pd.DataFrame:
    if not os.path.exists(C.MASTER_CSV):
        raise SystemExit("Master dataset not found. Run: python src/data_generator.py")
    return pd.read_csv(C.master_path())


# ===========================================================================
# Figure 1 - recipe adjustment shift across the five stages
# ===========================================================================
def fig01_recipe_shift(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.2),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    colors = T.stage_colors(5)
    labels = [C.STAGE_LABELS[s].replace(" (", "\n(") for s in C.STAGES]

    # --- left: distribution of the adjustment factor at each stage ----------
    # The lab dip is the reference (factor identically 1.000, zero variance), so
    # it is drawn as the baseline rather than as a degenerate box.
    scale_stages = C.STAGES[1:]
    data = [df[f"{s}_adj_factor"] for s in scale_stages]
    bp = ax1.boxplot(data, patch_artist=True, widths=0.56, showfliers=False,
                     medianprops=dict(color=T.SURFACE, linewidth=2.0),
                     whiskerprops=dict(color=T.AXIS, linewidth=1.2),
                     capprops=dict(color=T.AXIS, linewidth=1.2))
    for patch, col in zip(bp["boxes"], colors[1:]):
        patch.set_facecolor(col)
        patch.set_edgecolor(T.SURFACE)   # 2px surface gap between adjacent fills
        patch.set_linewidth(2.0)

    ax1.axhline(1.0, color=T.INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)),
                zorder=1)
    ax1.text(4.56, 1.004, "lab dip\nreference\n= 1.000", fontsize=8,
             color=T.INK_MUTED, va="bottom", ha="left")
    ax1.set_xticks(range(1, 5), labels[1:], fontsize=8.5)
    ax1.set_xlim(0.4, 5.3)
    ax1.set_ylabel("Recipe adjustment factor (x lab dip % owf)")
    T.titled(ax1, "Recipes contract as the batch grows",
             "Distribution of the lab-dip-to-stage correction factor, 8,000 jobs")

    # Direct labels above each whisker cap, clear of the fills.
    for i, s in enumerate(scale_stages, start=1):
        med = df[f"{s}_adj_factor"].median()
        top = df[f"{s}_adj_factor"].quantile(0.75)
        iqr = top - df[f"{s}_adj_factor"].quantile(0.25)
        ax1.annotate(f"median {med:.3f}", (i, min(top + 1.5 * iqr, 1.30)),
                     textcoords="offset points", xytext=(0, 7), ha="center",
                     fontsize=8.5, color=T.INK_2, fontweight="bold")

    # --- right: mean trajectory by depth class ------------------------------
    dcol = T.depth_colors()
    x = np.arange(5)
    for dc in T.DEPTH_ORDER:
        sub = df[df["depth_class"] == dc]
        if sub.empty:
            continue
        y = [sub[f"{s}_adj_factor"].mean() for s in C.STAGES]
        ax2.plot(x, y, color=dcol[dc], marker="o", linewidth=2.0,
                 markeredgecolor=T.SURFACE, markeredgewidth=2.0, label=dc,
                 zorder=3)
        ax2.annotate(f"{y[-1]:.3f}", (x[-1], y[-1]), textcoords="offset points",
                     xytext=(8, 0), fontsize=8.5, color=dcol[dc],
                     fontweight="bold", va="center")

    ax2.axhline(1.0, color=T.INK_MUTED, linewidth=1.0, linestyle=(0, (4, 3)))
    ax2.set_xticks(x, labels, fontsize=8.5)
    ax2.set_xlim(-0.25, 4.75)
    ax2.set_ylabel("Mean adjustment factor")
    ax2.legend(title="Depth class", title_fontsize=9, loc="lower left")
    T.titled(ax2, "Deep shades resist the contraction",
             "Bath saturation and dye aggregation offset the exhaustion gain")

    T.source_note(fig, "Source: textile_dyeing_scaleup_master.csv (n = 8,000 synthetic jobs). "
                       "Factor 1.000 = the lab-dip concentration carried over unchanged.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig01_recipe_shift_across_stages.png"))

    tbl = pd.DataFrame({
        "stage": [C.STAGE_LABELS[s] for s in C.STAGES],
        "mean_liquor_ratio": [df[f"{s}_liquor_ratio"].mean() for s in C.STAGES],
        "mean_adj_factor": [df[f"{s}_adj_factor"].mean() for s in C.STAGES],
        "p5": [df[f"{s}_adj_factor"].quantile(0.05) for s in C.STAGES],
        "p95": [df[f"{s}_adj_factor"].quantile(0.95) for s in C.STAGES],
        "mean_total_owf": [df[f"{s}_total_owf"].mean() for s in C.STAGES],
    }).round(4)
    C.write_csv(tbl, os.path.join(C.TBL_DIR, "eda_recipe_shift_by_stage.csv"))


# ===========================================================================
# Figure 2 - correlation heatmap
# ===========================================================================
def fig02_correlation(df: pd.DataFrame) -> None:
    work = df.copy()
    work["depth_ordinal"] = work["depth_class"].map(
        {"Pastel/Light": 0, "Medium": 1, "Dark": 2, "Extra Dark": 3})
    work["is_repeat_num"] = work["is_repeat"].astype(int)

    cols = {
        "gsm": "Fabric GSM",
        "cellulose_fraction": "Cellulose fraction",
        "depth_ordinal": "Depth class",
        "n_dyes": "Number of dyes",
        "dye_partition_K": "Dye affinity K",
        "difficulty_score": "Job difficulty",
        "is_repeat_num": "Repeat shade",
        "labdip_total_owf": "Lab dip % owf",
        "bulk_liquor_ratio": "Bulk liquor ratio",
        "bulk_total_owf": "Bulk % owf (target)",
        "bulk_adj_factor": "Bulk adj. factor",
        "bulk_salt_g_l": "Bulk salt g/L",
        "bulk_soda_g_l": "Bulk soda g/L",
        "bulk_trials": "Bulk trials",
        "total_cost_actual_usd": "Total cost USD",
    }
    mat = work[list(cols)].corr().rename(index=cols, columns=cols)

    fig, ax = plt.subplots(figsize=(10.4, 8.6))
    im = ax.imshow(mat.values, cmap=T.DIVERGING, vmin=-1, vmax=1)
    T.strip_grid(ax)

    n = len(mat)
    ax.set_xticks(range(n), mat.columns, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(n), mat.index, fontsize=8.5)
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    # 2px surface gap between adjacent cells
    ax.grid(which="minor", color=T.SURFACE, linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # Label every cell: the matrix is small enough that the numbers are the data.
    for i in range(n):
        for j in range(n):
            v = mat.values[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "-."),
                    ha="center", va="center", fontsize=7.2,
                    color=T.SURFACE if abs(v) > 0.55 else T.INK_2,
                    fontweight="bold" if abs(v) > 0.55 else "normal")

    cb = fig.colorbar(im, ax=ax, shrink=0.62, pad=0.02, ticks=[-1, -0.5, 0, 0.5, 1])
    cb.set_label("Pearson correlation", fontsize=9, color=T.INK_2)
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=8, color=T.AXIS)

    T.titled(ax, "What actually drives the bulk recipe",
             "Depth class and lab-dip concentration dominate; liquor ratio governs "
             "the correction factor")
    T.source_note(fig, "Source: textile_dyeing_scaleup_master.csv (n = 8,000). "
                       "Diverging scale: blue = negative, red = positive, grey = no association.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig02_correlation_heatmap.png"))
    C.write_csv(mat.round(4), os.path.join(C.TBL_DIR, "eda_correlation_matrix_labelled.csv"),
                index=True)


# ===========================================================================
# Figure 3 - trial distributions, baseline vs AI design target
# ===========================================================================
def fig03_trials(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0),
                                   gridspec_kw={"width_ratios": [1, 1.1]})
    base_c, ai_c = T.SERIES[0], T.SERIES[1]

    # --- left: attempts per stage, baseline vs target ----------------------
    stages = C.STAGES
    base_means = [df[f"{s}_trials"].mean() for s in stages]
    # The design target shrinks the corrective-trial budget proportionally across
    # stages, floored at 1 attempt (a stage never needs fewer than one dyeing).
    # Both figures are per-job corrective totals, per the unit convention
    # documented in config.
    scale = (C.TARGET_CORRECTIVE_TRIALS_PER_JOB
             / C.BASELINE_CORRECTIVE_TRIALS_PER_JOB)
    target_means = [1.0 + (m - 1.0) * scale for m in base_means]

    x = np.arange(len(stages))
    w = 0.38
    ax1.bar(x - w / 2 - 0.01, base_means, w, label="Current practice",
            color=base_c, edgecolor=T.SURFACE, linewidth=2.0)
    ax1.bar(x + w / 2 + 0.01, target_means, w, label="AI platform (design target)",
            color=ai_c, edgecolor=T.SURFACE, linewidth=2.0)
    for xi, (b, t) in enumerate(zip(base_means, target_means)):
        ax1.text(xi - w / 2 - 0.01, b + 0.09, f"{b:.2f}", ha="center",
                 fontsize=8.5, color=T.INK_2, fontweight="bold")
        ax1.text(xi + w / 2 + 0.01, t + 0.09, f"{t:.2f}", ha="center",
                 fontsize=8.5, color=T.INK_2, fontweight="bold")
    ax1.set_xticks(x, [C.STAGE_LABELS[s].replace(" (", "\n(") for s in stages],
                   fontsize=8.5)
    ax1.set_ylabel("Mean dyeing attempts per stage")
    ax1.set_ylim(0, max(base_means) * 1.22)
    ax1.legend(loc="upper right")
    T.titled(ax1, "Where the iterations are spent",
             f"Corrective trials per shade fall from "
             f"{C.BASELINE_CORRECTIVE_TRIALS_PER_JOB:.1f} to "
             f"{C.TARGET_CORRECTIVE_TRIALS_PER_JOB:.1f}")

    # --- right: distribution of corrective trials per job ------------------
    # Single series: the title names it, so no legend box.
    vals = df["corrective_trials"]
    hi = int(vals.quantile(0.999)) + 1
    bins = np.arange(-0.5, hi + 1.5, 1)
    ax2.hist(vals, bins=bins, color=base_c, edgecolor=T.SURFACE, linewidth=2.0)
    ax2.set_xlim(-0.8, hi + 0.8)
    ax2.axvline(vals.mean(), color=T.STATUS["critical"], linewidth=2.0,
                linestyle=(0, (5, 3)), zorder=5)
    ax2.annotate(f"mean {vals.mean():.1f} corrective\ntrials per job",
                 (vals.mean(), ax2.get_ylim()[1] * 0.95),
                 textcoords="offset points", xytext=(14, 0), fontsize=8.5,
                 color=T.STATUS["critical"], fontweight="bold", va="top",
                 bbox=dict(facecolor=T.SURFACE, edgecolor="none", pad=2.0))
    ax2.set_xlabel("Corrective trials per job (attempts beyond one per stage)")
    ax2.set_ylabel("Jobs")
    T.titled(ax2, "A long right tail of rework",
             f"Bulk right-first-time rate is {df['rft_bulk'].mean() * 100:.1f}%; "
             f"only {df['rft_all_stages'].mean() * 100:.1f}% of jobs clear every stage first time")

    T.source_note(fig, "Source: textile_dyeing_scaleup_master.csv (n = 8,000). "
                       "The AI target scenario here is the design specification; the "
                       "measured outcome is derived from model residuals in Step 5.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig03_trial_distributions.png"))

    C.write_csv(pd.DataFrame({
        "stage": [C.STAGE_LABELS[s] for s in stages],
        "mean_trials_baseline": np.round(base_means, 3),
        "mean_trials_target": np.round(target_means, 3),
        "rft_rate_baseline": [round(float((df[f"{s}_trials"] == 1).mean()), 4)
                              for s in stages],
    }), os.path.join(C.TBL_DIR, "eda_trials_by_stage.csv"))


# ===========================================================================
# Figure 4 - cost impact
# ===========================================================================
def fig04_cost(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.4),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # --- left: cost of poor quality by shade family (single hue, ordered) ---
    g = (df.groupby("shade_family")
           .agg(cpq=("cost_of_poor_quality_usd", "mean"),
                corrective=("corrective_trials", "mean"),
                owf=("bulk_total_owf", "mean"),
                n=("job_id", "size"))
           .sort_values("cpq"))
    r_trials = g["cpq"].corr(g["corrective"])
    r_owf = g["cpq"].corr(g["owf"])
    ypos = np.arange(len(g))
    ax1.barh(ypos, g["cpq"], color=T.SEQ_HUE, edgecolor=T.SURFACE,
             linewidth=2.0, height=0.74)
    ax1.set_yticks(ypos, [f"{i}  (n={int(nn)})" for i, nn in
                          zip(g.index, g["n"])], fontsize=8.5)
    ax1.grid(False)
    ax1.xaxis.grid(True)
    ax1.set_xlabel("Mean cost of poor quality per job (USD)")
    # label only the extremes, not every bar
    for k in (0, len(g) - 1):
        ax1.text(g["cpq"].iloc[k] + g["cpq"].max() * 0.015, ypos[k],
                 f"${g['cpq'].iloc[k]:,.0f}", va="center", fontsize=8.5,
                 color=T.INK_2, fontweight="bold")
    T.titled(ax1, "Rework cost follows the trial count",
             f"Across families, cost of poor quality tracks corrective trials "
             f"(r = {r_trials:.2f}) and shade depth (r = {r_owf:.2f})")

    # --- right: distribution by depth class (ordinal hue) ------------------
    dcol = T.depth_colors()
    data = [df.loc[df["depth_class"] == dc, "cost_of_poor_quality_usd"]
            for dc in T.DEPTH_ORDER]
    parts = ax2.violinplot(data, showextrema=False, widths=0.78)
    for body, dc in zip(parts["bodies"], T.DEPTH_ORDER):
        body.set_facecolor(dcol[dc])
        body.set_edgecolor(T.SURFACE)
        body.set_linewidth(2.0)
        body.set_alpha(1.0)
    # Clip the axis to the bulk of the distribution; the thin upper tail would
    # otherwise push every violin into the bottom fifth of the panel.
    ytop = float(max(d.quantile(0.99) for d in data)) * 1.18
    ax2.set_ylim(0, ytop)
    for i, d in enumerate(data, start=1):
        ax2.plot([i], [d.median()], marker="o", color=T.SURFACE,
                 markersize=7, markeredgecolor=T.INK_2, markeredgewidth=1.6,
                 zorder=6)
        # Label above the violin's visible extent, never on top of the fill.
        ax2.annotate(f"median ${d.median():,.0f}",
                     (i, min(d.quantile(0.985), ytop * 0.93)),
                     textcoords="offset points", xytext=(0, 9), ha="center",
                     fontsize=8, color=T.INK_2, fontweight="bold")
    ax2.set_xticks(range(1, 5), [d.replace("/", "/\n") for d in T.DEPTH_ORDER],
                   fontsize=8.5)
    ax2.set_ylabel("Cost of poor quality per job (USD)")
    ax2.legend(handles=[Patch(facecolor=dcol[d], edgecolor=T.SURFACE, label=d)
                        for d in T.DEPTH_ORDER],
               title="Depth class", title_fontsize=9, loc="upper left")
    T.titled(ax2, "Depth drives the exposure",
             "Extra-dark shades combine the most dye, the most salt and the most rework")

    T.source_note(fig, "Source: textile_dyeing_scaleup_master.csv (n = 8,000). "
                       "Costs are fully absorbed: dyes, chemicals, water, steam, "
                       "effluent, machine occupancy and stage overhead.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig04_cost_impact.png"))

    out = (df.groupby(["shade_family", "depth_class"])
             .agg(jobs=("job_id", "size"),
                  mean_total_cost_usd=("total_cost_actual_usd", "mean"),
                  mean_cpq_usd=("cost_of_poor_quality_usd", "mean"),
                  mean_corrective_trials=("corrective_trials", "mean"))
             .round(2).reset_index())
    C.write_csv(out, os.path.join(C.TBL_DIR, "eda_cost_by_family_depth.csv"))


# ===========================================================================
# Figure 5 - the exhaustion physics made visible
# ===========================================================================
def fig05_physics(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.0))

    # --- left: adjustment factor vs bulk liquor ratio, binned by dye class --
    classes = list(C.DYE_CLASSES.keys())
    # Four nominal categories: fold to the three validated slots plus a
    # positional facet would be cleaner, so bin and use direct labels instead.
    cols = T.stage_colors(len(classes))
    for cls, col in zip(classes, cols):
        sub = df[df["dye_class"] == cls]
        if len(sub) < 30:
            continue
        bins = pd.cut(sub["bulk_liquor_ratio"], bins=np.arange(6, 10.5, 0.5))
        grp = sub.groupby(bins, observed=True)["bulk_adj_factor"].mean()
        xs = [iv.mid for iv in grp.index]
        ax1.plot(xs, grp.values, marker="o", color=col, linewidth=2.0,
                 markeredgecolor=T.SURFACE, markeredgewidth=1.8,
                 label=cls.replace("Reactive ", "R. "))
    ax1.set_xlabel("Bulk liquor ratio (L per kg fabric)")
    ax1.set_ylabel("Mean bulk adjustment factor")
    ax1.legend(title="Dye class", title_fontsize=9, loc="upper left")
    T.titled(ax1, "Shorter liquor needs less dye",
             "Exhaustion rises as the bath contracts; high-affinity systems are flatter")

    # --- right: realised vs theoretical exhaustion curve -------------------
    lr = np.linspace(5, 21, 200)
    for cls, col in zip(classes, cols):
        K = C.DYE_CLASSES[cls]["K"]
        ax2.plot(lr, 100.0 * K / (K + lr), color=col, linewidth=2.0,
                 label=f"{cls.replace('Reactive ', 'R. ')}  (K={K:g})")
    for s, sc in zip(C.STAGES, T.stage_colors(5)):
        m = df[f"{s}_liquor_ratio"].mean()
        ax2.axvline(m, color=T.AXIS, linewidth=1.0, linestyle=(0, (3, 3)), zorder=1)
        ax2.annotate(C.STAGE_LABELS[s].split(" (")[0], (m, 44), rotation=90,
                     fontsize=7.5, color=T.INK_MUTED, va="bottom", ha="right")
    ax2.set_xlabel("Liquor ratio (L per kg fabric)")
    ax2.set_ylabel("Equilibrium exhaustion  E = K / (K + L)   [%]")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_ylim(40, 82)
    T.titled(ax2, "The mechanism behind the correction",
             "Partition model; dashed lines mark the mean liquor ratio of each stage")

    T.source_note(fig, "Left panel: measured in the generated population. Right panel: "
                       "the analytic partition model from which the correction is derived.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig05_exhaustion_physics.png"))


# ===========================================================================
# Figure 6 - concentration windows by depth class
# ===========================================================================
def fig06_depth_windows(df: pd.DataFrame) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 4.9))
    dcol = T.depth_colors()

    for i, dc in enumerate(T.DEPTH_ORDER):
        sub = df[df["depth_class"] == dc]
        ax1.scatter(sub["labdip_total_owf"], sub["bulk_total_owf"], s=9,
                    color=dcol[dc], alpha=0.45, linewidths=0, label=dc)
    lim = [0, df["labdip_total_owf"].max() * 1.03]
    ax1.plot(lim, lim, color=T.INK_MUTED, linewidth=1.2, linestyle=(0, (4, 3)),
             zorder=5)
    ax1.annotate("parity", (lim[1] * 0.78, lim[1] * 0.80), rotation=38,
                 fontsize=8, color=T.INK_MUTED)
    ax1.set_xlabel("Lab dip total % owf")
    ax1.set_ylabel("Bulk total % owf")
    ax1.set_xlim(*lim)
    ax1.set_ylim(*lim)
    ax1.legend(title="Depth class", title_fontsize=9, loc="upper left",
               markerscale=2.2)
    T.titled(ax1, "Bulk sits below lab dip, consistently",
             "Every depth class falls under the parity line, by a depth-dependent margin")

    # salt and soda demand by depth class
    xb = np.arange(len(T.DEPTH_ORDER))
    salt = [df.loc[df["depth_class"] == d, "bulk_salt_g_l"].mean() for d in T.DEPTH_ORDER]
    soda = [df.loc[df["depth_class"] == d, "bulk_soda_g_l"].mean() for d in T.DEPTH_ORDER]
    w = 0.38
    ax2.bar(xb - w / 2 - 0.01, salt, w, color=T.SERIES[0], edgecolor=T.SURFACE,
            linewidth=2.0, label="Glauber's salt")
    ax2.bar(xb + w / 2 + 0.01, soda, w, color=T.SERIES[1], edgecolor=T.SURFACE,
            linewidth=2.0, label="Soda ash")
    for xi, (a, b) in enumerate(zip(salt, soda)):
        ax2.text(xi - w / 2 - 0.01, a + 1.0, f"{a:.0f}", ha="center", fontsize=8.5,
                 color=T.INK_2, fontweight="bold")
        ax2.text(xi + w / 2 + 0.01, b + 1.0, f"{b:.0f}", ha="center", fontsize=8.5,
                 color=T.INK_2, fontweight="bold")
    ax2.set_xticks(xb, [d.replace("/", "/\n") for d in T.DEPTH_ORDER], fontsize=8.5)
    ax2.set_ylabel("Mean bulk concentration (g/L)")
    ax2.legend(loc="upper left")
    T.titled(ax2, "Electrolyte demand scales with depth",
             "Salt dosing is the single largest chemical load in reactive dyeing")

    T.source_note(fig, "Source: textile_dyeing_scaleup_master.csv (n = 8,000).")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig06_depth_concentration_windows.png"))


# ===========================================================================
def print_eda_summary(df: pd.DataFrame) -> None:
    print("\n--- Dataset shape -------------------------------------------------")
    print(f"records: {len(df):,}   columns: {df.shape[1]}")
    print(f"distinct shades: {df['shade_code'].nunique()}   "
          f"families: {df['shade_family'].nunique()}   "
          f"customers: {df['customer'].nunique()}")

    print("\n--- Recipe contraction along the ladder ---------------------------")
    for s in C.STAGES:
        print(f"  {C.STAGE_LABELS[s]:<24s} LR {df[f'{s}_liquor_ratio'].mean():5.2f}   "
              f"adj {df[f'{s}_adj_factor'].mean():.4f}   "
              f"%owf {df[f'{s}_total_owf'].mean():.4f}   "
              f"trials {df[f'{s}_trials'].mean():.2f}")

    print("\n--- Baseline operating performance --------------------------------")
    print(f"  mean attempts per stage      {df[[f'{s}_trials' for s in C.STAGES]].to_numpy().mean():.3f}")
    print(f"  mean corrective trials/job   {df['corrective_trials'].mean():.3f}")
    print(f"  bulk right-first-time        {df['rft_bulk'].mean() * 100:.1f}%")
    print(f"  all-stage right-first-time   {df['rft_all_stages'].mean() * 100:.1f}%")
    print(f"  mean realised cost per job   ${df['total_cost_actual_usd'].mean():,.2f}")
    print(f"  mean cost of poor quality    ${df['cost_of_poor_quality_usd'].mean():,.2f}"
          f"  ({df['cost_of_poor_quality_usd'].sum() / df['total_cost_actual_usd'].sum() * 100:.1f}% of spend)")
    print(f"  mean water per job           {df['total_water_l_actual'].mean():,.0f} L")
    print(f"  mean machine hours per job   {df['total_machine_hours_actual'].mean():.1f} h")

    print("\n--- Strongest correlates of the bulk recipe -----------------------")
    num = df.select_dtypes(include=[np.number])
    cor = num.corr(numeric_only=True)["bulk_total_owf"].drop(
        [c for c in num.columns if "total_owf" in c or "dye_mass" in c],
        errors="ignore").sort_values(key=abs, ascending=False).head(10)
    print(cor.round(3).to_string())


def main() -> None:
    print("=" * 78)
    print("STEP 2  Exploratory data analysis & publication figures")
    print("=" * 78)
    T.apply_theme()
    df = load_master()
    print_eda_summary(df)

    for fn in (fig01_recipe_shift, fig02_correlation, fig03_trials,
               fig04_cost, fig05_physics, fig06_depth_windows):
        fn(df)
        print(f"[eda] {fn.__name__} done")

    print(f"\n[eda] figures written to {os.path.relpath(C.FIG_DIR, C.ROOT)}")
    print(f"[eda] companion tables written to {os.path.relpath(C.TBL_DIR, C.ROOT)}")


if __name__ == "__main__":
    main()
