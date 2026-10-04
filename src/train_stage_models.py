"""
train_stage_models.py
=====================
Trains and persists the per-stage prediction models that the Streamlit
prototype serves.

The benchmark in ml_pipeline.py answers "which model family is best?". This
script takes that answer and fits the production artefact: four independent
models, each mapping the LAB DIP plus the planned conditions of one stage onto
that stage's recipe.

Why four direct models rather than the chained sequential pipeline
------------------------------------------------------------------
The chain compounds its own error at every link, and in the benchmark it loses
to the direct mapping at bulk for exactly that reason. For a tool a dyer uses
to issue a recipe, each stage is predicted from the one piece of information
that is actually measured and approved - the lab dip - rather than from a
prediction of a prediction.

Artefact: outputs/models/stage_models.joblib
"""

from __future__ import annotations

import os

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import train_test_split

try:
    from . import config as C
    from . import ml_pipeline as M
except ImportError:
    import config as C
    import ml_pipeline as M


def pick_champion() -> str:
    """Read the champion family from the benchmark, falling back sensibly."""
    path = os.path.join(C.TBL_DIR, "model_benchmark_summary.csv")
    zoo = M.model_zoo(0)
    if os.path.exists(path):
        s = pd.read_csv(path)
        s = s[s["track"] == "direct"].sort_values("recipe_rel_error_pct")
        for name in s["model"]:
            if name in zoo:
                return name
    for fallback in ("CatBoost", "XGBoost", "LightGBM", "Random Forest"):
        if fallback in zoo:
            return fallback
    return "Random Forest"


def main() -> dict:
    print("=" * 78)
    print("Training per-stage production models for the Streamlit prototype")
    print("=" * 78)
    df = pd.read_csv(C.master_path())
    champion = pick_champion()
    est, params, sf, st = M.model_zoo(0)[champion]
    print(f"champion family: {champion}")

    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=C.TEST_SIZE,
        random_state=C.RANDOM_SEED, shuffle=True)

    bundle = {"champion": champion, "stages": {}}
    report = []
    for stage in C.STAGES[1:]:
        X, cat_cols, num_cols = M.build_features(df, "labdip", stage)
        targets = C.stage_targets(stage)
        y = df[targets].to_numpy(float)

        fitted, metrics, _, _, secs = M.fit_and_score(
            f"{champion}:{stage}", clone(est), params, sf, st,
            X.iloc[idx_tr], y[idx_tr], X.iloc[idx_te], y[idx_te],
            cat_cols, num_cols, targets, tune=False)

        recipe = metrics[~metrics["target"].str.endswith("trials")]
        dyes = metrics[metrics["target"].str.contains("dye")]
        print(f"  {C.STAGE_LABELS[stage]:<24s} R2_recipe {recipe['R2'].mean():.4f}   "
              f"MAPE_dyes {dyes['MAPE_pct'].mean():.2f}%   ({secs:.1f}s)")

        bundle["stages"][stage] = {
            "model": fitted, "targets": targets,
            "cat_cols": cat_cols, "num_cols": num_cols,
            "r2_recipe": float(recipe["R2"].mean()),
            "mape_dyes_pct": float(dyes["MAPE_pct"].mean()),
            "mae_dyes": float(dyes["MAE"].mean()),
        }
        report.append({"stage": stage, "stage_label": C.STAGE_LABELS[stage],
                       "r2_recipe": recipe["R2"].mean(),
                       "mape_dyes_pct": dyes["MAPE_pct"].mean(),
                       "mae_dyes_owf": dyes["MAE"].mean()})

    # Reference statistics the app uses for its confidence read-out.
    bundle["reference"] = {
        "mean_adj_factor": {s: float(df[f"{s}_adj_factor"].mean())
                            for s in C.STAGES},
        "tolerance": C.RECIPE_TOLERANCE_REL,
        "n_training_jobs": int(len(idx_tr)),
    }

    path = os.path.join(C.MODEL_DIR, "stage_models.joblib")
    joblib.dump(bundle, path)
    C.write_csv(pd.DataFrame(report).round(5),
                os.path.join(C.TBL_DIR, "stage_model_accuracy.csv"))
    print(f"\nsaved {path}")
    return bundle


if __name__ == "__main__":
    main()
