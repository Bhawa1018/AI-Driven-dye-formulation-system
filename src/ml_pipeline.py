"""
ml_pipeline.py  --  STEPS 3 & 4 of the research pipeline
========================================================
Multi-stage machine-learning pipeline, benchmarking and validation.

Two prediction tracks
---------------------
DIRECT      Lab dip + substrate + planned bulk conditions  ->  bulk recipe.
            One multi-output model jumps the whole ladder in a single step.

SEQUENTIAL  Lab dip -> 1 kg -> 7 kg -> Mini Bulk -> Bulk.
            Four chained multi-output models, each predicting the next stage
            from the previous stage's approved recipe. At inference the chain
            is run on its OWN predictions, so the reported bulk error includes
            accumulated error propagation - this is the honest comparison
            against the direct track, and the reason the direct track usually
            wins.

Targets (both tracks, per stage)
--------------------------------
    [dye1_owf, dye2_owf, dye3_owf, dye4_owf, salt_g_l, soda_g_l, trials]

Model families benchmarked
--------------------------
    1. Linear       - OLS, Ridge, Lasso
    2. Decision Tree
    3. Random Forest
    4. Gradient boosting - XGBoost, LightGBM, CatBoost (whichever are installed)
    5. MLP (artificial neural network)

Leakage control
---------------
Features are chosen by an explicit WHITELIST, never by dropping columns. The
build is then audited: any feature naming a stage at or beyond the target
stage, any adjustment factor, any cost/trial/outcome column, and the latent
difficulty score are rejected with an error. A model that can see
``bulk_adj_factor`` can reconstruct the answer exactly, so this matters more
than any hyper-parameter.
"""

from __future__ import annotations

import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import make_scorer

try:
    from . import config as C
    from . import viz_theme as T
except ImportError:
    import config as C
    import viz_theme as T

warnings.filterwarnings("ignore")

# Optional boosters / explainers -------------------------------------------
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
try:
    from lightgbm import LGBMRegressor
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False
try:
    from catboost import CatBoostRegressor
    HAS_CAT = True
except ImportError:
    HAS_CAT = False
try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

# Search budget. Raise for a final publication run; these values keep a full
# benchmark inside a few minutes on a laptop.
SEARCH_ITER = 8
SEARCH_CV = 3


# ===========================================================================
# 1. Feature contract
# ===========================================================================
STATIC_CATEGORICAL = [
    "shade_family", "depth_class", "fabric_structure", "composition",
    "pretreatment", "special_finish", "dye_class", "customer",
]
STATIC_NUMERIC = [
    "gsm", "cellulose_fraction", "dye_partition_K", "n_dyes", "is_repeat_num",
]

# Conditions of the TARGET stage that are known before the batch is dyed: the
# production plan fixes the machine, the batch size, the liquor ratio and the
# agitation. These are inputs, not outcomes.
def planned_conditions(stage: str) -> tuple:
    cat = [f"{stage}_machine"]
    num = [f"{stage}_batch_kg", f"{stage}_liquor_ratio", f"{stage}_rpm"]
    return cat, num


# Recipe actually approved at a SOURCE stage (the information carried forward).
def source_recipe(stage: str) -> list:
    return [f"{stage}_dye{j}_owf" for j in range(1, 5)] + [
        f"{stage}_total_owf", f"{stage}_salt_g_l", f"{stage}_soda_g_l",
        f"{stage}_temp_c", f"{stage}_time_min", f"{stage}_liquor_ratio",
        f"{stage}_rpm",
    ]


BANNED_SUBSTRINGS = (
    "adj_factor", "cost", "trials", "rft", "difficulty_score",
    "dye_mass", "salt_mass", "soda_mass", "bath_volume", "machine_hours",
    "water_l",
)


def build_features(df: pd.DataFrame, source_stage: str, target_stage: str):
    """
    Assemble the design matrix for 'predict target_stage from source_stage'.

    Returns (X, categorical_cols, numeric_cols). Raises on any leaked column.
    """
    src_cat, src_num = planned_conditions(target_stage)
    cat_cols = STATIC_CATEGORICAL + src_cat + [f"{source_stage}_machine"]
    num_cols = STATIC_NUMERIC + src_num + source_recipe(source_stage)

    # --- audit -------------------------------------------------------------
    order = {s: i for i, s in enumerate(C.STAGES)}
    t_idx = order[target_stage]
    for col in cat_cols + num_cols:
        for bad in BANNED_SUBSTRINGS:
            if bad in col:
                raise ValueError(f"LEAK: outcome column '{col}' in feature set")
        for s, i in order.items():
            if col.startswith(f"{s}_") and i >= t_idx:
                # the only columns allowed from the target stage are the
                # planned conditions declared above
                if col not in src_cat + src_num:
                    raise ValueError(
                        f"LEAK: '{col}' belongs to stage '{s}' at or beyond the "
                        f"target stage '{target_stage}'")

    work = df.copy()
    work["is_repeat_num"] = work["is_repeat"].astype(int)
    missing = [c for c in cat_cols + num_cols if c not in work.columns]
    if missing:
        raise KeyError(f"missing feature columns: {missing}")
    return work[cat_cols + num_cols], cat_cols, num_cols


def make_preprocessor(cat_cols, num_cols, scale: bool = False) -> ColumnTransformer:
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                                 min_frequency=10)),
    ])
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("cat", cat_pipe, cat_cols), ("num", Pipeline(num_steps), num_cols)],
        remainder="drop", verbose_feature_names_out=False)


# ===========================================================================
# 2. Metrics
# ===========================================================================
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                       names: list) -> pd.DataFrame:
    """
    Per-target R2, MAE, MSE, RMSE and MAPE.

    MAPE is undefined where the truth is zero, which happens legitimately for
    dye slots a shade does not use (a 2-dye shade has dye3 = dye4 = 0). It is
    therefore computed only over rows with a non-trivial true value, and the
    share of rows used is reported alongside so the figure cannot be read as
    covering the whole test set.
    """
    rows = []
    eps = 1e-6
    for j, name in enumerate(names):
        yt = np.asarray(y_true)[:, j].astype(float)
        yp = np.asarray(y_pred)[:, j].astype(float)
        err = yp - yt
        ss_res = float(np.sum(err ** 2))
        ss_tot = float(np.sum((yt - yt.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > eps else np.nan
        mae = float(np.mean(np.abs(err)))
        mse = float(np.mean(err ** 2))
        mask = np.abs(yt) > max(eps, 1e-4 * (np.abs(yt).max() or 1.0))
        mape = (float(np.mean(np.abs(err[mask] / yt[mask])) * 100.0)
                if mask.sum() else np.nan)
        rows.append({
            "target": name, "R2": r2, "MAE": mae, "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "MAPE_pct": mape,
            "MAPE_coverage_pct": 100.0 * mask.mean(),
        })
    return pd.DataFrame(rows)


def recipe_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Concentration-weighted relative error of the four-dye recipe.

    This is the quantity that decides whether a batch is accepted: a shade is
    re-dyed when the issued recipe sits outside the colour tolerance, and that
    tolerance is relative, not absolute. Weighting by the true total also
    stops a trace component present at 0.01 %owf from dominating the score.

    It is used BOTH as the hyper-parameter search objective and as the champion
    criterion, so tuning, selection and the economic analysis in Step 5 all
    optimise the same thing. Selecting on R2 instead measurably degrades it:
    R2 rewards getting the large concentrations right, which is not the same
    as keeping every concentration inside a relative tolerance band.
    """
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    t, p = y_true[:, :4], y_pred[:, :4]
    denom = t.sum(axis=1)
    mask = denom > 1e-9
    if not mask.any():
        return np.nan
    return float(np.mean(np.abs(p[mask] - t[mask]).sum(axis=1) / denom[mask]))


RECIPE_SCORER = make_scorer(recipe_relative_error, greater_is_better=False)


def attainable_r2_for_counts(y: np.ndarray, driver: np.ndarray | None = None,
                             n_bins: int = 25) -> dict:
    """
    Upper bound on R2 for a trial count generated as 1 + min(Poisson(lambda), c).

    Any model can only recover the variance of the conditional mean; the rest
    is shot noise no model can predict. Two estimators are returned:

    'moment'  Var(lambda)/Var(y) via Var(y) = E[lambda] + Var(lambda). Exact
              for an UNCAPPED Poisson, but the generator caps the draw at
              TRIAL_MAX, which shrinks Var(y) and can drive this estimate to
              zero or below. It is reported for transparency, not relied upon.

    'binned'  Var(E[y | driver]) / Var(y), with the conditional mean estimated
              by quantile-binning the driver. This makes no distributional
              assumption and tolerates the cap, so it is the figure quoted.

    Reporting a raw R2 near zero for this target without such a bound would
    misreport an irreducible property of the process as a modelling failure.
    """
    y = np.asarray(y, dtype=float)
    var_y = float(y.var())
    if var_y <= 1e-12:
        return {"moment": np.nan, "binned": np.nan, "var_y": var_y}

    moment = (var_y - float(y.mean() - 1.0)) / var_y

    binned = np.nan
    if driver is not None:
        d = np.asarray(driver, dtype=float)
        edges = np.unique(np.quantile(d, np.linspace(0, 1, n_bins + 1)))
        if len(edges) > 2:
            idx = np.clip(np.digitize(d, edges[1:-1]), 0, len(edges) - 2)
            means = np.array([y[idx == b].mean() if (idx == b).any() else y.mean()
                              for b in range(len(edges) - 1)])
            binned = float(means[idx].var() / var_y)
    return {"moment": float(moment), "binned": binned, "var_y": var_y}


def summarise(metrics: pd.DataFrame, model: str, track: str) -> dict:
    """Collapse a per-target table into one benchmark row."""
    dye_rows = metrics[metrics["target"].str.contains("dye")]
    # The recipe targets are what the platform actually issues to the dye-house;
    # the trial count is a stochastic outcome handled separately in Step 5, so
    # it must not drive champion selection.
    recipe_rows = metrics[~metrics["target"].str.endswith("trials")]
    return {
        "model": model,
        "track": track,
        # The decision metric: concentration-weighted relative recipe error and
        # the share of jobs it pushes outside the colour-acceptance tolerance.
        "recipe_rel_error_pct": float(metrics["recipe_rel_error_pct"].iloc[0])
        if "recipe_rel_error_pct" in metrics else np.nan,
        "off_tolerance_pct": float(metrics["off_tolerance_pct"].iloc[0])
        if "off_tolerance_pct" in metrics else np.nan,
        "R2_recipe": recipe_rows["R2"].mean(),
        "MAE_recipe": recipe_rows["MAE"].mean(),
        "RMSE_recipe": recipe_rows["RMSE"].mean(),
        "MAPE_recipe_pct": recipe_rows["MAPE_pct"].mean(),
        "R2_macro": metrics["R2"].mean(),
        "MAE_macro": metrics["MAE"].mean(),
        "MSE_macro": metrics["MSE"].mean(),
        "RMSE_macro": metrics["RMSE"].mean(),
        "MAPE_macro_pct": metrics["MAPE_pct"].mean(),
        "R2_dyes": dye_rows["R2"].mean(),
        "MAE_dyes": dye_rows["MAE"].mean(),
        "MAPE_dyes_pct": dye_rows["MAPE_pct"].mean(),
        "R2_trials": float(metrics.loc[metrics["target"].str.endswith("trials"),
                                       "R2"].iloc[0]),
    }


# ===========================================================================
# 3. Model zoo
# ===========================================================================
def model_zoo(n_features: int) -> dict:
    """
    name -> (estimator, param distribution, scale_features, scale_targets).

    Estimators that support multi-output natively are used directly and get
    target standardisation (see ScaledTargetRegressor for why this is
    required, not optional). The boosters are wrapped in MultiOutputRegressor,
    which fits one model per target and so needs neither.
    """
    zoo: dict = {}

    zoo["Linear (OLS)"] = (LinearRegression(), {}, True, True)
    zoo["Ridge"] = (Ridge(),
                    {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}, True, True)
    zoo["Lasso"] = (Lasso(max_iter=8000),
                    {"model__alpha": [1e-4, 1e-3, 1e-2, 1e-1]}, True, True)

    zoo["Decision Tree"] = (
        DecisionTreeRegressor(random_state=C.RANDOM_SEED),
        {"model__max_depth": [6, 10, 14, 20, None],
         "model__min_samples_leaf": [1, 5, 10, 20],
         "model__min_samples_split": [2, 10, 20]},
        False, True)

    zoo["Random Forest"] = (
        RandomForestRegressor(n_estimators=300, random_state=C.RANDOM_SEED,
                              n_jobs=-1),
        {"model__max_depth": [12, 18, 24, None],
         "model__min_samples_leaf": [1, 2, 4],
         "model__max_features": ["sqrt", 0.4, 0.7]},
        False, True)

    if HAS_XGB:
        zoo["XGBoost"] = (
            MultiOutputRegressor(XGBRegressor(
                n_estimators=400, learning_rate=0.06, max_depth=6,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.0,
                tree_method="hist", random_state=C.RANDOM_SEED, n_jobs=-1,
                verbosity=0)),
            {"model__estimator__max_depth": [4, 6, 8],
             "model__estimator__learning_rate": [0.03, 0.06, 0.12],
             "model__estimator__subsample": [0.7, 0.85, 1.0],
             "model__estimator__n_estimators": [300, 500]},
            False, False)

    if HAS_LGBM:
        zoo["LightGBM"] = (
            MultiOutputRegressor(LGBMRegressor(
                n_estimators=500, learning_rate=0.06, num_leaves=31,
                subsample=0.85, colsample_bytree=0.85,
                random_state=C.RANDOM_SEED, n_jobs=-1, verbose=-1)),
            {"model__estimator__num_leaves": [15, 31, 63],
             "model__estimator__learning_rate": [0.03, 0.06, 0.12],
             "model__estimator__n_estimators": [300, 500]},
            False, False)

    if HAS_CAT:
        zoo["CatBoost"] = (
            MultiOutputRegressor(CatBoostRegressor(
                iterations=400, learning_rate=0.08, depth=6,
                random_seed=C.RANDOM_SEED, verbose=0, allow_writing_files=False)),
            {"model__estimator__depth": [4, 6, 8],
             "model__estimator__learning_rate": [0.04, 0.08, 0.15]},
            False, False)

    zoo["MLP (ANN)"] = (
        MLPRegressor(hidden_layer_sizes=(256, 128, 64), activation="relu",
                     solver="adam", alpha=1e-4, learning_rate_init=1e-3,
                     max_iter=600, early_stopping=True, n_iter_no_change=20,
                     random_state=C.RANDOM_SEED),
        {"model__hidden_layer_sizes": [(128, 64), (256, 128, 64), (512, 256)],
         "model__alpha": [1e-5, 1e-4, 1e-3],
         "model__learning_rate_init": [5e-4, 1e-3, 3e-3]},
        True, True)

    return zoo


from sklearn.base import BaseEstimator, RegressorMixin, clone as _clone


class ScaledTargetRegressor(RegressorMixin, BaseEstimator):
    """
    Standardises the multi-output target before fitting and inverts afterwards.

    This is NOT cosmetic. The target vector mixes dye concentrations
    (~0.1-7 %owf), electrolyte concentrations (~5-95 g/L) and a trial count
    (1-7). Any learner that minimises an unweighted squared error across the
    whole vector will therefore spend almost all of its capacity on the salt
    column and effectively ignore the dye concentrations - which are the
    outputs the platform exists to produce.

    It applies to every estimator that handles multi-output NATIVELY, trees
    included: a multi-output decision tree selects splits on summed variance
    reduction across targets, so it is scale-sensitive in the target even
    though it is scale-invariant in the features. Benchmarking trees without
    this understates them severely. Boosters wrapped in MultiOutputRegressor
    fit one model per target and are unaffected either way.
    """

    def __init__(self, estimator=None):
        self.estimator = estimator

    def fit(self, X, y):
        self.estimator_ = _clone(self.estimator)
        return self._fit(X, y)

    def _fit(self, X, y):
        y = np.asarray(y, dtype=float)
        self.mu_ = y.mean(axis=0)
        self.sigma_ = y.std(axis=0)
        self.sigma_[self.sigma_ < 1e-9] = 1.0
        self.estimator_.fit(X, (y - self.mu_) / self.sigma_)
        return self

    def predict(self, X):
        return self.estimator_.predict(X) * self.sigma_ + self.mu_


# ===========================================================================
# 4. Training / benchmarking one track
# ===========================================================================
def fit_and_score(name, estimator, params, scale_features, scale_targets,
                  X_tr, y_tr, X_te, y_te, cat_cols, num_cols, target_names,
                  tune: bool = True):
    """Fit one model family (optionally tuned) and score it on the test set."""
    pre = make_preprocessor(cat_cols, num_cols, scale=scale_features)
    core = ScaledTargetRegressor(estimator) if scale_targets else estimator
    pipe = Pipeline([("pre", pre), ("model", core)])

    # ScaledTargetRegressor hides the estimator one level deeper.
    if scale_targets and params:
        params = {k.replace("model__", "model__estimator__"): v
                  for k, v in params.items()}

    t0 = time.time()
    if tune and params:
        search = RandomizedSearchCV(
            pipe, params, n_iter=min(SEARCH_ITER,
                                     int(np.prod([len(v) for v in params.values()]))),
            cv=KFold(SEARCH_CV, shuffle=True, random_state=C.RANDOM_SEED),
            scoring=RECIPE_SCORER, n_jobs=-1, random_state=C.RANDOM_SEED,
            error_score="raise", refit=True)
        search.fit(X_tr, y_tr)
        fitted = search.best_estimator_
        best = search.best_params_
    else:
        pipe.fit(X_tr, y_tr)
        fitted = pipe
        best = {}
    fit_s = time.time() - t0

    pred = np.asarray(fitted.predict(X_te), dtype=float)
    # Physical constraints: concentrations are non-negative, trials >= 1.
    pred[:, :6] = np.clip(pred[:, :6], 0.0, None)
    pred[:, 6] = np.clip(pred[:, 6], 1.0, C.TRIAL_MAX)

    y_te_arr = np.asarray(y_te, dtype=float)
    metrics = regression_metrics(y_te_arr, pred, target_names)
    # Attach the application-level metric to every row so it survives the
    # concat into the benchmark table.
    rel = recipe_relative_error(y_te_arr, pred)
    metrics["recipe_rel_error_pct"] = rel * 100.0
    metrics["off_tolerance_pct"] = _off_tolerance(y_te_arr, pred) * 100.0
    return fitted, metrics, pred, best, fit_s


def _off_tolerance(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Share of jobs whose predicted recipe falls outside the colour tolerance."""
    t, p = np.asarray(y_true, float)[:, :4], np.asarray(y_pred, float)[:, :4]
    denom = t.sum(axis=1)
    mask = denom > 1e-9
    if not mask.any():
        return np.nan
    err = np.abs(p[mask] - t[mask]).sum(axis=1) / denom[mask]
    return float((err > C.RECIPE_TOLERANCE_REL).mean())


def run_direct_track(df: pd.DataFrame, tune: bool = True):
    """Lab dip -> bulk, in one jump."""
    print("\n" + "=" * 78)
    print("DIRECT TRACK   lab dip + planned bulk conditions  ->  bulk recipe")
    print("=" * 78)

    X, cat_cols, num_cols = build_features(df, "labdip", "bulk")
    targets = C.TARGETS_BULK
    y = df[targets].to_numpy(dtype=float)
    print(f"features: {len(cat_cols)} categorical + {len(num_cols)} numeric   "
          f"targets: {len(targets)}")

    X_tr, X_te, y_tr, y_te, idx_tr, idx_te = train_test_split(
        X, y, np.arange(len(df)), test_size=C.TEST_SIZE,
        random_state=C.RANDOM_SEED, shuffle=True)
    print(f"train: {len(X_tr):,}   test: {len(X_te):,}")

    zoo = model_zoo(len(cat_cols) + len(num_cols))
    all_metrics, summary, fitted_models, preds = [], [], {}, {}

    for name, (est, params, sf, st) in zoo.items():
        try:
            fitted, metrics, pred, best, secs = fit_and_score(
                name, est, params, sf, st, X_tr, y_tr, X_te, y_te,
                cat_cols, num_cols, targets, tune=tune)
        except Exception as exc:
            print(f"  {name:<16s} FAILED  {type(exc).__name__}: {exc}")
            continue
        metrics.insert(0, "model", name)
        metrics.insert(1, "track", "direct")
        all_metrics.append(metrics)
        row = summarise(metrics, name, "direct")
        row["fit_seconds"] = round(secs, 1)
        row["best_params"] = json.dumps(best)
        summary.append(row)
        fitted_models[name] = fitted
        preds[name] = pred
        print(f"  {name:<16s} rel.err {row['recipe_rel_error_pct']:6.3f}%   "
              f"off-tol {row['off_tolerance_pct']:5.1f}%   "
              f"R2_recipe {row['R2_recipe']:.4f}   ({secs:.1f}s)")

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    summary_df = pd.DataFrame(summary).sort_values("recipe_rel_error_pct")
    return dict(metrics=metrics_df, summary=summary_df, models=fitted_models,
                preds=preds, X_te=X_te, y_te=y_te, idx_te=idx_te,
                targets=targets, cat_cols=cat_cols, num_cols=num_cols,
                X_tr=X_tr, y_tr=y_tr)


def run_sequential_track(df: pd.DataFrame, model_name: str, tune: bool = False):
    """
    Lab dip -> 1 kg -> 7 kg -> Mini Bulk -> Bulk, as four chained models.

    Each link is trained on TRUE upstream recipes (teacher forcing), but
    evaluated by running the chain on its own predictions, so the reported
    bulk error includes error propagation down the ladder.
    """
    print("\n" + "=" * 78)
    print(f"SEQUENTIAL TRACK   chained {model_name}, 4 links")
    print("=" * 78)

    idx_tr, idx_te = train_test_split(
        np.arange(len(df)), test_size=C.TEST_SIZE,
        random_state=C.RANDOM_SEED, shuffle=True)

    zoo = model_zoo(0)
    est, params, sf, st = zoo[model_name]

    links, link_metrics = [], []
    for src, tgt in zip(C.STAGES[:-1], C.STAGES[1:]):
        X, cat_cols, num_cols = build_features(df, src, tgt)
        targets = C.stage_targets(tgt)
        y = df[targets].to_numpy(dtype=float)

        fitted, metrics, _, _, secs = fit_and_score(
            f"{model_name}:{tgt}", _clone(est), params, sf, st,
            X.iloc[idx_tr], y[idx_tr], X.iloc[idx_te], y[idx_te],
            cat_cols, num_cols, targets, tune=tune)
        links.append((src, tgt, fitted, cat_cols, num_cols, targets))
        metrics.insert(0, "link", f"{src}->{tgt}")
        link_metrics.append(metrics)
        print(f"  link {src:>8s} -> {tgt:<9s} R2_macro {metrics['R2'].mean():.4f} "
              f"(teacher-forced)   ({secs:.1f}s)")

    # ---- run the chain on its own predictions -----------------------------
    test = df.iloc[idx_te].copy().reset_index(drop=True)
    rolling = test.copy()
    for src, tgt, fitted, cat_cols, num_cols, targets in links:
        Xs, _, _ = build_features(rolling, src, tgt)
        pred = np.asarray(fitted.predict(Xs), dtype=float)
        pred[:, :6] = np.clip(pred[:, :6], 0.0, None)
        pred[:, 6] = np.clip(pred[:, 6], 1.0, C.TRIAL_MAX)
        for j, col in enumerate(targets):
            rolling[col] = pred[:, j]
        # the chain must also carry forward the derived total it will consume
        rolling[f"{tgt}_total_owf"] = rolling[
            [f"{tgt}_dye{k}_owf" for k in range(1, 5)]].sum(axis=1)

    y_true = test[C.TARGETS_BULK].to_numpy(dtype=float)
    y_chain = rolling[C.TARGETS_BULK].to_numpy(dtype=float)
    chain_metrics = regression_metrics(y_true, y_chain, C.TARGETS_BULK)
    # Same decision metric as the direct track, so the two are comparable on the
    # quantity that actually governs acceptance.
    chain_metrics["recipe_rel_error_pct"] = recipe_relative_error(y_true, y_chain) * 100.0
    chain_metrics["off_tolerance_pct"] = _off_tolerance(y_true, y_chain) * 100.0
    chain_metrics.insert(0, "model", f"{model_name} (chained)")
    chain_metrics.insert(1, "track", "sequential")
    print(f"  CHAINED end-to-end  R2_macro {chain_metrics['R2'].mean():.4f}   "
          f"MAE_dyes {chain_metrics[chain_metrics['target'].str.contains('dye')]['MAE'].mean():.4f} %owf")

    return dict(links=links, link_metrics=pd.concat(link_metrics, ignore_index=True),
                chain_metrics=chain_metrics,
                summary=pd.DataFrame([summarise(chain_metrics,
                                                f"{model_name} (chained)",
                                                "sequential")]))


# ===========================================================================
# 5. Feature importance
# ===========================================================================
def feature_importance(result: dict, champion_name: str, n_repeats: int = 5):
    """Permutation importance on the test set, plus native/SHAP where available."""
    model = result["models"][champion_name]
    X_te, y_te = result["X_te"], result["y_te"]

    print(f"\n[importance] permutation importance for {champion_name} "
          f"({n_repeats} repeats)")
    perm = permutation_importance(
        model, X_te, y_te, n_repeats=n_repeats, random_state=C.RANDOM_SEED,
        scoring="r2", n_jobs=-1)
    imp = (pd.DataFrame({"feature": X_te.columns,
                         "permutation_importance": perm.importances_mean,
                         "std": perm.importances_std})
             .sort_values("permutation_importance", ascending=False)
             .reset_index(drop=True))
    C.write_csv(imp.round(6),
                os.path.join(C.TBL_DIR, "feature_importance_permutation.csv"))
    return imp


def shap_analysis(result: dict, champion_name: str, target: str = "bulk_dye1_owf",
                  n_sample: int = 400):
    """
    SHAP values for one representative target of the champion model.

    Only tree ensembles are explained with TreeExplainer; for anything else the
    permutation importance above is the reported attribution.
    """
    if not HAS_SHAP:
        print("[shap] shap not installed - skipped")
        return None
    model = result["models"][champion_name]
    try:
        pre = model.named_steps["pre"]
        core = model.named_steps["model"]
        if isinstance(core, MultiOutputRegressor):
            j = result["targets"].index(target)
            booster = core.estimators_[j]
        else:
            booster = core
        Xs = pre.transform(result["X_te"].iloc[:n_sample])
        names = list(pre.get_feature_names_out())
        explainer = shap.TreeExplainer(booster)
        sv = explainer.shap_values(Xs)
        if isinstance(sv, list):
            sv = sv[0]
        if sv.ndim == 3:
            sv = sv[:, :, result["targets"].index(target)]
        mean_abs = np.abs(sv).mean(axis=0)
        out = (pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
                 .sort_values("mean_abs_shap", ascending=False)
                 .reset_index(drop=True))
        C.write_csv(out.round(8),
                    os.path.join(C.TBL_DIR, f"feature_importance_shap_{target}.csv"))
        print(f"[shap] computed for {champion_name} / {target} "
              f"on {Xs.shape[0]} test rows")
        return dict(shap_values=sv, X=Xs, names=names, table=out, target=target)
    except Exception as exc:
        print(f"[shap] skipped ({type(exc).__name__}: {exc})")
        return None


# ===========================================================================
# 6. Figures for Steps 3-4
# ===========================================================================
def fig07_model_comparison(summary: pd.DataFrame) -> None:
    d = summary.sort_values("recipe_rel_error_pct", ascending=False)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.2, 5.6),
                                   gridspec_kw={"width_ratios": [1, 1]})
    y = np.arange(len(d))
    champ = d["recipe_rel_error_pct"].idxmin()

    colors = [T.SERIES[0]] * len(d)
    colors[-1] = T.STATUS["good"]          # champion highlighted
    ax1.barh(y, d["recipe_rel_error_pct"], color=colors, edgecolor=T.SURFACE,
             linewidth=2.0, height=0.74)
    ax1.set_yticks(y, d["model"], fontsize=9)
    ax1.grid(False); ax1.xaxis.grid(True)
    ax1.set_xlabel("Concentration-weighted recipe error (%)")
    ax1.axvline(C.RECIPE_TOLERANCE_REL * 100, color=T.STATUS["critical"],
                linewidth=2.0, linestyle=(0, (5, 3)), zorder=5)
    ax1.annotate(f"colour tolerance {C.RECIPE_TOLERANCE_REL * 100:.1f}%",
                 (C.RECIPE_TOLERANCE_REL * 100, -0.75),
                 textcoords="offset points", xytext=(7, 0), fontsize=8.5,
                 color=T.STATUS["critical"], fontweight="bold", va="center")
    for yi, v in zip(y, d["recipe_rel_error_pct"]):
        ax1.text(v + d["recipe_rel_error_pct"].max() * 0.012, yi, f"{v:.2f}%",
                 va="center", fontsize=8.5, color=T.INK_2, fontweight="bold")
    T.titled(ax1, "Model benchmark: accuracy",
             "The metric that decides acceptance; green is the champion")

    ax2.barh(y, d["off_tolerance_pct"], color=colors, edgecolor=T.SURFACE,
             linewidth=2.0, height=0.74)
    ax2.set_yticks(y, [""] * len(d))
    ax2.grid(False); ax2.xaxis.grid(True)
    ax2.set_xlabel("Jobs outside colour tolerance (%)")
    ax2.set_xlim(0, 104)
    for yi, v in zip(y, d["off_tolerance_pct"]):
        ax2.text(v + 1.4, yi, f"{v:.1f}%", va="center", fontsize=8.5,
                 color=T.INK_2, fontweight="bold")
    T.titled(ax2, "Model benchmark: rework exposure",
             "Share of batches a model would send back for a second dyeing")

    T.source_note(fig, "Held-out test set, 20% of 8,000 jobs. Recipe error is the "
                       "concentration-weighted relative error of the four-dye "
                       "recipe - the quantity that decides whether a batch is accepted.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig07_model_comparison.png"))


def fig08_importance(imp: pd.DataFrame, shap_res, champion: str) -> None:
    n = 2 if shap_res else 1
    fig, axes = plt.subplots(1, n, figsize=(7.4 * n, 6.0))
    axes = np.atleast_1d(axes)

    top = imp.head(16).iloc[::-1]
    y = np.arange(len(top))
    axes[0].barh(y, top["permutation_importance"], xerr=top["std"],
                 color=T.SEQ_HUE, edgecolor=T.SURFACE, linewidth=2.0,
                 height=0.74, error_kw=dict(ecolor=T.AXIS, lw=1.0, capsize=2))
    axes[0].set_yticks(y, top["feature"], fontsize=8.5)
    axes[0].grid(False); axes[0].xaxis.grid(True)
    axes[0].set_xlabel("Drop in R$^2$ when the feature is shuffled")
    T.titled(axes[0], "What the champion model relies on",
             f"Permutation importance, {champion}, held-out test set")

    if shap_res:
        tbl = shap_res["table"].head(16).iloc[::-1]
        y2 = np.arange(len(tbl))
        axes[1].barh(y2, tbl["mean_abs_shap"], color=T.SERIES[1],
                     edgecolor=T.SURFACE, linewidth=2.0, height=0.74)
        axes[1].set_yticks(y2, tbl["feature"], fontsize=8.5)
        axes[1].grid(False); axes[1].xaxis.grid(True)
        axes[1].set_xlabel("Mean |SHAP value| (%owf)")
        T.titled(axes[1], "Attribution for the primary dye",
                 f"SHAP, target = {shap_res['target']}")

    T.source_note(fig, "Permutation importance is model-agnostic and measured on "
                       "held-out data; SHAP decomposes individual predictions.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig08_feature_importance.png"))


def fig09_pred_vs_actual(result: dict, champion: str) -> None:
    pred = result["preds"][champion]
    y_te = result["y_te"]
    show = ["bulk_dye1_owf", "bulk_salt_g_l", "bulk_soda_g_l", "bulk_trials"]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.3))
    for ax, name in zip(axes, show):
        j = result["targets"].index(name)
        yt, yp = y_te[:, j], pred[:, j]
        ax.scatter(yt, yp, s=7, color=T.SERIES[0], alpha=0.35, linewidths=0)
        lo, hi = float(min(yt.min(), yp.min())), float(max(yt.max(), yp.max()))
        ax.plot([lo, hi], [lo, hi], color=T.INK_MUTED, linewidth=1.3,
                linestyle=(0, (4, 3)), zorder=5)
        ss = 1 - np.sum((yp - yt) ** 2) / np.sum((yt - yt.mean()) ** 2)
        ax.set_title(f"{name}\nR$^2$ = {ss:.4f}", loc="left", fontsize=10,
                     color=T.INK, fontweight="bold")
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
    T.source_note(fig, f"Champion model: {champion}. Dashed line is perfect prediction.")
    fig.tight_layout()
    T.save(fig, os.path.join(C.FIG_DIR, "fig09_pred_vs_actual.png"))


# ===========================================================================
# 7. Main
# ===========================================================================
def main(tune: bool = True) -> dict:
    print("=" * 78)
    print("STEPS 3 & 4  Multi-stage ML pipeline, benchmarking and validation")
    print("=" * 78)
    T.apply_theme()
    df = pd.read_csv(C.master_path())
    print(f"loaded {len(df):,} jobs")
    print(f"boosters available -> XGBoost:{HAS_XGB}  LightGBM:{HAS_LGBM}  "
          f"CatBoost:{HAS_CAT}   SHAP:{HAS_SHAP}")

    # ---- direct track -----------------------------------------------------
    direct = run_direct_track(df, tune=tune)
    champion = direct["summary"].iloc[0]["model"]
    print(f"\n*** CHAMPION (direct track): {champion} "
          f"recipe error = {direct['summary'].iloc[0]['recipe_rel_error_pct']:.3f}% ***")

    # ---- sequential track -------------------------------------------------
    seq_name = champion if champion in model_zoo(0) else "Random Forest"
    sequential = run_sequential_track(df, seq_name, tune=False)

    # ---- tables -----------------------------------------------------------
    C.write_csv(direct["metrics"].round(6),
                os.path.join(C.TBL_DIR, "model_benchmark_direct_per_target.csv"))
    C.write_csv(sequential["link_metrics"].round(6),
                os.path.join(C.TBL_DIR, "model_benchmark_sequential_links.csv"))
    combined = pd.concat([direct["summary"], sequential["summary"]],
                         ignore_index=True)
    C.write_csv(combined.round(6),
                os.path.join(C.TBL_DIR, "model_benchmark_summary.csv"))

    print("\n--- Benchmark summary (sorted by recipe error) ------------------------")
    cols = ["model", "track", "recipe_rel_error_pct", "off_tolerance_pct",
            "R2_recipe", "MAE_recipe", "RMSE_recipe", "MAPE_dyes_pct",
            "R2_dyes", "R2_trials"]
    print(combined[cols].round(4).to_string(index=False))

    # --- how predictable is the trial count, really? -----------------------
    j = direct["targets"].index("bulk_trials")
    bound = attainable_r2_for_counts(
        df["bulk_trials"].to_numpy(), df["difficulty_score"].to_numpy())
    r2_got = float(direct["summary"].iloc[0]["R2_trials"])
    print("\n--- Trial count: attainable vs achieved --------------------------")
    print("  bulk_trials is generated as 1 + min(Poisson(lambda), cap), so only")
    print("  the variance of the conditional mean is reducible at all.")
    print(f"    attainable R2 (binned estimator) : {bound['binned']:.4f}")
    print(f"    attainable R2 (moment estimator) : {bound['moment']:.4f}"
          "   [biased low by the cap]")
    print(f"    champion achieved R2             : {r2_got:.4f}")
    print("  A single batch's count is therefore close to unpredictable; what")
    print("  the platform shifts is the RATE, which Step 5 derives from recipe")
    print("  accuracy rather than from this regression target.")
    C.write_csv(pd.DataFrame([{
        "target": "bulk_trials",
        "attainable_r2_binned": bound["binned"],
        "attainable_r2_moment": bound["moment"],
        "achieved_r2": r2_got,
        "variance_of_target": bound["var_y"],
    }]).round(6), os.path.join(C.TBL_DIR, "trial_count_predictability.csv"))

    # ---- importance & explanation ----------------------------------------
    imp = feature_importance(direct, champion)
    print("\n--- Top 12 features ----------------------------------------------")
    print(imp.head(12).round(5).to_string(index=False))
    shap_res = shap_analysis(direct, champion)

    # ---- figures ----------------------------------------------------------
    fig07_model_comparison(direct["summary"])
    fig08_importance(imp, shap_res, champion)
    fig09_pred_vs_actual(direct, champion)
    print(f"\n[ml] figures written to {os.path.relpath(C.FIG_DIR, C.ROOT)}")

    # ---- persist champion + residuals for the economic analysis -----------
    joblib.dump({
        "model": direct["models"][champion],
        "champion": champion,
        "targets": direct["targets"],
        "cat_cols": direct["cat_cols"],
        "num_cols": direct["num_cols"],
    }, os.path.join(C.MODEL_DIR, "champion_direct.joblib"))

    pred = direct["preds"][champion]
    resid = pd.DataFrame({"job_index": direct["idx_te"]})
    for j, name in enumerate(direct["targets"]):
        resid[f"{name}_true"] = direct["y_te"][:, j]
        resid[f"{name}_pred"] = pred[:, j]
    C.write_csv(resid.round(6),
                os.path.join(C.TBL_DIR, "champion_test_predictions.csv"))
    print(f"[ml] champion model + test predictions saved")

    return dict(direct=direct, sequential=sequential, champion=champion,
                summary=combined, importance=imp)


if __name__ == "__main__":
    import sys
    main(tune="--no-tune" not in sys.argv)
