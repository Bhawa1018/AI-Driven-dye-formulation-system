"""
app.py  --  STEP 6 of the research pipeline
===========================================
Interactive prototype of the AI-Driven Dye Formulation & Scale-Up Platform.

Run with:
    streamlit run app.py

Four sections:
    Recipe Predictor   issue the recipe for every stage from the approved lab dip
    Scale-Up Explorer  interrogate the 8,000-programme population
    Model Performance  the benchmark behind the predictions
    Cost & ROI         factory economics under your own assumptions

Every number shown is either a model prediction or an explicit arithmetic
consequence of one. Nothing is a placeholder. If the trained artefacts are
missing the app says so and stops, rather than showing plausible fiction.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config as C                                    # noqa: E402
import viz_theme as T                                 # noqa: E402
from shade_master import build_shade_master           # noqa: E402
from data_generator import attempt_cost               # noqa: E402

MODEL_PATH = os.path.join(C.MODEL_DIR, "stage_models.joblib")

st.set_page_config(page_title="Dye Scale-Up Platform", page_icon="🎨",
                   layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------------------
# Styling - the same tokens as the printed figures, so the deliverables match
# ---------------------------------------------------------------------------
st.markdown(f"""
<style>
  .stApp {{ background: {T.PAGE}; }}
  .block-container {{ padding-top: 1.6rem; max-width: 1560px; }}
  h1, h2, h3 {{ color: {T.INK}; letter-spacing: -0.015em; }}
  h1 {{ font-size: 1.9rem; margin-bottom: .1rem; }}

  .kpi {{
      background: {T.SURFACE}; border: 1px solid {T.GRID};
      border-radius: 12px; padding: 15px 17px; height: 100%;
  }}
  .kpi .label {{
      font-size: 0.72rem; text-transform: uppercase; letter-spacing: .07em;
      color: {T.INK_MUTED}; font-weight: 600; margin-bottom: 5px;
  }}
  .kpi .value {{ font-size: 1.72rem; font-weight: 700; color: {T.INK};
                 line-height: 1.12; }}
  .kpi .sub {{ font-size: 0.78rem; color: {T.INK_2}; margin-top: 5px;
               line-height: 1.35; }}

  .note {{ color: {T.INK_MUTED}; font-size: 0.82rem; line-height: 1.5; }}
  .banner {{
      background: {T.SURFACE}; border: 1px solid {T.GRID};
      border-left: 4px solid {T.SERIES[0]};
      border-radius: 10px; padding: 12px 16px; margin: 6px 0 16px 0;
      color: {T.INK_2}; font-size: 0.86rem; line-height: 1.5;
  }}
  .banner.warn {{ border-left-color: {T.STATUS['warning']}; }}

  div[data-testid="stDataFrame"] {{ border: 1px solid {T.GRID};
                                    border-radius: 10px; }}
  .stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {T.GRID}; }}
  .stTabs [data-baseweb="tab"] {{
      height: 42px; padding: 0 18px; background: transparent;
      font-weight: 600; font-size: 0.92rem; color: {T.INK_MUTED};
  }}
  .stTabs [aria-selected="true"] {{ color: {T.INK}; }}
</style>
""", unsafe_allow_html=True)

PLOT_LAYOUT = dict(
    paper_bgcolor=T.SURFACE, plot_bgcolor=T.SURFACE,
    font=dict(color=T.INK_2, size=12),
    margin=dict(l=10, r=10, t=10, b=10),
)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    """
    Load the per-stage models, training them on the spot if they are absent.

    The trained artefacts are ~55 MB and, more importantly, are pickles: an
    unpickle is only valid against the library versions that produced it, which
    is the most common way a working app dies on a hosting platform. So they
    are deliberately NOT deployed. On a fresh host the app rebuilds them from
    the committed dataset in about 20 seconds, once, and Streamlit caches the
    result for every later visitor.
    """
    import joblib
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as exc:        # version drift, truncated file, ...
            st.warning(f"Stored models could not be loaded "
                       f"({type(exc).__name__}); retraining from the dataset.")

    try:
        C.master_path()
    except FileNotFoundError:
        return None

    with st.spinner("First run on this server: training the scale-up models "
                    "(about 30 seconds). This happens once."):
        import train_stage_models
        return train_stage_models.main()


@st.cache_data(show_spinner=False)
def load_shades() -> pd.DataFrame:
    return build_shade_master()


@st.cache_data(show_spinner=False)
def load_table(name: str):
    p = os.path.join(C.TBL_DIR, name)
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data(show_spinner=False)
def load_master():
    try:
        return pd.read_csv(C.master_path())
    except FileNotFoundError:
        return None


bundle = load_models()
shades = load_shades()

if bundle is None:
    st.title("AI-Driven Dye Formulation & Scale-Up Platform")
    st.error(
        "The dataset is missing, so the models cannot be built.\n\n"
        "Generate it first, from the project root:\n\n"
        "```\npython run_all.py\n```"
    )
    st.stop()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def primaries_for(family: str, n_dyes: int) -> list:
    mix = C.FAMILY_PRIMARY_MIX.get(family, [("Black", 1.0)])[:n_dyes]
    return [m[0] for m in mix]


def default_split(family: str, n_dyes: int, total_owf: float) -> list:
    """Nominal trichromatic split, used to pre-fill the lab-dip boxes."""
    mix = C.FAMILY_PRIMARY_MIX.get(family, [("Black", 1.0)])[:n_dyes]
    w = np.array([m[1] for m in mix], float)
    w = w / w.sum()
    strengths = np.array([C.DYE_PRIMARIES[m[0]]["strength"] for m in mix])
    raw = w / strengths
    conc = total_owf * raw / raw.sum()
    return [round(float(c), 4) for c in conc] + [0.0] * (4 - len(conc))


def build_input_row(sel: dict) -> pd.DataFrame:
    """One-row frame carrying every column the feature builder expects."""
    row = {
        "shade_family": sel["family"],
        "depth_class": sel["depth_class"],
        "fabric_structure": sel["structure"],
        "composition": sel["composition"],
        "pretreatment": sel["pretreatment"],
        "special_finish": sel["finish"],
        "dye_class": sel["dye_class"],
        "customer": sel["customer"],
        "gsm": float(sel["gsm"]),
        "cellulose_fraction": C.COMPOSITION_CELLULOSE_FRACTION[sel["composition"]],
        "dye_partition_K": C.DYE_CLASSES[sel["dye_class"]]["K"],
        "n_dyes": int(sel["n_dyes"]),
        "is_repeat": bool(sel["is_repeat"]),
        "labdip_machine": sel["lab_machine"],
        "labdip_batch_kg": 0.01,
        "labdip_liquor_ratio": float(sel["lab_lr"]),
        "labdip_rpm": float(sel["lab_rpm"]),
        "labdip_salt_g_l": float(sel["lab_salt"]),
        "labdip_soda_g_l": float(sel["lab_soda"]),
        "labdip_temp_c": float(sel["lab_temp"]),
        "labdip_time_min": float(sel["lab_time"]),
    }
    for j in range(4):
        row[f"labdip_dye{j + 1}_owf"] = float(sel["lab_conc"][j])
    row["labdip_total_owf"] = float(sum(sel["lab_conc"]))

    for stage in C.STAGES[1:]:
        row[f"{stage}_machine"] = sel["stage_machine"][stage]
        row[f"{stage}_batch_kg"] = float(sel["stage_batch"][stage])
        row[f"{stage}_liquor_ratio"] = float(sel["stage_lr"][stage])
        row[f"{stage}_rpm"] = float(sel["stage_rpm"][stage])
    return pd.DataFrame([row])


def predict_all_stages(bundle: dict, row: pd.DataFrame) -> dict:
    import ml_pipeline as M
    out = {}
    for stage, art in bundle["stages"].items():
        X, _, _ = M.build_features(row, "labdip", stage)
        X = X[art["cat_cols"] + art["num_cols"]]
        p = np.asarray(art["model"].predict(X), float)[0]
        rec = {f"dye{j + 1}_owf": float(max(p[j], 0.0)) for j in range(4)}
        rec["salt_g_l"] = float(max(p[4], 0.0))
        rec["soda_g_l"] = float(max(p[5], 0.0))
        rec["trials"] = float(np.clip(p[6], 1.0, C.TRIAL_MAX))
        rec["total_owf"] = sum(rec[f"dye{j + 1}_owf"] for j in range(4))
        rec["r2"] = art["r2_recipe"]
        rec["mape"] = art["mape_dyes_pct"]
        out[stage] = rec
    return out


def stage_cost(stage: str, rec: dict, sel: dict, primaries: list) -> dict:
    """Fully absorbed cost of one attempt at this stage, with a breakdown."""
    batch = float(sel["stage_batch"][stage])
    lr = float(sel["stage_lr"][stage])
    bath = batch * lr
    r = {
        f"{stage}_batch_kg": batch,
        f"{stage}_bath_volume_l": bath,
        f"{stage}_salt_g_l": rec["salt_g_l"],
        f"{stage}_soda_g_l": rec["soda_g_l"],
        f"{stage}_salt_mass_kg": rec["salt_g_l"] * bath / 1000.0,
        f"{stage}_soda_mass_kg": rec["soda_g_l"] * bath / 1000.0,
        f"{stage}_temp_c": float(sel["lab_temp"]),
        f"{stage}_time_min": float(sel["lab_time"]),
    }
    for j in range(4):
        r[f"{stage}_dye{j + 1}_owf"] = rec[f"dye{j + 1}_owf"]
        r[f"dye{j + 1}_primary"] = primaries[j] if j < len(primaries) else ""
    total = attempt_cost(stage, r)

    dye_usd = sum(
        (rec[f"dye{j + 1}_owf"] / 100.0 * batch)
        * C.DYE_PRIMARIES[primaries[j]]["price_usd_kg"]
        for j in range(min(len(primaries), 4)) if rec[f"dye{j + 1}_owf"] > 0)
    chem_usd = (r[f"{stage}_salt_mass_kg"] * C.CHEMICAL_PRICES_USD_KG["Glauber Salt"]
                + r[f"{stage}_soda_mass_kg"] * C.CHEMICAL_PRICES_USD_KG["Soda Ash"]
                + sum((g * bath / 1000.0) * C.CHEMICAL_PRICES_USD_KG[k]
                      for k, g in C.AUX_DOSING_G_L.items()))
    return {"total": total, "dye": dye_usd, "chem": chem_usd,
            "other": total - dye_usd - chem_usd, "bath_l": bath,
            "batch_kg": batch}


def kpi(col, label: str, value: str, sub: str = "", color: str | None = None):
    col.markdown(
        f"""<div class='kpi'><div class='label'>{label}</div>
        <div class='value'{f" style='color:{color}'" if color else ""}>{value}</div>
        <div class='sub'>{sub}</div></div>""", unsafe_allow_html=True)


# ===========================================================================
# SIDEBAR - inputs for the Recipe Predictor
# ===========================================================================
st.sidebar.markdown("## Recipe inputs")
st.sidebar.caption("These drive the **Recipe Predictor** tab.")

st.sidebar.markdown("### Shade & substrate")
shade_names = shades["shade_name"].tolist()
default_i = shade_names.index("Mesculen Green") if "Mesculen Green" in shade_names else 0
shade_name = st.sidebar.selectbox("Master shade", shade_names, index=default_i)
srow = shades[shades["shade_name"] == shade_name].iloc[0]

col_a, col_b = st.sidebar.columns(2)
fam_options = sorted(C.FAMILY_PRIMARY_MIX)
family = col_a.selectbox(
    "Shade family", fam_options,
    index=fam_options.index(srow["shade_family"])
    if srow["shade_family"] in fam_options else 0)
depth_class = col_b.selectbox("Depth class", C.DEPTH_CLASSES,
                              index=C.DEPTH_CLASSES.index(srow["depth_class"]))

structure = st.sidebar.selectbox("Fabric structure", C.FABRIC_STRUCTURES)
gsm_lo, gsm_hi = C.STRUCTURE_GSM_RANGE[structure]
gsm = st.sidebar.slider("Fabric weight (GSM)", 120, 320,
                        int((gsm_lo + gsm_hi) / 2), step=5)
composition = st.sidebar.selectbox("Composition", C.COMPOSITIONS)
c1, c2 = st.sidebar.columns(2)
pretreatment = c1.selectbox("Pre-treatment", C.PRETREATMENTS)
finish = c2.selectbox("Special finish", C.SPECIAL_FINISHES)
c3, c4 = st.sidebar.columns(2)
customer = c3.selectbox("Customer", ["GAP", "H&M", "Next", "Decathlon", "Tesco",
                                     "Primark", "M&S", "Uniqlo", "C&A", "Zara"])
is_repeat = c4.selectbox("Repeat shade", ["No", "Yes"]) == "Yes"

st.sidebar.markdown("---")
st.sidebar.markdown("### Lab dip (0.01 kg) — approved")
dye_class = st.sidebar.selectbox("Dye system", list(C.DYE_CLASSES))
max_dyes = len(C.FAMILY_PRIMARY_MIX.get(family, [("Black", 1.0)]))
n_dyes = st.sidebar.slider("Number of dyes", 1, max_dyes,
                           min(int(srow["n_dyes_typical"]), max_dyes))
primaries = primaries_for(family, n_dyes)

d_lo, d_hi = C.DEPTH_OWF_RANGE[depth_class]
preset_total = float(np.clip((d_lo + d_hi) / 2, 0.05, 7.0))
auto_fill = st.sidebar.checkbox("Auto-fill trichromatic split", value=True)
preset = default_split(family, n_dyes, preset_total) if auto_fill else [0.0] * 4

lab_conc = []
for j in range(4):
    if j < n_dyes:
        lab_conc.append(st.sidebar.number_input(
            f"Dye {j + 1} — {primaries[j]} (% owf)",
            min_value=0.0, max_value=10.0, value=float(preset[j]),
            step=0.01, format="%.4f", key=f"c{j}"))
    else:
        lab_conc.append(0.0)

tc = C.DYE_CLASSES[dye_class]
s1, s2 = st.sidebar.columns(2)
lab_lr = s1.number_input("Liquor ratio", 5.0, 30.0, 20.0, step=0.5)
lab_rpm = s2.number_input("RPM", 10.0, 40.0, 18.0, step=1.0)
s3, s4 = st.sidebar.columns(2)
salt_lo, salt_hi = C.DEPTH_SALT_G_L[depth_class]
soda_lo, soda_hi = C.DEPTH_SODA_G_L[depth_class]
lab_salt = s3.number_input("Salt (g/L)", 0.0, 100.0,
                           float((salt_lo + salt_hi) / 2), step=1.0)
lab_soda = s4.number_input("Soda ash (g/L)", 0.0, 40.0,
                           float((soda_lo + soda_hi) / 2), step=0.5)
s5, s6 = st.sidebar.columns(2)
lab_temp = s5.number_input("Temperature (C)", 40.0, 100.0,
                           float(np.mean(tc["temp_c"])), step=1.0)
lab_time = s6.number_input("Time (min)", 20.0, 120.0,
                           float(np.mean(tc["time_min"])), step=5.0)
lab_machine = st.sidebar.selectbox("Lab machine", C.STAGE_MACHINES["labdip"])

st.sidebar.markdown("---")
st.sidebar.markdown("### Production plan")
stage_batch, stage_lr, stage_rpm, stage_machine = {}, {}, {}, {}
for stage in C.STAGES[1:]:
    with st.sidebar.expander(C.STAGE_LABELS[stage], expanded=(stage == "bulk")):
        b_lo, b_hi = C.STAGE_BATCH_KG[stage]
        if b_lo == b_hi:
            stage_batch[stage] = b_lo
            st.caption(f"Fixed batch size: {b_lo:g} kg")
        else:
            stage_batch[stage] = st.number_input(
                "Batch (kg)", float(b_lo), float(b_hi), float((b_lo + b_hi) / 2),
                step=5.0, key=f"b{stage}")
        l_lo, l_hi = C.STAGE_LIQUOR_RATIO[stage]
        stage_lr[stage] = st.slider("Liquor ratio", float(l_lo), float(l_hi),
                                    float((l_lo + l_hi) / 2), step=0.1,
                                    key=f"l{stage}")
        r_lo, r_hi = C.STAGE_RPM[stage]
        stage_rpm[stage] = st.slider("RPM", int(r_lo), int(r_hi),
                                     int((r_lo + r_hi) / 2), key=f"r{stage}")
        stage_machine[stage] = st.selectbox("Machine", C.STAGE_MACHINES[stage],
                                            key=f"m{stage}")

sel = dict(family=family, depth_class=depth_class, structure=structure,
           composition=composition, pretreatment=pretreatment, finish=finish,
           dye_class=dye_class, customer=customer, gsm=gsm, n_dyes=n_dyes,
           is_repeat=is_repeat, lab_machine=lab_machine, lab_lr=lab_lr,
           lab_rpm=lab_rpm, lab_salt=lab_salt, lab_soda=lab_soda,
           lab_temp=lab_temp, lab_time=lab_time, lab_conc=lab_conc,
           stage_batch=stage_batch, stage_lr=stage_lr, stage_rpm=stage_rpm,
           stage_machine=stage_machine)


# ===========================================================================
# HEADER
# ===========================================================================
st.markdown("# AI-Driven Dye Formulation & Scale-Up Platform")
st.markdown(
    f"<span class='note'>Champion model <b>{bundle['champion']}</b> · trained on "
    f"{bundle['reference']['n_training_jobs']:,} scale-up programmes · "
    f"colour tolerance {bundle['reference']['tolerance'] * 100:.1f}% relative "
    f"recipe error</span>", unsafe_allow_html=True)
st.markdown("")

tab_predict, tab_explore, tab_models, tab_roi = st.tabs(
    ["  Recipe Predictor  ", "  Scale-Up Explorer  ",
     "  Model Performance  ", "  Cost & ROI  "])


# ===========================================================================
# TAB 1 - RECIPE PREDICTOR
# ===========================================================================
with tab_predict:
    total_lab = sum(lab_conc)
    if total_lab <= 0:
        st.warning("Enter at least one lab-dip dye concentration to run a prediction.")
    else:
        row = build_input_row(sel)
        preds = predict_all_stages(bundle, row)
        bulk = preds["bulk"]

        st.markdown(
            f"<div class='banner'>Predicting <b>{shade_name}</b> "
            f"({family}, {depth_class}) on {gsm:g} gsm {structure.lower()}, "
            f"{composition}, {pretreatment.lower()} — "
            f"{n_dyes} dye{'s' if n_dyes > 1 else ''} "
            f"({', '.join(primaries)}), {dye_class}.</div>",
            unsafe_allow_html=True)

        adj_bulk = bulk["total_owf"] / total_lab
        costs = {s: stage_cost(s, preds[s], sel, primaries) for s in C.STAGES[1:]}
        bulk_cost = costs["bulk"]

        tol = bundle["reference"]["tolerance"]
        mape_bulk = bulk["mape"] / 100.0
        confidence = float(np.clip(1.0 - mape_bulk / tol, 0.0, 1.0))
        if confidence >= 0.5:
            conf_col, conf_txt = T.STATUS["good"], "High"
        elif confidence >= 0.2:
            conf_col, conf_txt = T.STATUS["warning"], "Moderate"
        else:
            conf_col, conf_txt = T.STATUS["critical"], "Low"

        pred_corrective = sum(preds[s]["trials"] - 1 for s in C.STAGES[1:])

        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "Bulk recipe",
            f"{bulk['total_owf']:.4f}<span style='font-size:1rem'> % owf</span>",
            f"Lab dip {total_lab:.4f} % owf · factor <b>{adj_bulk:.4f}</b>")
        kpi(k2, "Bulk batch cost", f"${bulk_cost['total']:,.0f}",
            f"{bulk_cost['batch_kg']:g} kg · {bulk_cost['bath_l']:,.0f} L bath · "
            f"LKR {bulk_cost['total'] * C.LKR_PER_USD:,.0f}")
        kpi(k3, "Expected corrective trials", f"{pred_corrective:.2f}",
            f"Mill baseline {C.BASELINE_CORRECTIVE_TRIALS_PER_JOB:.1f} across the ladder")
        kpi(k4, "Recipe confidence", conf_txt,
            f"Held-out dye error {bulk['mape']:.2f}% vs {tol * 100:.1f}% tolerance",
            color=conf_col)

        st.markdown("")
        left, right = st.columns([1.05, 1])

        with left:
            st.markdown("#### Scale-up trajectory")
            labels = [C.STAGE_LABELS[s].split(" (")[0] for s in C.STAGES]
            fig = go.Figure()
            for j in range(n_dyes):
                ys = [lab_conc[j]] + [preds[s][f"dye{j + 1}_owf"] for s in C.STAGES[1:]]
                fig.add_trace(go.Scatter(
                    x=labels, y=ys, mode="lines+markers",
                    name=f"Dye {j + 1} · {primaries[j]}",
                    line=dict(color=T.CATEGORICAL[j], width=2),
                    marker=dict(size=9, line=dict(color=T.SURFACE, width=2)),
                    hovertemplate="%{y:.4f} %% owf<extra>"
                                  f"Dye {j + 1} {primaries[j]}</extra>"))
            totals = [total_lab] + [preds[s]["total_owf"] for s in C.STAGES[1:]]
            fig.add_trace(go.Scatter(
                x=labels, y=totals, mode="lines+markers", name="Total",
                line=dict(color=T.INK_2, width=2, dash="dot"),
                marker=dict(size=8, line=dict(color=T.SURFACE, width=2)),
                hovertemplate="total %{y:.4f} %% owf<extra></extra>"))
            fig.update_layout(
                height=375, hovermode="x unified",
                legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=11)),
                yaxis=dict(title="% owf", gridcolor=T.GRID, zeroline=False),
                xaxis=dict(gridcolor=T.SURFACE), **PLOT_LAYOUT)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(
                f"<span class='note'>The correction is not one factor: it tightens "
                f"as the liquor ratio contracts from {lab_lr:g}:1 at lab dip to "
                f"{stage_lr['bulk']:.1f}:1 at bulk, and deep shades resist it.</span>",
                unsafe_allow_html=True)

        with right:
            st.markdown("#### Recipes to issue")
            rows = []
            for stage in C.STAGES:
                if stage == "labdip":
                    rec = {f"dye{j + 1}_owf": lab_conc[j] for j in range(4)}
                    rec.update(salt_g_l=lab_salt, soda_g_l=lab_soda,
                               total_owf=total_lab)
                    batch, lr = 0.01, lab_lr
                else:
                    rec, batch, lr = preds[stage], stage_batch[stage], stage_lr[stage]
                entry = {
                    "Stage": C.STAGE_LABELS[stage].split(" (")[0],
                    "Batch kg": round(batch, 2),
                    "LR": round(lr, 1),
                    "Total % owf": round(rec["total_owf"], 4),
                    "Factor": round(rec["total_owf"] / total_lab, 4),
                }
                for j in range(n_dyes):
                    entry[f"D{j + 1} g"] = round(
                        rec[f"dye{j + 1}_owf"] / 100.0 * batch * 1000.0, 2)
                entry["Salt g/L"] = round(rec["salt_g_l"], 1)
                entry["Soda g/L"] = round(rec["soda_g_l"], 1)
                rows.append(entry)
            recipe_df = pd.DataFrame(rows)
            st.dataframe(recipe_df, hide_index=True, use_container_width=True)
            st.download_button(
                "Download recipe card (CSV)",
                recipe_df.to_csv(index=False).encode("utf-8"),
                file_name=f"recipe_{shade_name.replace(' ', '_')}.csv",
                mime="text/csv")

        st.markdown("---")
        st.markdown("#### Cost of one attempt, by stage")
        cc1, cc2 = st.columns([1.25, 1])
        with cc1:
            bars = go.Figure()
            stages_x = [C.STAGE_LABELS[s].split(" (")[0] for s in C.STAGES[1:]]
            for key, name, col in (("dye", "Dyestuff", T.SERIES[0]),
                                   ("chem", "Chemicals", T.SERIES[1]),
                                   ("other", "Water, steam, effluent, machine",
                                    T.SERIES[2])):
                bars.add_trace(go.Bar(
                    x=stages_x, y=[costs[s][key] for s in C.STAGES[1:]], name=name,
                    marker=dict(color=col, line=dict(color=T.SURFACE, width=2)),
                    hovertemplate="%{y:$,.2f}<extra>" + name + "</extra>"))
            bars.update_layout(
                barmode="stack", height=330, hovermode="x unified",
                legend=dict(orientation="h", y=-0.22, x=0, font=dict(size=11)),
                yaxis=dict(title="Cost of one attempt (USD)", gridcolor=T.GRID),
                xaxis=dict(gridcolor=T.SURFACE), **PLOT_LAYOUT)
            st.plotly_chart(bars, use_container_width=True)
        with cc2:
            avoided = max(C.BASELINE_CORRECTIVE_TRIALS_PER_JOB - pred_corrective, 0.0)
            ladder_cost = sum(costs[s]["total"] for s in C.STAGES[1:]) / 4.0
            saving = avoided * ladder_cost
            programmes = C.FACTORY["distinct_shades_per_year"]
            net = saving * programmes - C.INVESTMENT["annual_opex_usd"]
            st.markdown(f"""
<div class='kpi'>
  <div class='label'>Saving, this shade</div>
  <div class='value'>${saving:,.0f}</div>
  <div class='sub'>{avoided:.2f} corrective trials avoided, costed at the mean
  attempt cost across the four production stages (${ladder_cost:,.0f}).</div>
  <div style='margin-top:14px' class='label'>Annualised at {programmes} programmes</div>
  <div class='value'>${saving * programmes:,.0f}</div>
  <div class='sub'>LKR {saving * programmes * C.LKR_PER_USD:,.0f}
  {"· payback in %.1f months" % (C.INVESTMENT['capex_usd'] / net * 12) if net > 0 else ""}
  </div>
</div>""", unsafe_allow_html=True)
            st.markdown(
                "<span class='note'>A single-shade projection. The audited "
                "factory-wide figure, derived from measured residuals across the "
                "whole test set, is in the <b>Cost & ROI</b> tab.</span>",
                unsafe_allow_html=True)


# ===========================================================================
# TAB 2 - SCALE-UP EXPLORER
# ===========================================================================
with tab_explore:
    master = load_master()
    if master is None:
        st.error("Dataset not found. Run `python src/data_generator.py`.")
    else:
        st.markdown("#### Explore the scale-up population")
        st.markdown(
            f"<span class='note'>{len(master):,} scale-up programmes across "
            f"{master['shade_code'].nunique()} shades. Filters apply to every "
            f"chart and figure below.</span>", unsafe_allow_html=True)
        st.markdown("")

        f1, f2, f3, f4 = st.columns(4)
        fams = f1.multiselect("Shade family",
                              sorted(master["shade_family"].unique()), default=[])
        depths = f2.multiselect("Depth class", C.DEPTH_CLASSES, default=[])
        structs = f3.multiselect("Fabric structure", C.FABRIC_STRUCTURES, default=[])
        dcls = f4.multiselect("Dye system", list(C.DYE_CLASSES), default=[])

        d = master
        if fams:
            d = d[d["shade_family"].isin(fams)]
        if depths:
            d = d[d["depth_class"].isin(depths)]
        if structs:
            d = d[d["fabric_structure"].isin(structs)]
        if dcls:
            d = d[d["dye_class"].isin(dcls)]

        if d.empty:
            st.warning("No jobs match those filters.")
        else:
            st.markdown("")
            e1, e2, e3, e4 = st.columns(4)
            kpi(e1, "Programmes selected", f"{len(d):,}",
                f"{len(d) / len(master) * 100:.1f}% of the population")
            kpi(e2, "Bulk adjustment factor", f"{d['bulk_adj_factor'].mean():.4f}",
                f"p5 {d['bulk_adj_factor'].quantile(0.05):.3f} · "
                f"p95 {d['bulk_adj_factor'].quantile(0.95):.3f}")
            kpi(e3, "Corrective trials", f"{d['corrective_trials'].mean():.2f}",
                f"Bulk right-first-time {d['rft_bulk'].mean() * 100:.1f}%")
            kpi(e4, "Cost of poor quality",
                f"${d['cost_of_poor_quality_usd'].mean():,.0f}",
                f"{d['cost_of_poor_quality_usd'].sum() / d['total_cost_actual_usd'].sum() * 100:.1f}% "
                f"of total spend")

            st.markdown("")
            g1, g2 = st.columns(2)

            with g1:
                st.markdown("##### Recipe correction along the ladder")
                fig = go.Figure()
                for s, col in zip(C.STAGES[1:], T.STAGE_RAMP[1:]):
                    fig.add_trace(go.Box(
                        y=d[f"{s}_adj_factor"],
                        name=C.STAGE_LABELS[s].split(" (")[0],
                        marker=dict(color=col), line=dict(width=2),
                        fillcolor=col, boxpoints=False,
                        hovertemplate="median %{median:.4f}<extra></extra>"))
                fig.add_hline(y=1.0, line=dict(color=T.INK_MUTED, width=1,
                                               dash="dash"),
                              annotation_text="lab dip reference",
                              annotation_font=dict(size=10, color=T.INK_MUTED))
                fig.update_layout(
                    height=360, showlegend=False,
                    yaxis=dict(title="Adjustment factor", gridcolor=T.GRID),
                    xaxis=dict(gridcolor=T.SURFACE), **PLOT_LAYOUT)
                st.plotly_chart(fig, use_container_width=True)

            with g2:
                st.markdown("##### Rework cost rises with shade depth")
                dcol = T.depth_colors()
                present = [x for x in C.DEPTH_CLASSES if x in set(d["depth_class"])]
                fig2 = go.Figure()
                for dc in present:
                    sub = d[d["depth_class"] == dc]
                    fig2.add_trace(go.Box(
                        y=sub["cost_of_poor_quality_usd"], name=dc,
                        marker=dict(color=dcol[dc]), fillcolor=dcol[dc],
                        line=dict(width=2), boxpoints=False,
                        hovertemplate="median $%{median:,.0f}<extra></extra>"))
                fig2.update_layout(
                    height=360, showlegend=False,
                    yaxis=dict(title="Cost of poor quality (USD)",
                               gridcolor=T.GRID,
                               range=[0, float(d["cost_of_poor_quality_usd"]
                                               .quantile(0.98))]),
                    xaxis=dict(gridcolor=T.SURFACE), **PLOT_LAYOUT)
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("##### Mean bulk recipe by shade family")
            g = (d.groupby("shade_family")
                   .agg(owf=("bulk_total_owf", "mean"),
                        factor=("bulk_adj_factor", "mean"),
                        trials=("corrective_trials", "mean"),
                        cpq=("cost_of_poor_quality_usd", "mean"),
                        n=("job_id", "size"))
                   .sort_values("owf", ascending=True))
            fig3 = go.Figure(go.Bar(
                x=g["owf"], y=g.index, orientation="h",
                marker=dict(color=T.SEQ_HUE,
                            line=dict(color=T.SURFACE, width=2)),
                customdata=np.stack([g["n"], g["factor"], g["cpq"]], axis=-1),
                hovertemplate="<b>%{y}</b><br>%{x:.3f} %% owf<br>"
                              "factor %{customdata[1]:.4f}<br>"
                              "CPQ $%{customdata[2]:,.0f}<br>"
                              "%{customdata[0]} jobs<extra></extra>"))
            fig3.update_layout(
                height=max(300, 26 * len(g)),
                xaxis=dict(title="Mean bulk total % owf", gridcolor=T.GRID),
                yaxis=dict(gridcolor=T.SURFACE), **PLOT_LAYOUT)
            st.plotly_chart(fig3, use_container_width=True)

            with st.expander("Browse the filtered records"):
                show_cols = ["job_id", "shade_name", "shade_family", "depth_class",
                             "fabric_structure", "gsm", "n_dyes",
                             "labdip_total_owf", "bulk_total_owf",
                             "bulk_adj_factor", "bulk_liquor_ratio",
                             "corrective_trials", "total_cost_actual_usd"]
                st.dataframe(d[show_cols].head(400), hide_index=True,
                             use_container_width=True)
                st.download_button(
                    "Download filtered selection (CSV)",
                    d.to_csv(index=False).encode("utf-8"),
                    file_name="scaleup_selection.csv", mime="text/csv")


# ===========================================================================
# TAB 3 - MODEL PERFORMANCE
# ===========================================================================
with tab_models:
    bench = load_table("model_benchmark_summary.csv")
    stage_acc = load_table("stage_model_accuracy.csv")
    trial_pred = load_table("trial_count_predictability.csv")

    if bench is None:
        st.error("Benchmark not found. Run `python src/ml_pipeline.py`.")
    else:
        st.markdown("#### Why this model")
        st.markdown(
            "<div class='banner'>Models are ranked on <b>concentration-weighted "
            "relative recipe error</b> — the quantity that decides whether a batch "
            "is accepted — not on R². Every model below except the two unregularised "
            "trees scores R² above 0.96, yet their rework exposure ranges from 31% "
            "to 97%. R² is dominated by the enormous spread between pastel and "
            "extra-dark shades; getting roughly the right depth earns almost all "
            "of it.</div>", unsafe_allow_html=True)

        direct = bench[bench["track"] == "direct"].sort_values("recipe_rel_error_pct")
        champ = direct.iloc[0]

        m1, m2, m3, m4 = st.columns(4)
        kpi(m1, "Champion", str(champ["model"]),
            f"Direct track · {bundle['reference']['n_training_jobs']:,} training jobs")
        kpi(m2, "Recipe error", f"{champ['recipe_rel_error_pct']:.2f}%",
            f"Colour tolerance {C.RECIPE_TOLERANCE_REL * 100:.1f}%",
            color=T.STATUS["good"])
        kpi(m3, "Rework exposure", f"{champ['off_tolerance_pct']:.1f}%",
            "Share of batches needing a second dyeing")
        kpi(m4, "R² (recipe targets)", f"{champ['R2_recipe']:.4f}",
            f"MAE {champ['MAE_recipe']:.4f} · RMSE {champ['RMSE_recipe']:.4f}")

        st.markdown("")
        b1, b2 = st.columns([1.1, 1])
        with b1:
            st.markdown("##### Benchmark: recipe error by model")
            dd = bench.dropna(subset=["recipe_rel_error_pct"]).sort_values(
                "recipe_rel_error_pct", ascending=False)
            cols = [T.STATUS["good"] if m == champ["model"] and t == "direct"
                    else T.SERIES[0]
                    for m, t in zip(dd["model"], dd["track"])]
            figb = go.Figure(go.Bar(
                x=dd["recipe_rel_error_pct"], y=dd["model"], orientation="h",
                marker=dict(color=cols, line=dict(color=T.SURFACE, width=2)),
                customdata=np.stack([dd["off_tolerance_pct"], dd["R2_recipe"]],
                                    axis=-1),
                hovertemplate="<b>%{y}</b><br>recipe error %{x:.2f}%<br>"
                              "off-tolerance %{customdata[0]:.1f}%<br>"
                              "R² %{customdata[1]:.4f}<extra></extra>"))
            figb.add_vline(x=C.RECIPE_TOLERANCE_REL * 100,
                           line=dict(color=T.STATUS["critical"], width=2,
                                     dash="dash"),
                           annotation_text="colour tolerance",
                           annotation_font=dict(size=10,
                                                color=T.STATUS["critical"]))
            figb.update_layout(
                height=430,
                xaxis=dict(title="Concentration-weighted recipe error (%)",
                           gridcolor=T.GRID),
                yaxis=dict(gridcolor=T.SURFACE), **PLOT_LAYOUT)
            st.plotly_chart(figb, use_container_width=True)

        with b2:
            st.markdown("##### Accuracy by stage")
            if stage_acc is not None:
                figs = go.Figure(go.Bar(
                    x=stage_acc["stage_label"], y=stage_acc["mape_dyes_pct"],
                    marker=dict(color=T.STAGE_RAMP[1:1 + len(stage_acc)],
                                line=dict(color=T.SURFACE, width=2)),
                    hovertemplate="<b>%{x}</b><br>MAPE %{y:.2f}%<extra></extra>"))
                figs.add_hline(y=C.RECIPE_TOLERANCE_REL * 100,
                               line=dict(color=T.STATUS["critical"], width=2,
                                         dash="dash"))
                figs.update_layout(
                    height=430,
                    yaxis=dict(title="MAPE on dye concentrations (%)",
                               gridcolor=T.GRID),
                    xaxis=dict(gridcolor=T.SURFACE, tickangle=-20), **PLOT_LAYOUT)
                st.plotly_chart(figs, use_container_width=True)
                st.markdown(
                    "<span class='note'>Error grows monotonically with distance "
                    "from the lab dip — the further the extrapolation, the larger "
                    "the liquor-ratio contraction.</span>",
                    unsafe_allow_html=True)

        st.markdown("##### Full benchmark")
        show = bench[["model", "track", "recipe_rel_error_pct", "off_tolerance_pct",
                      "R2_recipe", "MAE_recipe", "RMSE_recipe", "MAPE_dyes_pct",
                      "R2_dyes", "R2_trials"]].round(4)
        show.columns = ["Model", "Track", "Recipe err %", "Off-tol %", "R² recipe",
                        "MAE", "RMSE", "MAPE dyes %", "R² dyes", "R² trials"]
        st.dataframe(show, hide_index=True, use_container_width=True)

        if trial_pred is not None:
            tp = trial_pred.iloc[0]
            st.markdown(
                f"<div class='banner warn'><b>Why R² on the trial count is near "
                f"zero, and why that is correct.</b> The count is generated as "
                f"1 + Poisson(λ), so only the variance of the conditional mean is "
                f"reducible. The attainable ceiling is "
                f"<b>{tp['attainable_r2_binned']:.3f}</b>; the champion reaches "
                f"{tp['achieved_r2']:.3f}. An individual batch's trial count is "
                f"not predictable and is not presented as a platform output — what "
                f"the platform shifts is the <i>rate</i>, derived from recipe "
                f"accuracy in the Cost & ROI tab.</div>",
                unsafe_allow_html=True)

        st.markdown("##### Diagnostic figures")
        figs_to_show = [
            ("outputs/figures/fig07_model_comparison.png",
             "Model benchmark: recipe error and rework exposure"),
            ("outputs/figures/fig08_feature_importance.png",
             "Permutation and SHAP feature attribution"),
            ("outputs/figures/fig09_pred_vs_actual.png",
             "Predicted vs actual, champion model"),
        ]
        for path, cap in figs_to_show:
            full = os.path.join(C.ROOT, path)
            if os.path.exists(full):
                st.image(full, caption=cap, use_container_width=True)


# ===========================================================================
# TAB 4 - COST & ROI
# ===========================================================================
with tab_roi:
    scen = load_table("cba_per_job_scenarios.csv")
    summary = load_table("cba_factory_summary.csv")
    regimes = load_table("cba_regime_accuracy.csv")
    gapt = load_table("cba_target_gap_analysis.csv")

    if scen is None or summary is None:
        st.error("Cost-benefit results not found. Run `python src/cost_benefit.py`.")
    else:
        base = summary.iloc[0]
        st.markdown("#### Factory economics")
        st.markdown(
            "<div class='banner'>These savings are <b>derived from measured model "
            "error</b>, not assumed. The probability of landing outside the colour "
            "tolerance was measured on held-out jobs under three regimes, and the "
            "rework rate scaled by the resulting ratio. Were the model poor, that "
            "ratio would approach 1 and the savings would vanish.</div>",
            unsafe_allow_html=True)

        if regimes is not None:
            st.markdown("##### Recipe accuracy by regime (held-out jobs)")
            rg = regimes.copy()
            colmap = [T.STATUS["critical"], T.SERIES[1], T.STATUS["good"]]
            figr = go.Figure(go.Bar(
                x=rg["regime"], y=rg["off_tolerance_rate_pct"],
                marker=dict(color=colmap[:len(rg)],
                            line=dict(color=T.SURFACE, width=2)),
                text=[f"{v:.1f}%" for v in rg["off_tolerance_rate_pct"]],
                textposition="outside",
                customdata=np.stack([rg["median_rel_error_pct"],
                                     rg["p90_rel_error_pct"]], axis=-1),
                hovertemplate="<b>%{x}</b><br>outside tolerance %{y:.1f}%<br>"
                              "median error %{customdata[0]:.2f}%<br>"
                              "p90 error %{customdata[1]:.2f}%<extra></extra>"))
            figr.update_layout(
                height=330,
                yaxis=dict(title="Jobs outside colour tolerance (%)",
                           gridcolor=T.GRID, range=[0, 100]),
                xaxis=dict(gridcolor=T.SURFACE), **PLOT_LAYOUT)
            st.plotly_chart(figr, use_container_width=True)

        st.markdown("##### Your assumptions")
        a1, a2, a3, a4 = st.columns(4)
        programmes = a1.number_input("Scale-up programmes per year", 50, 2000,
                                     int(base["programmes_per_year"]), step=10)
        capex = a2.number_input("Capital outlay (USD)", 5_000, 500_000,
                                int(base["capex_usd"]), step=1_000)
        opex = a3.number_input("Annual platform opex (USD)", 0, 100_000,
                               int(base["annual_opex_usd"]), step=500)
        rate = a4.slider("Discount rate (%)", 0, 30,
                         int(C.INVESTMENT["discount_rate"] * 100)) / 100.0
        horizon = st.slider("Appraisal horizon (years)", 1, 10,
                            C.INVESTMENT["horizon_years"])

        per_programme = float(base["mean_saving_per_programme_usd"])
        gross = per_programme * programmes
        net = gross - opex
        payback = capex / net if net > 0 else np.inf
        npv = -capex + sum(net / (1 + rate) ** t for t in range(1, horizon + 1))

        irr = np.nan
        if net > 0:
            lo, hi = 0.0, 10.0
            def _npv(r):
                return -capex + sum(net / (1 + r) ** t for t in range(1, horizon + 1))
            if _npv(hi) < 0:
                for _ in range(200):
                    mid = 0.5 * (lo + hi)
                    if _npv(mid) > 0:
                        lo = mid
                    else:
                        hi = mid
                irr = 0.5 * (lo + hi)

        st.markdown("")
        r1, r2, r3, r4 = st.columns(4)
        kpi(r1, "Net annual saving", f"${net:,.0f}",
            f"Gross ${gross:,.0f} less ${opex:,.0f} opex · "
            f"LKR {net * C.LKR_PER_USD:,.0f}")
        kpi(r2, "Payback",
            f"{payback * 12:,.1f} mo" if np.isfinite(payback) else "never",
            f"On ${capex:,.0f} capital outlay")
        kpi(r3, f"NPV ({horizon} yr @ {rate * 100:.0f}%)", f"${npv:,.0f}",
            "Negative NPV means the case does not clear the hurdle rate"
            if npv < 0 else "Clears the hurdle rate",
            color=T.STATUS["good"] if npv > 0 else T.STATUS["critical"])
        kpi(r4, "IRR", f"{irr * 100:,.0f}%" if np.isfinite(irr) else "n/a",
            f"Saving per programme ${per_programme:,.0f}")

        st.markdown("")
        c1, c2 = st.columns([1.15, 1])
        with c1:
            st.markdown("##### Cumulative net cash flow")
            yrs = list(range(0, horizon + 1))
            cash = [-capex + net * t for t in yrs]
            figc = go.Figure()
            figc.add_trace(go.Scatter(
                x=yrs, y=cash, mode="lines+markers",
                line=dict(color=T.SERIES[0], width=2),
                marker=dict(size=9, line=dict(color=T.SURFACE, width=2)),
                name="Cumulative cash",
                hovertemplate="year %{x}<br>%{y:$,.0f}<extra></extra>"))
            figc.add_hline(y=0, line=dict(color=T.INK_MUTED, width=1, dash="dash"))
            if np.isfinite(payback) and payback <= horizon:
                figc.add_vline(
                    x=payback, line=dict(color=T.STATUS["good"], width=2,
                                         dash="dash"),
                    annotation_text=f"payback {payback * 12:.1f} mo",
                    annotation_font=dict(size=11, color=T.STATUS["good"]))
            figc.update_layout(
                height=360, showlegend=False,
                yaxis=dict(title="Cumulative net cash flow (USD)",
                           gridcolor=T.GRID, tickprefix="$"),
                xaxis=dict(title="Year", gridcolor=T.SURFACE), **PLOT_LAYOUT)
            st.plotly_chart(figc, use_container_width=True)

        with c2:
            st.markdown("##### Savings by shade depth")
            dcol = T.depth_colors()
            g = (scen.groupby("depth_class")["saving_usd"].mean()
                     .reindex(C.DEPTH_CLASSES).dropna())
            figd = go.Figure(go.Bar(
                x=g.index, y=g.values,
                marker=dict(color=[dcol[i] for i in g.index],
                            line=dict(color=T.SURFACE, width=2)),
                text=[f"${v:,.0f}" for v in g.values], textposition="outside",
                hovertemplate="<b>%{x}</b><br>%{y:$,.0f} per programme<extra></extra>"))
            figd.update_layout(
                height=360,
                yaxis=dict(title="Mean saving per programme (USD)",
                           gridcolor=T.GRID,
                           range=[0, float(g.max()) * 1.2]),
                xaxis=dict(gridcolor=T.SURFACE), **PLOT_LAYOUT)
            st.plotly_chart(figd, use_container_width=True)

        st.markdown("##### Operational and environmental impact per year")
        scale = programmes / float(base["programmes_per_year"])
        imp = pd.DataFrame([
            {"Measure": "Corrective trials per shade",
             "Baseline": f"{base['corrective_trials_baseline']:.2f}",
             "With platform": f"{base['corrective_trials_ai']:.2f}",
             "Target": f"<= {C.TARGET_CORRECTIVE_TRIALS_PER_JOB}",
             "Status": "Met"},
            {"Measure": "Bulk right-first-time",
             "Baseline": f"{base['rft_bulk_baseline'] * 100:.1f}%",
             "With platform": f"{base['rft_bulk_ai'] * 100:.1f}%",
             "Target": f">= {C.TARGET_RFT_BULK * 100:.0f}%",
             "Status": "Not met"},
            {"Measure": "Cost per programme",
             "Baseline": f"${base['mean_cost_baseline_usd']:,.0f}",
             "With platform": f"${base['mean_cost_ai_usd']:,.0f}",
             "Target": "—",
             "Status": f"-{base['saving_pct_of_spend']:.1f}%"},
        ])
        st.dataframe(imp, hide_index=True, use_container_width=True)

        env = pd.DataFrame([
            {"Resource": "Dyestuff",
             "Annual reduction": f"{base['annual_dye_saved_kg'] * scale:,.0f} kg"},
            {"Resource": "Chemicals (salt, soda)",
             "Annual reduction": f"{base['annual_chemicals_saved_kg'] * scale:,.0f} kg"},
            {"Resource": "Water",
             "Annual reduction": f"{base['annual_water_saved_m3'] * scale:,.0f} m³"},
            {"Resource": "Thermal energy",
             "Annual reduction": f"{base['annual_thermal_mwh_saved'] * scale:,.1f} MWh"},
            {"Resource": "CO₂e",
             "Annual reduction": f"{base['annual_co2e_saved_t'] * scale:,.1f} t"},
            {"Resource": "Machine time released",
             "Annual reduction": f"{base['annual_machine_hours_saved'] * scale:,.0f} h"},
        ])
        st.dataframe(env, hide_index=True, use_container_width=True)

        if gapt is not None:
            gp = gapt.iloc[0]
            st.markdown(
                f"<div class='banner warn'><b>The 85% right-first-time target is "
                f"not met.</b> The platform reaches "
                f"{gp['rft_achieved'] * 100:.1f}%. Reaching "
                f"{gp['rft_target'] * 100:.0f}% requires the off-tolerance rate to "
                f"fall from {gp['off_tolerance_achieved_pct']:.1f}% to "
                f"{gp['off_tolerance_required_pct']:.1f}% — roughly halving the "
                f"model's recipe error, <i>or</i> establishing that the true "
                f"colour-acceptance tolerance is nearer "
                f"{gp['tolerance_band_required_pct']:.2f}% than the "
                f"{C.RECIPE_TOLERANCE_REL * 100:.1f}% assumed here. The second is "
                f"an empirical question the mill can settle from its own ΔE "
                f"acceptance records, and it costs nothing to check.</div>",
                unsafe_allow_html=True)


# ---------------------------------------------------------------------------
st.markdown("---")
st.markdown(
    f"<span class='note'>Trained on a physics-grounded synthetic population of "
    f"{C.N_RECORDS:,} scale-up programmes built from the mill's own 499-shade "
    f"dictionary and five-stage job-card schema. Metrics come from a held-out "
    f"{C.TEST_SIZE:.0%} split. <b>Every result characterises the method, not a "
    f"deployed system</b> — before production use the models must be re-fitted on "
    f"real job-card history.</span>", unsafe_allow_html=True)
