"""
config.py
=========
Single source of truth for the domain model behind the AI-Driven Dye Formulation
& Scale-Up Optimization Platform.

Everything that is a *physical constant*, an *industrial price*, or a *process
window* lives here so it can be audited, cited and replaced with real factory
data without touching the modelling code.

Sources / grounding
-------------------
* Scale-up stage ladder, machine names and the 5-stage recipe layout were taken
  from the factory job-card workbook supplied with this project
  (ZXCV-.123.xlsx, 87 columns, one row per job).
* The 499-entry master shade dictionary (shade name / family / depth direction)
  was taken from 500_Textile_Dyeing_Shades.xlsx.
* Exhaustion / liquor-ratio theory follows the partition model for reactive
  dyeing of cellulose (see RESEARCH_PAPER.md Section 2).
* Prices are indicative 2024-2025 South-Asian mill landed costs, in USD.
"""

from __future__ import annotations

import os

# ---------------------------------------------------------------------------
# 0. Reproducibility & dataset size
# ---------------------------------------------------------------------------
RANDOM_SEED = 20251001
N_RECORDS = 8000          # master dataset size (brief asks for 5,000-10,000)
TEST_SIZE = 0.20
VAL_SIZE = 0.10           # carved out of the training partition

LKR_PER_USD = 305.0       # for dual-currency reporting

# ---------------------------------------------------------------------------
# 1. The five scale-up stages
# ---------------------------------------------------------------------------
STAGES = ["labdip", "s1kg", "s7kg", "minibulk", "bulk"]

STAGE_LABELS = {
    "labdip": "Lab Dip (0.01 kg)",
    "s1kg": "1 kg Sample",
    "s7kg": "7 kg Sample",
    "minibulk": "Mini Bulk (20-75 kg)",
    "bulk": "Bulk (100-500 kg)",
}

# Batch weight (kg of dry fabric) sampling window per stage.
STAGE_BATCH_KG = {
    "labdip": (0.01, 0.01),
    "s1kg": (1.0, 1.0),
    "s7kg": (7.0, 7.0),
    "minibulk": (20.0, 75.0),
    "bulk": (100.0, 500.0),
}

# Liquor ratio (litres of bath per kg of fabric) window per stage. Contracts
# monotonically with scale: the main driver of exhaustion change.
STAGE_LIQUOR_RATIO = {
    "labdip": (20.0, 20.0),
    "s1kg": (15.0, 20.0),
    "s7kg": (12.0, 15.0),
    "minibulk": (10.0, 12.0),
    "bulk": (6.0, 10.0),
}

# Mechanical agitation window (RPM / reel-speed proxy) per stage.
STAGE_RPM = {
    "labdip": (15, 20),
    "s1kg": (18, 24),
    "s7kg": (22, 28),
    "minibulk": (24, 32),
    "bulk": (26, 35),
}

# Machines actually in use at each stage (names as per the factory workbook).
STAGE_MACHINES = {
    "labdip": ["Lab Beaker Dyer", "Infrared Lab Dyer", "Mathis Labomat"],
    "s1kg": ["Sample Aventec", "Sample Jet 1 kg", "Sample Soft-flow 1 kg"],
    "s7kg": ["Sample Tonello", "Sample Soft-flow Jet 7 kg", "Sample Aventec 7 kg"],
    "minibulk": ["Bataji Soft-flow", "Soft-flow Jet 50 kg", "Thies Mini Jet"],
    "bulk": ["Industrial Soft-flow Jet", "Aventec Bulk", "High-Capacity Winch",
             "Thies Eco-Bloc"],
}

# Machine running cost, USD per hour, by stage (occupancy cost of a trial).
STAGE_MACHINE_USD_PER_HR = {
    "labdip": 1.80,
    "s1kg": 4.50,
    "s7kg": 9.00,
    "minibulk": 16.00,
    "bulk": 25.00,
}

# Fixed overhead charged to every attempt at a stage (labour, lab QC, colour
# measurement, re-handling), USD.
STAGE_FIXED_OVERHEAD_USD = {
    "labdip": 2.50,
    "s1kg": 6.00,
    "s7kg": 12.00,
    "minibulk": 28.00,
    "bulk": 55.00,
}

# ---------------------------------------------------------------------------
# 2. Substrate (fabric) domain
# ---------------------------------------------------------------------------
FABRIC_STRUCTURES = ["Single Jersey", "Interlock", "Rib", "Woven", "Fleece"]
FABRIC_STRUCTURE_P = [0.34, 0.18, 0.16, 0.20, 0.12]

# GSM window per structure (g/m2); overall brief window is 120-320.
STRUCTURE_GSM_RANGE = {
    "Single Jersey": (120, 200),
    "Interlock": (170, 260),
    "Rib": (180, 280),
    "Woven": (130, 240),
    "Fleece": (240, 320),
}

COMPOSITIONS = ["100% Cotton", "Cotton-Poly (60/40)", "Cotton-Poly (80/20)",
                "Viscose", "Modal"]
COMPOSITION_P = [0.46, 0.14, 0.16, 0.14, 0.10]

# Cellulosic fraction: only the cellulose takes the reactive dye.
COMPOSITION_CELLULOSE_FRACTION = {
    "100% Cotton": 1.00,
    "Cotton-Poly (60/40)": 0.60,
    "Cotton-Poly (80/20)": 0.80,
    "Viscose": 1.00,
    "Modal": 1.00,
}

# Relative dye affinity / accessibility of the substrate. Above 1 means the
# fibre picks up colour more readily, so LESS dye is needed for the same depth.
COMPOSITION_AFFINITY = {
    "100% Cotton": 1.00,
    "Cotton-Poly (60/40)": 0.86,
    "Cotton-Poly (80/20)": 0.93,
    "Viscose": 1.14,     # regenerated cellulose: high accessibility
    "Modal": 1.09,
}

PRETREATMENTS = ["Bleached", "Scoured", "Causticized", "Mercerized"]
PRETREATMENT_P = [0.44, 0.26, 0.14, 0.16]

# Mercerizing / causticizing swells the fibre and raises dye uptake.
PRETREATMENT_AFFINITY = {
    "Bleached": 1.00,
    "Scoured": 0.96,
    "Causticized": 1.06,
    "Mercerized": 1.11,
}

SPECIAL_FINISHES = ["None", "Enzyme Wash", "Silicone Softener", "Catanizing",
                    "Peach Finish"]
SPECIAL_FINISH_P = [0.52, 0.16, 0.14, 0.10, 0.08]

# ---------------------------------------------------------------------------
# 3. Shade depth classes
# ---------------------------------------------------------------------------
DEPTH_CLASSES = ["Pastel/Light", "Medium", "Dark", "Extra Dark"]

# Total-dye-concentration window (% on weight of fabric) for each class.
DEPTH_OWF_RANGE = {
    "Pastel/Light": (0.05, 0.50),
    "Medium": (0.50, 1.50),
    "Dark": (1.50, 3.50),
    "Extra Dark": (3.50, 7.00),
}

# The supplied shade dictionary uses 11 descriptive depth labels; map them onto
# the 4 canonical classes. The en-dash in the source file is normalised to "-"
# before lookup.
DEPTH_LABEL_MAP = {
    "very light": "Pastel/Light",
    "light": "Pastel/Light",
    "light-medium": "Medium",
    "medium": "Medium",
    "bright": "Medium",
    "medium-bright": "Medium",
    "medium-dark": "Dark",
    "dark": "Dark",
    "light-dark": "Dark",
    "deep": "Extra Dark",
    "medium-deep": "Extra Dark",
    "pastel/light": "Pastel/Light",
    "extra dark": "Extra Dark",
}

# Salt (Glauber salt) g/L window by depth class: electrolyte demand rises with
# depth to overcome the Donnan potential at the fibre surface.
DEPTH_SALT_G_L = {
    "Pastel/Light": (10, 25),
    "Medium": (25, 45),
    "Dark": (45, 65),
    "Extra Dark": (60, 80),
}

# Soda ash g/L window by depth class (alkali for fixation).
DEPTH_SODA_G_L = {
    "Pastel/Light": (5, 10),
    "Medium": (8, 15),
    "Dark": (14, 20),
    "Extra Dark": (18, 25),
}

# ---------------------------------------------------------------------------
# 4. Dye chemistry
# ---------------------------------------------------------------------------
# 'temp_c' is the fixation temperature window; 'K' is the dimensionless
# partition/affinity constant used in the exhaustion model (larger K => higher
# intrinsic substantivity => flatter response to liquor-ratio change).
DYE_CLASSES = {
    "Reactive Vinyl Sulphone": {
        "p": 0.44, "temp_c": (58, 64), "time_min": (45, 75), "K": 11.0,
        "trade_prefix": "REMAZOL",
    },
    "Reactive Monochlorotriazine": {
        "p": 0.26, "temp_c": (76, 84), "time_min": (55, 90), "K": 14.5,
        "trade_prefix": "CIBACRON",
    },
    "Reactive Bifunctional (ED/RGB)": {
        "p": 0.20, "temp_c": (58, 66), "time_min": (50, 85), "K": 13.0,
        "trade_prefix": "NOVACRON",
    },
    "Pigment / Direct": {
        "p": 0.10, "temp_c": (66, 74), "time_min": (40, 60), "K": 20.0,
        "trade_prefix": "EMACOAL CT",
    },
}

# Trichromatic primaries. 'price' = USD/kg landed, 'strength' = relative
# colouristic strength used when apportioning a shade into primaries.
DYE_PRIMARIES = {
    "Yellow":    {"price_usd_kg": 12.80, "strength": 1.00},
    "Red":       {"price_usd_kg": 14.50, "strength": 0.94},
    "Blue":      {"price_usd_kg": 16.20, "strength": 0.88},
    "Black":     {"price_usd_kg": 8.50,  "strength": 1.22},
    "Orange":    {"price_usd_kg": 13.40, "strength": 0.97},
    "Navy":      {"price_usd_kg": 11.90, "strength": 1.10},
    "Turquoise": {"price_usd_kg": 21.50, "strength": 0.80},
}

# Which primaries a shade family is built from, and the nominal mixing
# proportion of each. Proportions are perturbed per record.
FAMILY_PRIMARY_MIX = {
    "Black":     [("Black", 0.86), ("Navy", 0.09), ("Red", 0.05)],
    "Grey":      [("Black", 0.80), ("Blue", 0.13), ("Red", 0.07)],
    "Blue":      [("Blue", 0.70), ("Turquoise", 0.16), ("Red", 0.08), ("Yellow", 0.06)],
    "Navy":      [("Navy", 0.62), ("Blue", 0.25), ("Black", 0.13)],
    "Green":     [("Yellow", 0.47), ("Blue", 0.37), ("Turquoise", 0.11), ("Black", 0.05)],
    "Teal":      [("Turquoise", 0.52), ("Blue", 0.31), ("Yellow", 0.17)],
    "Turquoise": [("Turquoise", 0.74), ("Blue", 0.19), ("Yellow", 0.07)],
    "Red":       [("Red", 0.76), ("Yellow", 0.17), ("Blue", 0.07)],
    "Pink":      [("Red", 0.82), ("Yellow", 0.10), ("Blue", 0.08)],
    "Purple":    [("Red", 0.49), ("Blue", 0.44), ("Yellow", 0.07)],
    "Violet":    [("Blue", 0.52), ("Red", 0.42), ("Black", 0.06)],
    "Orange":    [("Orange", 0.56), ("Yellow", 0.33), ("Red", 0.11)],
    "Yellow":    [("Yellow", 0.86), ("Orange", 0.10), ("Red", 0.04)],
    "Brown":     [("Yellow", 0.40), ("Red", 0.32), ("Blue", 0.18), ("Black", 0.10)],
    "Beige":     [("Yellow", 0.52), ("Red", 0.27), ("Blue", 0.14), ("Black", 0.07)],
    "Olive":     [("Yellow", 0.44), ("Blue", 0.28), ("Black", 0.17), ("Red", 0.11)],
    "Charcoal":  [("Black", 0.84), ("Blue", 0.11), ("Yellow", 0.05)],
    "White":     [("Blue", 0.55), ("Red", 0.25), ("Yellow", 0.20)],  # optical tinting
}

# Shade families that are notoriously hard to hit right-first-time (metameric,
# steep dye-uptake curves, or delicate multi-dye tonal balance).
FAMILY_CRITICALITY = {
    "Navy": 0.95, "Charcoal": 0.92, "Olive": 0.90, "Grey": 0.85, "Teal": 0.82,
    "Brown": 0.80, "Beige": 0.78, "Purple": 0.74, "Violet": 0.74, "Green": 0.68,
    "Turquoise": 0.66, "Blue": 0.58, "Black": 0.50, "Red": 0.45, "Pink": 0.42,
    "Orange": 0.38, "Yellow": 0.32, "White": 0.25,
}

# ---------------------------------------------------------------------------
# 5. Scale-up adjustment-factor model coefficients
# ---------------------------------------------------------------------------
# The net recipe correction factor from lab dip to stage s is modelled as
#
#     f_s = f_LR * f_shear * f_depth * f_substrate * f_machine * eps
#
# See RESEARCH_PAPER.md Section 3 for the derivation of each term.

# Exponent damping the pure exhaustion term, because the dyer simultaneously
# re-optimises the electrolyte at each stage, which partially cancels the
# liquor-ratio exhaustion gain. gamma = 1 would be "no compensation at all".
GAMMA_LR_DAMPING = 0.30

# Mechanical-shear coefficient: extra agitation in bulk machines increases
# hydrolysed-dye removal during wash-off, so slightly more dye is required.
C_SHEAR = 0.0022          # per RPM above the lab-dip baseline

# Depth-dependent compensation: in deep shades the bath approaches saturation,
# dye-dye aggregation rises and marginal exhaustion falls.
DEPTH_COMPENSATION = {
    "Pastel/Light": -0.012,
    "Medium": 0.010,
    "Dark": 0.034,
    "Extra Dark": 0.052,
}

# Diffusion penalty for heavy / dense constructions, per 100 gsm above 150.
C_GSM = 0.011

# Machine-family bias (soft-flow vs winch vs beaker hydrodynamics).
MACHINE_BIAS = {
    "Industrial Soft-flow Jet": 0.995,
    "Aventec Bulk": 1.004,
    "High-Capacity Winch": 1.021,
    "Thies Eco-Bloc": 0.991,
    "Bataji Soft-flow": 1.000,
    "Soft-flow Jet 50 kg": 0.997,
    "Thies Mini Jet": 1.006,
    "Sample Tonello": 1.008,
    "Sample Soft-flow Jet 7 kg": 0.998,
    "Sample Aventec 7 kg": 1.003,
    "Sample Aventec": 1.002,
    "Sample Jet 1 kg": 1.000,
    "Sample Soft-flow 1 kg": 0.997,
    "Lab Beaker Dyer": 1.000,
    "Infrared Lab Dyer": 1.000,
    "Mathis Labomat": 1.000,
}

# Irreducible process noise on the adjustment factor (lognormal sigma).
SIGMA_PROCESS = 0.016

# Sanity clamp on the realised adjustment factor: real mills re-engineer the
# shade rather than accept a factor outside this band.
ADJ_FACTOR_CLAMP = (0.78, 1.28)

# ---------------------------------------------------------------------------
# 6. Trial / Right-First-Time model
# ---------------------------------------------------------------------------
# UNIT CONVENTION. The brief quotes two trial figures that cannot both describe
# the same quantity ("3.8 -> 1.2 trials per shade", and "reduce trials from 4 to
# 1 at each stage"). Taking 3.8 as attempts PER STAGE implies 19 dyeings per
# job, which contradicts the brief's own "3-5 iterative trials across the
# intermediate stages". This platform therefore adopts the coherent reading:
#
#   corrective_trials = attempts beyond the first, SUMMED over the five stages
#                       baseline 3.8 per shade  ->  target 1.2 per shade
#   attempts per stage = 1 + corrective/5        baseline 1.76 -> target 1.24
#   rft_bulk           = share of bulk batches needing only one dyeing
#                       baseline 40%            ->  target >= 85%
#
# The brief's "4 -> 1" is reproduced as the attempt count at a difficult stage
# for a hard shade, which the right tail of this model still generates.
BASELINE_RFT_BULK = 0.40                  # bulk batches right first time
BASELINE_CORRECTIVE_TRIALS_PER_JOB = 3.8  # summed over the five stages
TARGET_RFT_BULK = 0.85
TARGET_CORRECTIVE_TRIALS_PER_JOB = 1.2

# Relative trial propensity by stage.
#
# Calibrating to BOTH industrial baselines above forces the bulk stage to be the
# hardest one: a 40% bulk RFT rate alone consumes 0.92 of the 3.8 corrective-
# trial budget. That is not an artefact, it is this project's premise - a lab
# recipe does not translate to bulk. The ordering of the remaining weights
# follows the magnitude of the liquor-ratio contraction at each transition
# (20 -> 17.5 -> 13.5 -> 11.0 -> 8.0 L/kg), with the lab dip carrying its own
# load because it is a de-novo match against the customer's colour standard
# rather than a carry-over. The bulk weight is solved numerically, not asserted.
STAGE_TRIAL_WEIGHT = {
    "labdip": 1.15,     # de-novo colour matching against the standard
    "s1kg": 0.80,       # smallest liquor-ratio step; recipe essentially carried over
    "s7kg": 0.95,       # largest intermediate contraction (17.5 -> 13.5)
    "minibulk": 0.85,   # first production-type machine
    "bulk": 1.25,       # largest contraction + sample-to-production machine change
}

# Difficulty -> Poisson rate mapping:  lambda = k * w_stage * (A + B * d)
TRIAL_DIFFICULTY_A = 0.30
TRIAL_DIFFICULTY_B = 1.95
TRIAL_MAX = 7                       # attempts are capped in practice

# Weights that build the job difficulty score d in [0, 1].
DIFFICULTY_WEIGHTS = {
    "depth": 0.26,         # Extra Dark is hardest
    "n_dyes": 0.22,        # 4-dye tonal balance is hardest
    "family": 0.20,        # critical shade families
    "repeat": 0.18,        # a proven repeat shade is easy
    "gsm": 0.07,
    "finish": 0.07,
}

P_REPEAT_SHADE = 0.34      # share of jobs that are proven repeat shades

# Colour-acceptance tolerance: a stage needs another attempt when the realised
# recipe deviates from the approved recipe by more than this relative amount
# (roughly equivalent to dE_cmc <= 1.0 for a medium-depth shade).
RECIPE_TOLERANCE_REL = 0.030

# ---------------------------------------------------------------------------
# 7. Auxiliary chemicals, utilities & effluent
# ---------------------------------------------------------------------------
CHEMICAL_PRICES_USD_KG = {
    "Glauber Salt": 0.35,
    "Soda Ash": 0.55,
    "Acetic Acid": 1.20,
    "Soaping Agent": 2.40,
    "Leveling Agent": 2.10,
    "Sequestering Agent": 1.75,
    "Wetting Agent": 1.95,
}

# Auxiliary dosing, g/L of bath (independent of depth class).
AUX_DOSING_G_L = {
    "Leveling Agent": 1.0,
    "Sequestering Agent": 0.8,
    "Wetting Agent": 0.6,
    "Acetic Acid": 0.5,
    "Soaping Agent": 1.5,
}

UTILITIES = {
    "water_usd_per_l": 0.003,
    "electricity_usd_per_kwh": 0.11,
    "steam_usd_per_kwh_thermal": 0.042,
    "effluent_usd_per_m3": 0.95,
    "boiler_efficiency": 0.80,
    "ambient_temp_c": 30.0,
    "water_cp_kj_kg_k": 4.186,
    "n_rinse_baths": 4,            # rinses + soaping + neutralisation
    "rinse_volume_factor": 0.85,   # each rinse uses this fraction of bath volume
}

# Non-dyeing time added to dyeing time to get machine occupancy, minutes.
STAGE_HANDLING_MIN = {
    "labdip": 25,
    "s1kg": 45,
    "s7kg": 60,
    "minibulk": 80,
    "bulk": 105,
}

# ---------------------------------------------------------------------------
# 8. Factory / investment assumptions for the techno-economic analysis
# ---------------------------------------------------------------------------
FACTORY = {
    "annual_output_kg": 100_000.0,
    "avg_bulk_batch_kg": 250.0,
    "distinct_shades_per_year": 420,
    "working_days": 300,
}

INVESTMENT = {
    "capex_usd": 46_000.0,        # platform build, spectrophotometer integration, MES hooks
    "annual_opex_usd": 8_400.0,   # hosting, retraining, licence, support
    "discount_rate": 0.12,
    "horizon_years": 5,
}

# ---------------------------------------------------------------------------
# 9. ML target definitions
# ---------------------------------------------------------------------------
TARGETS_BULK = [
    "bulk_dye1_owf", "bulk_dye2_owf", "bulk_dye3_owf", "bulk_dye4_owf",
    "bulk_salt_g_l", "bulk_soda_g_l", "bulk_trials",
]


def stage_targets(stage: str) -> list:
    """Per-stage target vector used by the sequential multi-stage track."""
    return [
        f"{stage}_dye1_owf", f"{stage}_dye2_owf", f"{stage}_dye3_owf",
        f"{stage}_dye4_owf", f"{stage}_salt_g_l", f"{stage}_soda_g_l",
        f"{stage}_trials",
    ]


# ---------------------------------------------------------------------------
# 10. Paths
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "outputs")
FIG_DIR = os.path.join(OUT_DIR, "figures")
TBL_DIR = os.path.join(OUT_DIR, "tables")
MODEL_DIR = os.path.join(OUT_DIR, "models")

MASTER_CSV = os.path.join(ROOT, "textile_dyeing_scaleup_master.csv")
SHADE_MASTER_CSV = os.path.join(DATA_DIR, "shade_master.csv")
FACTORY_FORMAT_CSV = os.path.join(DATA_DIR, "textile_dyeing_scaleup_factory_format.csv")

for _d in (DATA_DIR, OUT_DIR, FIG_DIR, TBL_DIR, MODEL_DIR):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# 11. Robust file output
# ---------------------------------------------------------------------------
def write_csv(df, path: str, retries: int = 6, **kwargs) -> str:
    """
    Write a DataFrame to CSV, tolerating the transient file locks that OneDrive
    (and antivirus scanners) place on freshly created files in a synced folder.

    Retries with a short backoff; if the target stays locked, writes alongside
    it with a '.new.csv' suffix and returns the path actually used, so a run is
    never lost to a sync lock.
    """
    import time

    kwargs.setdefault("index", False)
    last = None
    for attempt in range(retries):
        try:
            df.to_csv(path, **kwargs)
            return path
        except PermissionError as exc:
            last = exc
            time.sleep(0.4 * (attempt + 1))
    alt = path[:-4] + ".new.csv" if path.lower().endswith(".csv") else path + ".new"
    try:
        df.to_csv(alt, **kwargs)
        print(f"[write_csv] '{os.path.basename(path)}' was locked; wrote "
              f"'{os.path.basename(alt)}' instead")
        return alt
    except PermissionError:
        raise last


# ---------------------------------------------------------------------------
# 12. Dataset location
# ---------------------------------------------------------------------------
MASTER_CSV_GZ = os.path.join(DATA_DIR, "textile_dyeing_scaleup_master.csv.gz")


def master_path() -> str:
    """
    Where the master dataset actually is.

    The full CSV is 8.7 MB and is produced locally by the generator; a gzipped
    copy (3.1 MB) is what gets committed for deployment. pandas reads either
    transparently, so callers just use this.
    """
    if os.path.exists(MASTER_CSV):
        return MASTER_CSV
    if os.path.exists(MASTER_CSV_GZ):
        return MASTER_CSV_GZ
    raise FileNotFoundError(
        "Master dataset not found. Run: python src/data_generator.py")
