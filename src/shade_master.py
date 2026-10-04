"""
shade_master.py
===============
Builds the canonical shade master table used by the data generator and the
Streamlit app.

Primary source is the factory's own 499-row shade dictionary
(500_Textile_Dyeing_Shades.xlsx). If that workbook is not available the module
falls back to a procedurally generated 500-shade dictionary with the same
schema, so the whole pipeline remains runnable standalone.

Output schema (data/shade_master.csv)
-------------------------------------
shade_no, shade_code, shade_name, shade_family, depth_label, depth_class,
colour_direction, n_dyes_typical, criticality
"""

from __future__ import annotations

import os
import re
import unicodedata

import numpy as np
import pandas as pd

try:
    from . import config as C
except ImportError:  # run as a plain script
    import config as C


# The workbook is looked for in data/ first: files sitting directly in a synced
# OneDrive folder are intermittently lock-protected, so a local copy under
# data/ is the reliable read path.
SOURCE_CANDIDATES = [
    os.path.join(C.DATA_DIR, "500_Textile_Dyeing_Shades.xlsx"),
    os.path.join(C.ROOT, "500_Textile_Dyeing_Shades.xlsx"),
]


def _find_source() -> str | None:
    for p in SOURCE_CANDIDATES:
        if os.path.exists(p):
            return p
    return None

# Families present in the supplied dictionary but not keyed in config are
# mapped onto the nearest family that has a primary mix defined.
FAMILY_ALIASES = {
    "white": "White", "black": "Black", "grey": "Grey", "gray": "Grey",
    "blue": "Blue", "navy": "Navy", "green": "Green", "olive": "Olive",
    "teal": "Teal", "turquoise": "Turquoise", "red": "Red", "pink": "Pink",
    "purple": "Purple", "violet": "Violet", "orange": "Orange",
    "yellow": "Yellow", "brown": "Brown", "beige": "Beige",
    "charcoal": "Charcoal", "earth": "Brown", "earth/brown": "Brown",
}


def _ascii(s: str) -> str:
    """Normalise unicode dashes/spaces that appear in the source workbook."""
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFKD", s)
    s = s.replace("–", "-").replace("—", "-").replace("−", "-")
    # The workbook was saved with a legacy codepage; stray bytes land as U+FFFD.
    s = s.replace("�", "-")
    return re.sub(r"\s+", " ", s).strip()


def _norm_depth(label: str) -> str:
    """Map one of the 11 descriptive depth labels onto a canonical class."""
    key = _ascii(label).lower().replace(" - ", "-").replace(" -", "-").replace("- ", "-")
    key = key.replace(" ", "-") if "-" in key else key
    if key in C.DEPTH_LABEL_MAP:
        return C.DEPTH_LABEL_MAP[key]
    # tolerate e.g. "medium dark"
    key2 = _ascii(label).lower().replace(" ", "-")
    if key2 in C.DEPTH_LABEL_MAP:
        return C.DEPTH_LABEL_MAP[key2]
    # last resort: longest keyword match
    for kw in ("very light", "deep", "dark", "bright", "medium", "light"):
        if kw in _ascii(label).lower():
            return C.DEPTH_LABEL_MAP.get(kw.replace(" ", "-"), "Medium")
    return "Medium"


def _norm_family(label: str) -> str:
    key = _ascii(label).lower()
    if key in FAMILY_ALIASES:
        return FAMILY_ALIASES[key]
    for alias, fam in FAMILY_ALIASES.items():
        if alias in key:
            return fam
    return "Grey"


def _shade_code(name: str, i: int) -> str:
    """Factory-style shade code, e.g. SH-0123-NAV."""
    letters = re.sub(r"[^A-Za-z]", "", _ascii(name)).upper()
    tag = (letters + "XXX")[:3]
    return f"SH-{i:04d}-{tag}"


def _typical_n_dyes(family: str, depth_class: str, rng: np.random.Generator) -> int:
    """
    How many primaries a shade of this family/depth is normally built from.
    Deep neutral shades and tertiary families need more primaries.
    """
    base = len(C.FAMILY_PRIMARY_MIX.get(family, [("Black", 1.0)]))
    if depth_class in ("Dark", "Extra Dark"):
        base = min(4, base + 1)
    if family in ("Yellow", "Pink", "White"):
        base = max(1, base - 1)
    # +/- 1 jitter, clipped to [1, 4]
    n = int(np.clip(base + rng.integers(-1, 1, endpoint=True), 1, 4))
    return n


def _fallback_dictionary(rng: np.random.Generator) -> pd.DataFrame:
    """Procedural 500-shade dictionary, used only if the workbook is absent."""
    adjectives = [
        "Midnight", "Desert", "Arctic", "Dusty", "Deep", "Pale", "Burnt", "Faded",
        "Royal", "Antique", "Smoked", "Washed", "Electric", "Soft", "Rich",
        "Muted", "Vivid", "Shadow", "Sun", "Storm", "Mineral", "Forest",
        "Ocean", "Sand", "Coral", "Ash", "Ink", "Copper", "Slate", "Moss",
    ]
    nouns = {
        "Blue": ["Sky", "Denim", "Cobalt", "Indigo", "Sapphire", "Lagoon"],
        "Green": ["Fern", "Pine", "Jade", "Mint", "Basil", "Emerald"],
        "Red": ["Rose", "Scarlet", "Cherry", "Brick", "Ruby", "Crimson"],
        "Yellow": ["Corn", "Lemon", "Amber", "Mustard", "Honey", "Straw"],
        "Orange": ["Apricot", "Rust", "Tangerine", "Clay", "Papaya", "Ember"],
        "Purple": ["Plum", "Orchid", "Mauve", "Lilac", "Grape", "Wine"],
        "Violet": ["Iris", "Amethyst", "Heather", "Thistle"],
        "Olive": ["Olive", "Khaki", "Sage", "Moss", "Reed"],
        "Navy": ["Navy", "Marine", "Abyss", "Fleet"],
        "Charcoal": ["Charcoal", "Graphite", "Carbon", "Anthracite"],
        "Brown": ["Chocolate", "Coffee", "Chestnut", "Walnut", "Toffee"],
        "Beige": ["Beige", "Oat", "Linen", "Taupe", "Biscuit", "Ecru"],
        "Grey": ["Pewter", "Smoke", "Steel", "Fog", "Granite"],
        "Black": ["Black", "Onyx", "Raven", "Ebony", "Obsidian"],
        "Pink": ["Blush", "Fuchsia", "Peony", "Candy", "Petal"],
        "Teal": ["Teal", "Lagoon", "Verdigris"],
        "Turquoise": ["Turquoise", "Aqua", "Cyan"],
        "White": ["White", "Snow", "Ivory", "Chalk"],
    }
    fams = list(nouns.keys())
    rows, seen = [], set()
    i = 0
    while len(rows) < 500:
        fam = fams[i % len(fams)]
        name = f"{rng.choice(adjectives)} {rng.choice(nouns[fam])}"
        if name in seen:
            i += 1
            continue
        seen.add(name)
        depth_label = rng.choice(
            ["Very Light", "Light", "Light-Medium", "Medium", "Bright",
             "Medium-Dark", "Dark", "Deep"],
            p=[0.05, 0.14, 0.11, 0.22, 0.08, 0.14, 0.18, 0.08],
        )
        rows.append({
            "shade_no": len(rows) + 1,
            "shade_name": name,
            "shade_family": fam,
            "depth_label": depth_label,
            "colour_direction": f"{fam.lower()} direction",
        })
        i += 1
    return pd.DataFrame(rows)


def build_shade_master(force: bool = False) -> pd.DataFrame:
    """Load (or build) and cache the shade master table."""
    if (not force) and os.path.exists(C.SHADE_MASTER_CSV):
        return pd.read_csv(C.SHADE_MASTER_CSV)

    rng = np.random.default_rng(C.RANDOM_SEED)

    df = None
    src = _find_source()
    if src:
        try:
            raw = pd.read_excel(src)
            cols = {_ascii(c).lower(): c for c in raw.columns}
            name_c = cols.get("shade name")
            fam_c = cols.get("shade family")
            dep_c = cols.get("depth class")
            dir_c = cols.get("typical colour direction") or cols.get("typical color direction")
            if name_c and fam_c and dep_c:
                df = pd.DataFrame({
                    "shade_no": range(1, len(raw) + 1),
                    "shade_name": raw[name_c].map(_ascii),
                    "shade_family": raw[fam_c].map(_ascii),
                    "depth_label": raw[dep_c].map(_ascii),
                    "colour_direction": (raw[dir_c].map(_ascii) if dir_c else ""),
                })
                print(f"[shade_master] loaded {len(df)} shades from {os.path.basename(src)}")
        except Exception as exc:        # permission / corrupt workbook
            print(f"[shade_master] could not read workbook ({exc.__class__.__name__}: {exc});"
                  " falling back to procedural dictionary")
            df = None

    if df is None:
        df = _fallback_dictionary(rng)
        print(f"[shade_master] built procedural dictionary of {len(df)} shades")

    df["shade_family"] = df["shade_family"].map(_norm_family)
    df["depth_class"] = df["depth_label"].map(_norm_depth)
    df["shade_code"] = [_shade_code(n, i) for i, n in
                        zip(df["shade_no"], df["shade_name"])]
    df["n_dyes_typical"] = [
        _typical_n_dyes(f, d, rng)
        for f, d in zip(df["shade_family"], df["depth_class"])
    ]
    df["criticality"] = df["shade_family"].map(C.FAMILY_CRITICALITY).fillna(0.6)

    df = df[["shade_no", "shade_code", "shade_name", "shade_family",
             "depth_label", "depth_class", "colour_direction",
             "n_dyes_typical", "criticality"]]
    df.to_csv(C.SHADE_MASTER_CSV, index=False)
    return df


if __name__ == "__main__":
    d = build_shade_master(force=True)
    print(d.shape)
    print(d.head(10).to_string(index=False))
    print("\nFamilies:\n", d["shade_family"].value_counts().to_string())
    print("\nDepth classes:\n", d["depth_class"].value_counts().to_string())
