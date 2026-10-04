"""
run_all.py
==========
Reproduce the entire study end to end, in order.

    python run_all.py                 full run, with hyper-parameter search
    python run_all.py --fast          skip the search (minutes instead of tens)

Each step writes its own artefacts and can also be run on its own; this script
only guarantees the ordering and reports what was produced.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

import config as C                       # noqa: E402


STEPS = [
    ("1. Synthetic dataset generation", "data_generator", "main", {}),
    ("2. Exploratory data analysis", "eda", "main", {}),
    ("3-4. ML benchmark & validation", "ml_pipeline", "main", {"tune": True}),
    ("   Per-stage production models", "train_stage_models", "main", {}),
    ("5. Techno-economic analysis", "cost_benefit", "main", {}),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true",
                    help="skip hyper-parameter search in step 3")
    args = ap.parse_args()

    t_start = time.time()
    for title, module_name, fn_name, kwargs in STEPS:
        if args.fast and module_name == "ml_pipeline":
            kwargs = {"tune": False}
        print("\n" + "#" * 78)
        print(f"# {title}")
        print("#" * 78)
        t0 = time.time()
        module = __import__(module_name)
        getattr(module, fn_name)(**kwargs)
        print(f"\n[run_all] '{title}' finished in {time.time() - t0:,.1f}s")

    print("\n" + "=" * 78)
    print(f"ALL STEPS COMPLETE in {time.time() - t_start:,.1f}s")
    print("=" * 78)
    print("\nArtefacts:")
    print(f"  dataset   {os.path.basename(C.MASTER_CSV)}")
    for folder in (C.FIG_DIR, C.TBL_DIR, C.MODEL_DIR):
        rel = os.path.relpath(folder, C.ROOT)
        files = sorted(os.listdir(folder)) if os.path.isdir(folder) else []
        print(f"  {rel:<18s} {len(files)} files")
    print("\nLaunch the prototype with:  streamlit run app.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
