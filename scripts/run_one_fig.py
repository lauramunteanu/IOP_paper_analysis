"""Driver helper: import a figure script, force a non-interactive backend,
and save whatever figures the script creates to a target directory.

usage:  python3 run_one_fig.py <script.py> <out_dir> [<basename>]
"""
import os
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ROOT in batch mode (some of the figure scripts pull ROOT in via FlatTreeMod
# and call gROOT.SetBatch(True) themselves, but better safe than sorry)
try:
    import ROOT
    ROOT.gROOT.SetBatch(True)
except ImportError:
    pass


def main():
    if len(sys.argv) < 3:
        print("usage: run_one_fig.py <script.py> <out_dir> [<basename>]")
        sys.exit(2)
    script = Path(sys.argv[1]).resolve()
    out_dir = Path(sys.argv[2]).resolve()
    base = sys.argv[3] if len(sys.argv) > 3 else script.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    # Run the script in its own dir so its relative paths resolve like
    # `python3 Fig3_xxx.py` did before.
    cwd = os.getcwd()
    os.chdir(script.parent)
    sys.path.insert(0, str(script.parent))

    success = False
    err_msg = None
    try:
        import runpy
        runpy.run_path(str(script), run_name="__main__")
        success = True
    except SystemExit as e:
        if e.code in (None, 0):
            success = True
        else:
            err_msg = f"SystemExit({e.code})"
    except Exception:
        err_msg = traceback.format_exc(limit=4)

    n_saved = 0
    for i, num in enumerate(plt.get_fignums()):
        fig = plt.figure(num)
        out = out_dir / f"{base}_{i}.png"
        try:
            fig.savefig(out, bbox_inches="tight", dpi=150)
            n_saved += 1
            print(f"  saved {out}")
        except Exception as e:
            print(f"  failed to save {out}: {e}")
    plt.close("all")
    os.chdir(cwd)

    if not success:
        print(f"  ERROR in {script.name}:\n{err_msg}")
        sys.exit(1)
    if n_saved == 0:
        print(f"  WARN: {script.name} ran but produced no figures")
    sys.exit(0)


if __name__ == "__main__":
    main()
