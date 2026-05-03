"""validation_distributions.py — per-variant bias distributions that feed
the BW summary plots in fsi_iop_plots.py.

Multi-page PDF, one page per (category, flavour, observable, variant). Each
page shows the weighted bias histogram with the 1σ (16-84%) region shaded,
mean and median lines, the source filename printed at the top, and the key
numerical stats in a corner.

Uses the EXACT same readers (read_hk / read_dune / get_obs), stats helpers
(weighted_percentile / weighted_mean / stats), and variant catalogue from
fsi_iop_plots.py — any change there propagates here so the BW plots and the
validation pages stay in lock-step.

Output: <OUT_DIR>/bw_validation_distributions.pdf
"""
import os
import sys

# Make sure laura_extras/ is on sys.path so we can import fsi_iop_plots.
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fsi_iop_plots import (
    get_obs, get_obs_with_mode, resolve_path,
    stats, weighted_percentile,
    OBS_LABELS, hist_params_for,
    OUT_DIR,
    CATEGORY_VARIANTS,
    NU_LABELS,
)

OUT_PDF = f"{OUT_DIR}/bw_validation_distributions.pdf"


def plot_one_page(pdf, *, cat_label, flav, obs_key, var_label, color, source_path):
    """One validation page: histogram + 1σ shading + mean/median lines.
    `source_path` may be None or unresolvable — in that case prints a
    placeholder page so the PDF still has the slot."""
    nbins, hrange = hist_params_for(obs_key)
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    resolved = resolve_path(source_path)
    title_top = (f"{cat_label}  |  {NU_LABELS[flav]}  |  {OBS_LABELS[obs_key]}  |  "
                 f"variant: {var_label}")
    if resolved is None:
        ax.text(0.5, 0.5, f"file not found:\n{source_path or '(unset)'}",
                ha="center", va="center", transform=ax.transAxes,
                color="red")
        ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title_top + "\n(missing)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    res = get_obs_with_mode(resolved, obs_key)
    if res is None or len(res[0]) == 0:
        ax.text(0.5, 0.5, "no events passed selection",
                ha="center", va="center", transform=ax.transAxes,
                color="red")
        fig.suptitle(title_top)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    x, w, mode = res
    x = np.asarray(x); w = np.asarray(w); mode = np.asarray(mode)
    s = stats(x, w)

    # Stacked histogram broken down by |Mode|. Each NEUT/NuWro Mode gets
    # its own colour; modes are grouped by absolute value so ν and ν̄
    # share a colour scheme. Bin contents = Σ w_i / Δ_bin (dσ/dE units).
    bin_width = (hrange[1] - hrange[0]) / nbins
    abs_mode = np.abs(mode)
    MODE_GROUPS = [
        (1,  "CCQE",            "#0077BB"),  # Tol blue
        (2,  "CC2p2h",          "#009988"),  # Tol teal
        (11, r"CC RES $p+\pi^+$","#EE7733"), # Tol orange
        (12, r"CC RES $p+\pi^0$","#EE3377"), # Tol magenta
        (13, r"CC RES $n+\pi^+$","#33BBEE"), # Tol cyan
        (16, r"CC coherent $\pi$","#882255"),# Tol burgundy
        (21, r"CC multi-$\pi$",  "#AA4499"), # Tol purple
        (26, "CC DIS",           "#CC3311"), # Tol red
    ]
    known = {m for m, _, _ in MODE_GROUPS}

    data_list, weight_list, labels_list, colors_list = [], [], [], []
    for m, lbl, col in MODE_GROUPS:
        sel = abs_mode == m
        n = int(sel.sum())
        if n == 0:
            continue
        data_list.append(x[sel])
        weight_list.append(w[sel] / bin_width)
        labels_list.append(f"{lbl}  ({n})")
        colors_list.append(col)
    sel_other = ~np.isin(abs_mode, list(known))
    n_other = int(sel_other.sum())
    if n_other > 0:
        data_list.append(x[sel_other])
        weight_list.append(w[sel_other] / bin_width)
        labels_list.append(f"other  ({n_other})")
        colors_list.append("#888888")

    counts_stack = ax.hist(
        data_list, bins=nbins, range=hrange, weights=weight_list,
        histtype="stepfilled", stacked=True, alpha=0.80,
        color=colors_list, label=labels_list,
        edgecolor="white", linewidth=0.2,
    )
    # ymax from the topmost stacked layer (counts_stack[0] is a list of arrays
    # cumulative through the stack).
    if isinstance(counts_stack[0], list):
        top = counts_stack[0][-1]
    else:
        top = counts_stack[0]
    ymax = top.max() if hasattr(top, "max") and top.max() > 0 else 1.0

    # 1σ band (16-84%)
    ax.axvspan(s["p16"], s["p84"], alpha=0.20, color="orange",
               label=fr"1$\sigma$ (16-84): [{s['p16']:.1f}, {s['p84']:.1f}] MeV")

    # mean / median lines
    ax.axvline(s["median"], color="black", lw=1.6,
               label=f"median = {s['median']:+.1f} MeV")
    ax.axvline(s["mean"], color="red", lw=1.6, linestyle="--",
               label=f"mean = {s['mean']:+.1f} MeV")

    # 90% whiskers (5-95%) as thinner outer lines for context
    ax.axvline(s["p5"],  color="grey", lw=0.7, linestyle=":")
    ax.axvline(s["p95"], color="grey", lw=0.7, linestyle=":")

    ax.set_xlim(-1000, 1000)
    ax.set_ylim(0, ymax * 1.15)
    ax.set_xlabel(r"$E_\nu^{\rm reco} - E_\nu^{\rm true}$ [MeV]")
    ax.set_ylabel(r"d$\sigma$/d$E$  [arb. fScaleFactor units / MeV]")
    ax.legend(loc="upper right", framealpha=0.9)

    # Filename printed at the top of the page (over the figure title).
    fig.suptitle(title_top + "\n" + os.path.basename(resolved),
                 family="monospace")

    # Stats text block. NOTE: matplotlib renders strings with text.usetex=True
    # by default in this style, so any literal `%` must be escaped as `\%`
    # (otherwise LaTeX treats it as a comment and eats the rest of the line).
    txt = (f"  N events = {len(x)}\n"
           f"  90\\% interval (5-95): [{s['p5']:.1f}, {s['p95']:.1f}] MeV\n"
           f"  90\\% interval width:  {s['p95'] - s['p5']:.1f} MeV")
    ax.text(0.02, 0.97, txt, transform=ax.transAxes, ha="left", va="top",
            fontsize=8, family="monospace",
            bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    obs_keys = ("hk_qe", "dune_epi", "dune_tpi")
    print(f"writing {OUT_PDF}")
    n_pages = 0
    with PdfPages(OUT_PDF) as pdf:
        for cat_label, variants_fn in CATEGORY_VARIANTS:
            for flav in ("numu", "numubar"):
                for var_label, color, path_fn in variants_fn(flav):
                    for obs_key in obs_keys:
                        path = path_fn(obs_key)
                        plot_one_page(pdf,
                                      cat_label=cat_label,
                                      flav=flav,
                                      obs_key=obs_key,
                                      var_label=var_label,
                                      color=color,
                                      source_path=path)
                        n_pages += 1
        d = pdf.infodict()
        d["Title"] = "BW plot validation: per-variant bias distributions"
        d["Author"] = "fsi_iop_plots / validation_distributions"
    print(f"done: {n_pages} pages -> {OUT_PDF}")


if __name__ == "__main__":
    main()
