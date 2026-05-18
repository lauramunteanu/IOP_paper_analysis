"""validation_distributions.py — per-variant input + bias distributions
that feed the BW summary plots in fsi_iop_plots.py.

Two multi-page PDFs, one per bias mode:
  bw_validation_distributions_abs.pdf  (bias in MeV)
  bw_validation_distributions_rel.pdf  (bias dimensionless)

Each page has three panels (one per category, flavour, observable, variant):
  - top-left:  Enu_true distribution         [GeV]
  - top-right: Enu_reco distribution         [GeV]  (Enu_QE / Enu_had / Enu_avail)
  - bottom:    bias histogram                [MeV or dimensionless], stacked by |Mode|

This lets you eyeball that the bias-calculation inputs (Enu_true and Enu_reco)
look sensible per (variant, sample) before trusting the BW summary plots.

Output: <OUT_DIR>/bw_validation_distributions_{abs,rel}.pdf
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from fsi_iop_plots import (
    get_obs_with_mode, resolve_path,
    stats,
    OBS_LABELS, hist_params_for,
    OUT_DIR,
    CATEGORY_VARIANTS,
    NU_LABELS,
)

# Direct ROOT-array helpers for the Enu_true / Enu_reco input panels.
# These are mode-agnostic (always GeV) so abs / rel modes share them.
_SCRIPTS_DIR = os.path.dirname(_HERE)
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from FlatTreeMod import load_arrays, enu_true_arr, enu_reco_arr  # noqa: E402


def _bias_xlabel(mode):
    if mode == "abs":
        return r"$E_\nu^{\rm reco} - E_\nu^{\rm true}$ [MeV]"
    return r"$(E_\nu^{\rm reco} - E_\nu^{\rm true}) / E_\nu^{\rm true}$"


def _bias_ylabel(mode):
    if mode == "abs":
        return r"d$\sigma$/d$E$  [arb. fScaleFactor units / MeV]"
    return r"d$\sigma$/d$\delta$  [arb. fScaleFactor units]"


def _fmt(v, mode):
    return f"{v:+.1f}" if mode == "abs" else f"{v:+.4f}"


def _unit_suffix(mode):
    return "MeV" if mode == "abs" else ""


# Enu_reco label per observable (GeV axis).
_ENU_RECO_LABEL = {
    "hk_qe":   r"$E_\nu^{\rm QE}$ [GeV]",
    "dune_epi": r"$E_\nu^{\rm avail}$ [GeV]",
    "dune_tpi": r"$E_\nu^{\rm had}$ [GeV]",
}

# Observable → which bias_arr observable name (for compute selection match).
_OBS_NAME = {"hk_qe": "qe", "dune_epi": "avail", "dune_tpi": "had"}

# Per-experiment Enu axis range (GeV) for the input panels.
_ENU_XLIM = {
    "hk_qe":   (0.0, 2.0),
    "dune_epi": (0.0, 6.0),
    "dune_tpi": (0.0, 6.0),
}
_ENU_NBINS = 60


def _draw_enu_hist(ax, x_GeV, w, hist_range, label, color):
    if x_GeV is None or len(x_GeV) == 0:
        ax.text(0.5, 0.5, "no events", ha="center", va="center",
                transform=ax.transAxes, color="red")
        return
    bin_width = (hist_range[1] - hist_range[0]) / _ENU_NBINS
    ax.hist(x_GeV, bins=_ENU_NBINS, range=hist_range,
            weights=w / bin_width, histtype="stepfilled", alpha=0.7,
            color=color, edgecolor="black", linewidth=0.4)
    ax.set_xlim(*hist_range)
    ax.set_xlabel(label)
    ax.set_ylabel("d$\\sigma$/dE  [arb. / GeV]")


def plot_one_page(pdf, *, cat_label, flav, obs_key, var_label, color, source_path, mode):
    """Three-panel validation page: Enu_true, Enu_reco, bias stacked by |Mode|."""
    nbins, hrange = hist_params_for(obs_key, mode)
    # Validation override: zoom the DUNE rel-bias panels into (-1, 0.1) so
    # the body of the distribution fills the panel; tails outside the range
    # land in the over/underflow bins of the histogram and are not shown.
    if mode == "rel" and obs_key in ("dune_epi", "dune_tpi"):
        hrange = (-1.0, 0.1)

    fig = plt.figure(figsize=(8.0, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.6], hspace=0.42, wspace=0.30)
    ax_etrue = fig.add_subplot(gs[0, 0])
    ax_ereco = fig.add_subplot(gs[0, 1])
    ax_bias  = fig.add_subplot(gs[1, :])

    resolved = resolve_path(source_path)
    title_top = (f"{cat_label}  |  {NU_LABELS[flav]}  |  {OBS_LABELS[obs_key]}  |  "
                 f"variant: {var_label}  |  mode: {mode}")

    if resolved is None:
        for ax in (ax_etrue, ax_ereco, ax_bias):
            ax.text(0.5, 0.5, "file missing", ha="center", va="center",
                    transform=ax.transAxes, color="red")
            ax.set_xticks([]); ax.set_yticks([])
        fig.suptitle(title_top + "\n(missing)")
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    # ---- Bias panel (existing logic; stacked by |Mode|) ----
    res = get_obs_with_mode(resolved, obs_key, mode)
    if res is None or len(res[0]) == 0:
        for ax in (ax_etrue, ax_ereco, ax_bias):
            ax.text(0.5, 0.5, "no events", ha="center", va="center",
                    transform=ax.transAxes, color="red")
        fig.suptitle(title_top)
        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        return

    x, w, modes_arr = res
    x = np.asarray(x); w = np.asarray(w); modes_arr = np.asarray(modes_arr)
    s = stats(x, w)

    bin_width = (hrange[1] - hrange[0]) / nbins
    abs_mode = np.abs(modes_arr)
    MODE_GROUPS = [
        (1,  "CCQE",            "#0077BB"),
        (2,  "CC2p2h",          "#009988"),
        (11, r"CC RES $p+\pi^+$","#EE7733"),
        (12, r"CC RES $p+\pi^0$","#EE3377"),
        (13, r"CC RES $n+\pi^+$","#33BBEE"),
        (16, r"CC coherent $\pi$","#882255"),
        (21, r"CC multi-$\pi$",  "#AA4499"),
        (26, "CC DIS",           "#CC3311"),
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

    counts_stack = ax_bias.hist(
        data_list, bins=nbins, range=hrange, weights=weight_list,
        histtype="stepfilled", stacked=True, alpha=0.80,
        color=colors_list, label=labels_list,
        edgecolor="white", linewidth=0.2,
    )
    if isinstance(counts_stack[0], list):
        top = counts_stack[0][-1]
    else:
        top = counts_stack[0]
    ymax = top.max() if hasattr(top, "max") and top.max() > 0 else 1.0

    units = _unit_suffix(mode)
    ax_bias.axvspan(s["p16"], s["p84"], alpha=0.20, color="orange",
                    label=fr"1$\sigma$ (16-84): [{_fmt(s['p16'], mode)}, {_fmt(s['p84'], mode)}] {units}")
    ax_bias.axvline(s["median"], color="black", lw=1.6,
                    label=f"median = {_fmt(s['median'], mode)} {units}")
    ax_bias.axvline(s["mean"], color="red", lw=1.6, linestyle="--",
                    label=f"mean = {_fmt(s['mean'], mode)} {units}")
    ax_bias.axvline(s["p5"],  color="grey", lw=0.7, linestyle=":")
    ax_bias.axvline(s["p95"], color="grey", lw=0.7, linestyle=":")
    ax_bias.set_xlim(hrange[0], hrange[1])
    ax_bias.set_ylim(0, ymax * 1.15)
    ax_bias.set_xlabel(_bias_xlabel(mode))
    ax_bias.set_ylabel(_bias_ylabel(mode))
    ax_bias.legend(loc="upper right", framealpha=0.9, fontsize=7)

    # ---- Enu_true / Enu_reco panels (mode-agnostic; always GeV) ----
    try:
        arr = load_arrays(resolved, max_events=None)
        obs_name = _OBS_NAME[obs_key]
        etrue_GeV = enu_true_arr(arr, obs_name, vertex=False)
        ereco_GeV = enu_reco_arr(arr, obs_name, vertex=False)
        # weights are already selection-matched (same as bias)
        _draw_enu_hist(ax_etrue, etrue_GeV, w, _ENU_XLIM[obs_key],
                       r"$E_\nu^{\rm true}$ [GeV]", color="#377EB8")
        _draw_enu_hist(ax_ereco, ereco_GeV, w, _ENU_XLIM[obs_key],
                       _ENU_RECO_LABEL[obs_key], color="#E41A1C")
    except Exception as e:  # don't let a single-file glitch kill the whole PDF
        for ax in (ax_etrue, ax_ereco):
            ax.text(0.5, 0.5, f"Enu panel error:\n{e}",
                    ha="center", va="center", transform=ax.transAxes,
                    color="red", fontsize=8)

    fig.suptitle(title_top + "\n" + os.path.basename(resolved),
                 family="monospace")

    txt = (f"  N events = {len(x)}\n"
           f"  90\\% interval (5-95): [{_fmt(s['p5'], mode)}, {_fmt(s['p95'], mode)}] {units}\n"
           f"  90\\% interval width:  {_fmt(s['p95'] - s['p5'], mode)} {units}")
    ax_bias.text(0.02, 0.97, txt, transform=ax_bias.transAxes, ha="left", va="top",
                 fontsize=8, family="monospace",
                 bbox=dict(facecolor="white", edgecolor="grey", alpha=0.85))

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main():
    obs_keys = ("hk_qe", "dune_epi", "dune_tpi")
    os.makedirs(OUT_DIR, exist_ok=True)
    for mode in ("abs", "rel"):
        out_pdf = f"{OUT_DIR}/bw_validation_distributions_{mode}.pdf"
        print(f"writing {out_pdf}")
        n_pages = 0
        with PdfPages(out_pdf) as pdf:
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
                                          source_path=path,
                                          mode=mode)
                            n_pages += 1
            d = pdf.infodict()
            d["Title"] = f"BW plot validation: per-variant input + bias distributions ({mode})"
            d["Author"] = "fsi_iop_plots / validation_distributions"
        print(f"done [{mode}]: {n_pages} pages -> {out_pdf}")


if __name__ == "__main__":
    main()
