"""Dump per-distribution numbers for all 6 figures and per-group variation metrics.

Reads the pickle cache that fsi_iop_plots.py wrote, so this runs in seconds.
"""
import os
import pickle
import numpy as np

CACHE_DIR = "/eos/home-l/lamuntea/FSI_IOP_paper/run_genie_bw/_cache"
PI_UP, PI_DN = 1.4, 0.6
TUNES = ["10a", "10b", "10c", "10d"]


def weighted_percentile(x, w, q):
    keep = (w > 0) & np.isfinite(x)
    x, w = x[keep], w[keep]
    order = np.argsort(x)
    xs, ws = x[order], w[order]
    cw = np.cumsum(ws)
    p = (cw - 0.5 * ws) / cw[-1]
    return np.interp(np.asarray(q) / 100.0, p, xs)


def stats(x, w):
    return dict(
        mean=float(np.average(x, weights=w)),
        median=float(weighted_percentile(x, w, 50)),
        p16=float(weighted_percentile(x, w, 16)),
        p84=float(weighted_percentile(x, w, 84)),
        p5=float(weighted_percentile(x, w, 5)),
        p95=float(weighted_percentile(x, w, 95)),
    )


def load(kind, basename):
    return pickle.load(open(f"{CACHE_DIR}/{kind}_{basename}_all.pkl", "rb"))


def hk(path_basename):
    return load("hk", path_basename)  # (x, w, mode)


def dune(path_basename):
    return load("dune", path_basename)  # (x_tpi, x_epi, w, mode)


def pi_reweight(w, mode, factor):
    return w * np.where(np.abs(mode) > 2, factor, 1.0)


def fmt_row(label, s, width=44):
    return (f"{label:<{width}}  "
            f"{s['mean']:>+8.1f} {s['median']:>+8.1f} "
            f"{s['p16']:>+8.1f} {s['p84']:>+8.1f} "
            f"{s['p5']:>+8.1f} {s['p95']:>+8.1f} "
            f"{s['p84']-s['p16']:>7.1f} {s['p95']-s['p5']:>7.1f}")


HEADER = (f"{'row':<44}  "
          f"{'mean':>8} {'median':>8} {'p16':>8} {'p84':>8} "
          f"{'p5':>8} {'p95':>8} {'1σ width':>7} {'90% w':>7}")
SEP = "-" * len(HEADER)


def block(title):
    print()
    print("=" * len(HEADER))
    print(title)
    print("=" * len(HEADER))
    print(HEADER)
    print(SEP)


def variation_block(label, mvals, key="median"):
    """Print min/max/range/std across a list of stats dicts."""
    arr = np.array([m[key] for m in mvals])
    print(f"  {label:<24}  min={arr.min():+8.1f}  max={arr.max():+8.1f}  "
          f"range={arr.max()-arr.min():>6.1f}  std={arr.std(ddof=0):>6.1f}")


# observable retrieval helpers
def obs_arrays(spec):
    """spec = ('hk', basename) or ('dune', basename, 'tpi'/'epi'). Returns (x, w, mode)."""
    if spec[0] == "hk":
        x, w, mode = hk(spec[1])
        return x, w, mode
    _, basename, kind = spec
    x_tpi, x_epi, w, mode = dune(basename)
    return (x_tpi if kind == "tpi" else x_epi), w, mode


# ---------------------------------------------------------------------------
# Build figure->rows config
# ---------------------------------------------------------------------------
def hk_specs(flavour, fsi=True):
    pre = "HK"
    suf = "_FSI.flat.root" if fsi else "_noFSI.flat.root"
    if flavour == "numu":
        return ("hk", f"{pre}_numu{suf}")
    return ("hk", f"{pre}_numubar{suf}")

def dune_specs(flavour, fsi=True, kind="tpi"):
    suf = "_FSI.flat.root" if fsi else "_noFSI.flat.root"
    if flavour == "numu":
        return ("dune", f"DUNE_numu{suf}", kind)
    return ("dune", f"DUNE_numub{suf}", kind)

def hk_genie(flavour, tune):
    if flavour == "numu":
        return ("hk", f"T2KSK_unosc_FHC_numu_H2O_GENIEv3_G18_{tune}_00_000_1M_0000_NUISFLAT.root")
    return ("hk", f"T2KSK_unosc_RHC_numubar_H2O_GENIEv3_G18_{tune}_00_000_1M_0000_NUISFLAT.root")

def dune_genie(flavour, tune, kind):
    if flavour == "numu":
        return ("dune", f"DUNEFD_unosc_FHC_numu_Ar40_GENIEv3_G18_{tune}_00_000_1M_0000_NUISFLAT.root", kind)
    return ("dune", f"DUNEFD_unosc_RHC_numubar_Ar40_GENIEv3_G18_{tune}_00_000_1M_0000_NUISFLAT.root", kind)


OBS_TITLE = {
    "hk_qe":   "HK  E_nu^QE",
    "dune_avail": "DUNE  E_nu^avail  (charged-pi KE)",
    "dune_had":   "DUNE  E_nu^had    (charged-pi full E)",
}

# All dune labels in the script: dune_epi == E_avail (with_pi_corr=False), dune_tpi == E_had (with_pi_corr=True)
OBS_DUNE_KIND = {"dune_avail": "epi", "dune_had": "tpi"}


# ---------------------------------------------------------------------------
# pi-abs (3 obs x 3 variants per flavour)
# ---------------------------------------------------------------------------
for flavour in ("numu", "numubar"):
    block(f"PI-ABS  {flavour}  (NuWro SF, FSI files; reweight |Mode|>2 by ×factor)")
    for obs_label, obs_key in [
        ("hk_qe",      "hk_qe"),
        ("dune_avail", "dune_avail"),
        ("dune_had",   "dune_had"),
    ]:
        if obs_key == "hk_qe":
            x, w, mode = obs_arrays(hk_specs(flavour, fsi=True))
        else:
            kind = OBS_DUNE_KIND[obs_key]
            x, w, mode = obs_arrays(dune_specs(flavour, fsi=True, kind=kind))
        s_nom = stats(x, w)
        s_up  = stats(x, pi_reweight(w, mode, PI_UP))
        s_dn  = stats(x, pi_reweight(w, mode, PI_DN))
        title = OBS_TITLE[obs_key]
        print(fmt_row(f"{title}   nominal", s_nom))
        print(fmt_row(f"{title}   pi_abs +40%", s_up))
        print(fmt_row(f"{title}   pi_abs -40%", s_dn))
        print(f"  Δ vs nominal:   "
              f"+40%: med {s_up['median']-s_nom['median']:+.1f}  mean {s_up['mean']-s_nom['mean']:+.1f}   |   "
              f"-40%: med {s_dn['median']-s_nom['median']:+.1f}  mean {s_dn['mean']-s_nom['mean']:+.1f}")
        print(f"  spread (+40% - -40%):  med {s_up['median']-s_dn['median']:+.1f}  mean {s_up['mean']-s_dn['mean']:+.1f}")
        print(SEP)

# ---------------------------------------------------------------------------
# FSI vs noFSI (3 obs x 2 variants per flavour)
# ---------------------------------------------------------------------------
for flavour in ("numu", "numubar"):
    block(f"FSI vs noFSI  {flavour}  (NuWro SF)")
    for obs_label, obs_key in [
        ("hk_qe",      "hk_qe"),
        ("dune_avail", "dune_avail"),
        ("dune_had",   "dune_had"),
    ]:
        if obs_key == "hk_qe":
            xf, wf, _ = obs_arrays(hk_specs(flavour, fsi=True))
            xn, wn, _ = obs_arrays(hk_specs(flavour, fsi=False))
        else:
            kind = OBS_DUNE_KIND[obs_key]
            xf, wf, _ = obs_arrays(dune_specs(flavour, fsi=True, kind=kind))
            xn, wn, _ = obs_arrays(dune_specs(flavour, fsi=False, kind=kind))
        sf = stats(xf, wf)
        sn = stats(xn, wn)
        title = OBS_TITLE[obs_key]
        print(fmt_row(f"{title}   FSI",    sf))
        print(fmt_row(f"{title}   no FSI", sn))
        print(f"  Δ (FSI - noFSI):  med {sf['median']-sn['median']:+.1f}  "
              f"mean {sf['mean']-sn['mean']:+.1f}  "
              f"1σw {(sf['p84']-sf['p16'])-(sn['p84']-sn['p16']):+.1f}  "
              f"90%w {(sf['p95']-sf['p5'])-(sn['p95']-sn['p5']):+.1f}")
        print(SEP)

# ---------------------------------------------------------------------------
# Cascade comparison (3 obs x 4 cascades per flavour)
# ---------------------------------------------------------------------------
for flavour in ("numu", "numubar"):
    block(f"CASCADE  {flavour}  (GENIE 10a/b/c/d, all FSI-on)")
    for obs_label, obs_key in [
        ("hk_qe",      "hk_qe"),
        ("dune_avail", "dune_avail"),
        ("dune_had",   "dune_had"),
    ]:
        per_tune = {}
        for tune in TUNES:
            if obs_key == "hk_qe":
                x, w, _ = obs_arrays(hk_genie(flavour, tune))
            else:
                kind = OBS_DUNE_KIND[obs_key]
                x, w, _ = obs_arrays(dune_genie(flavour, tune, kind))
            per_tune[tune] = stats(x, w)
        title = OBS_TITLE[obs_key]
        for tune in TUNES:
            print(fmt_row(f"{title}   GENIE_{tune}", per_tune[tune]))
        # variation metrics
        for key, label in [("median", "median"), ("mean", "mean")]:
            arr = np.array([per_tune[t][key] for t in TUNES])
            print(f"  {label:>6} across cascades:  min {arr.min():+8.1f}  "
                  f"max {arr.max():+8.1f}  range {arr.max()-arr.min():>6.1f}  "
                  f"std {arr.std(ddof=0):>6.1f}")
        print(SEP)
