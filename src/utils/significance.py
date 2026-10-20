#!/usr/bin/env python3
"""
Comprehensive paired significance report for the per-subject metrics produced
by the pipeline (resolved_config[dataset][method][time] = list of per-subject
JSON files).

For every (time budget) x (metric) x (baseline) it reports:
  n            number of matched subjects
  wins         subjects on which 'proposed' beats the baseline
  med_diff     median of (proposed - baseline) over subjects
  p_wilcox     two-sided paired Wilcoxon signed-rank p-value (exact at this n)
  p_holm       Holm-Bonferroni adjusted p within its family (see below)
  sig          '*' if p_holm < ALPHA

Families for the Holm correction (corrected separately within each budget):
  PRIMARY : the 7 diffusion-relevant metrics x 2 baselines  = 14 tests
  IMAGE   : PSNR/SSIM (FA,GFA)             x 2 baselines     =  8 tests
The two families are corrected independently, so reporting the generic image
metrics does not weaken the correction on the primary claims.

Nothing is hand-computed: every number, including the Holm multiplier and the
exact-test floor, is printed by the code.

Usage:
    python src/utils/significance.py [resolved_config.json] [dataset]
    (defaults: output/artifacts/metrics/cv_metrics_resolved.json, HCP)
"""
import json
import re
import sys
import warnings
from pathlib import Path
import numpy as np
from scipy.stats import wilcoxon, ttest_rel

ALPHA = 0.05

# key -> True if LOWER is better (improvement means proposed < baseline)
LOWER_IS_BETTER = {
    ("NRMSE", "FA"): True,  ("NRMSE", "GFA"): True,
    ("FSIM",  "FA"): False, ("FSIM",  "GFA"): False,
    ("PSNR",  "FA"): False, ("PSNR",  "GFA"): False,
    ("SSIM",  "FA"): False, ("SSIM",  "GFA"): False,
    "ODF_L2_Norm_Error_median": True,
    "GFA_Absolute_Error_median": True,
    "Eigenvector_Angular_Error_median_deg": True,
}

PRIMARY = [("NRMSE","FA"), ("NRMSE","GFA"), ("FSIM","FA"), ("FSIM","GFA"),
           "ODF_L2_Norm_Error_median", "GFA_Absolute_Error_median",
           "Eigenvector_Angular_Error_median_deg"]
IMAGE   = [("PSNR","FA"), ("PSNR","GFA"), ("SSIM","FA"), ("SSIM","GFA")]

LABEL = {("NRMSE","FA"):"NRMSE-FA", ("NRMSE","GFA"):"NRMSE-GFA",
         ("FSIM","FA"):"FSIM-FA", ("FSIM","GFA"):"FSIM-GFA",
         ("PSNR","FA"):"PSNR-FA", ("PSNR","GFA"):"PSNR-GFA",
         ("SSIM","FA"):"SSIM-FA", ("SSIM","GFA"):"SSIM-GFA",
         "ODF_L2_Norm_Error_median":"ODF-L2",
         "GFA_Absolute_Error_median":"GFA-abs-err",
         "Eigenvector_Angular_Error_median_deg":"Angular-err"}

BASELINES = ("baseline", "hashenc")   # NODF, NODF-HashEnc
BLABEL = {"baseline": "NODF", "hashenc": "HashEnc"}


# ----------------------------- data loading --------------------------------
def _subject_id(path):
    """Parse a 6-digit HCP-YA subject id from the filename; fall back to stem."""
    m = re.search(r"(\d{6})", Path(path).name)
    return m.group(1) if m else Path(path).stem

def _load_value(path, key):
    d = json.load(open(path))
    return d[key[0]][key[1]] if isinstance(key, tuple) else d[key]

def _paired(proposed_paths, baseline_paths, key):
    prop = {_subject_id(p): _load_value(p, key) for p in proposed_paths}
    base = {_subject_id(p): _load_value(p, key) for p in baseline_paths}
    ids = sorted(set(prop) & set(base))
    if len(ids) != len(prop) or len(ids) != len(base):
        warnings.warn(f"subject-id mismatch for {key}: "
                      f"proposed={len(prop)} baseline={len(base)} matched={len(ids)}")
    a = np.array([prop[i] for i in ids], float)   # proposed
    b = np.array([base[i] for i in ids], float)   # baseline
    return ids, a, b


# ----------------------------- statistics ----------------------------------
def one_test(proposed_paths, baseline_paths, key):
    ids, a, b = _paired(proposed_paths, baseline_paths, key)
    n = len(ids)
    lower_better = LOWER_IS_BETTER[key]
    wins = int(np.sum(a < b)) if lower_better else int(np.sum(a > b))
    diff = a - b
    if n == 0 or np.all(diff == 0):
        return dict(key=key, n=n, wins=wins, med_diff=0.0, p=1.0, p_t=1.0)
    p   = float(wilcoxon(a, b, alternative="two-sided").pvalue)  # exact at small n
    p_t = float(ttest_rel(a, b).pvalue)
    return dict(key=key, n=n, wins=wins, med_diff=float(np.median(diff)),
                p=p, p_t=p_t)

def holm(pvals):
    """Holm-Bonferroni adjusted p-values in original order; returns (adj, m)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * p[idx]          # multiplier = m, m-1, ...
        running = max(running, val)        # enforce monotone non-decreasing
        adj[idx] = min(running, 1.0)
    return adj, m


# ----------------------------- reporting -----------------------------------
def exact_floor(n):
    """Two-sided exact Wilcoxon floor at sample size n (all-same-sign case)."""
    if n == 0:
        return 1.0
    return float(wilcoxon(np.arange(1, n + 1), alternative="two-sided").pvalue)

def family_table(cfg, time, family, fam_name, proposed="proposed"):
    tests, praw = [], []
    for base in BASELINES:
        for key in family:
            r = one_test(cfg[proposed][time], cfg[base][time], key)
            r["baseline"] = base
            tests.append(r); praw.append(r["p"])
    adj, m = holm(praw)
    for r, pa in zip(tests, adj):
        r["p_holm"] = float(pa)

    print(f"\n  Family: {fam_name}  ({m} tests, Holm-corrected within this budget)")
    print(f"  {'baseline':8}{'metric':13}{'n':>3}{'wins':>7}"
          f"{'med_diff':>12}{'p_wilcox':>11}{'p_holm':>10}  sig")
    print("  " + "-" * 74)
    for r in tests:
        sig = "*" if r["p_holm"] < ALPHA else " "
        print(f"  {BLABEL[r['baseline']]:8}{LABEL[r['key']]:13}{r['n']:>3}"
              f"{(str(r['wins'])+'/'+str(r['n'])):>7}{r['med_diff']:>12.4f}"
              f"{r['p']:>11.5f}{r['p_holm']:>10.5f}   {sig}")
    ps = [t["p_holm"] for t in tests]
    n_sig = sum(pp < ALPHA for pp in ps)
    print(f"  -> {n_sig}/{m} significant at alpha={ALPHA} "
          f"(adjusted p range {min(ps):.5f} .. {max(ps):.5f})")
    return tests


def full_report(resolved_config, dataset="HCP",
                times=("5_min", "10_min", "30_min")):
    cfg = resolved_config[dataset]
    n0 = len(cfg["proposed"][times[0]])
    print("=" * 80)
    print(f"Paired Wilcoxon signed-rank report  |  dataset={dataset}  |  n={n0} subjects")
    print(f"Two-sided EXACT floor at n={n0}: p = {exact_floor(n0):.6f}  "
          f"(smallest achievable p; reached at {n0}/{n0} wins)")
    print("med_diff = median(proposed - baseline). p_t (t-test) shown at end, reference only.")
    print("=" * 80)

    for t in times:
        print(f"\n----------------  BUDGET: {t}  ----------------")
        family_table(cfg, t, PRIMARY, "PRIMARY (7 diffusion metrics x 2 baselines)")
        family_table(cfg, t, IMAGE,   "IMAGE (PSNR/SSIM x 2 baselines)")

    print("\n" + "=" * 80)
    print("Reference only: paired t-test min p per budget (NOT reported; assumes normality)")
    print("=" * 80)
    for t in times:
        pts = [one_test(cfg["proposed"][t], cfg[b][t], k)["p_t"]
               for b in BASELINES for k in PRIMARY]
        print(f"  {t:8}: min p_t = {min(pts):.2e}")


def main():
    default_cfg = "output/artifacts/metrics/cv_metrics_resolved.json"
    path = sys.argv[1] if len(sys.argv) > 1 else default_cfg
    dataset = sys.argv[2] if len(sys.argv) > 2 else "HCP"
    full_report(json.load(open(path)), dataset=dataset)


if __name__ == "__main__":
    main()
