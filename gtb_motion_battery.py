# -*- coding: utf-8 -*-
"""
Pipeline: Motion battery for the gene-to-brain state effect (the make-or-break test)
Author: Drake H. Harbert (D.H.H.) | ORCID 0009-0007-7740-3616 | 2026-06-30
Description:
    Determines whether the reversible state strengthening of the gene-to-FC
    correspondence can be explained by head motion. (1) Per-subject mean framewise
    displacement (FD) in all THREE states (recovery FD was previously missing), and
    whether the FD trajectory reverses like the correspondence trajectory. (2) Whether
    each subject's change in correspondence (unconscious - awake) is explained by their
    change in FD. (3) The state effect adjusted for FD (regress per-subject delta
    correspondence on delta FD; the FD-adjusted effect is the intercept).
    Condition->run: awake=task-rest_run-1, unconscious=task-imagery_run-2,
    recovery=task-rest_run-2.
Dependencies: numpy, pandas, scipy
Output: results/motion_battery/
"""
import os, sys, glob, re, json
import numpy as np, pandas as pd
from scipy.stats import wilcoxon, spearmanr
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
XCP = r"G:/My Drive/inner_architecture_research/brain_connectivity_wj/data/raw/ds006623/derivatives/xcp_d_without_GSR_bandpass_output"
PSUB = r"G:/My Drive/inner_architecture_research/gene_to_brain_wj/results/per_subject/per_subject_mantel.csv"
OUT = r"G:/My Drive/inner_architecture_research/gene_to_brain_wj/results/motion_battery"; os.makedirs(OUT, exist_ok=True)
RUNMAP = {"awake": "task-rest_run-1", "unconscious": "task-imagery_run-2", "recovery": "task-rest_run-2"}

def fd_mean(s, cond):
    fp = os.path.join(XCP, s, "func", f"{s}_{RUNMAP[cond]}_motion.tsv")
    if not os.path.exists(fp): return np.nan
    df = pd.read_csv(fp, sep="\t")
    col = next((c for c in df.columns if "framewise" in c.lower() or c.lower() == "fd"), None)
    if col is None: return np.nan
    return float(pd.to_numeric(df[col], errors="coerce").dropna().mean())

def ols_intercept_slope(x, y):
    X = np.column_stack([np.ones(len(x)), x]); beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta; resid = y - yhat; n = len(x)
    s2 = np.sum(resid**2) / (n - 2); xtx_inv = np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(s2 * xtx_inv))
    from scipy.stats import t
    tval = beta / se; p = 2 * (1 - t.cdf(np.abs(tval), n - 2))
    return {"intercept": float(beta[0]), "intercept_p": float(p[0]),
            "slope_FD": float(beta[1]), "slope_p": float(p[1]),
            "resid_mean_delta": float(np.mean(y - x * beta[1]))}  # delta with FD effect removed

def main():
    subs = sorted({re.match(r"(sub-\d+)_", os.path.basename(p)).group(1)
                   for p in glob.glob(os.path.join(XCP, "sub-*", "func", "*_motion.tsv"))})
    rows = []
    for s in subs:
        r = {"subject": s}
        for c in RUNMAP: r[f"fd_{c}"] = fd_mean(s, c)
        rows.append(r)
    fd = pd.DataFrame(rows).dropna(subset=["fd_awake", "fd_unconscious", "fd_recovery"])
    fd.to_csv(os.path.join(OUT, "per_subject_fd_3state.csv"), index=False)
    res = {"n_with_all_fd": int(len(fd))}

    # (1) FD trajectory across 3 states
    a, u, rc = fd["fd_awake"].values, fd["fd_unconscious"].values, fd["fd_recovery"].values
    res["FD_means"] = {"awake": round(float(a.mean()), 4), "unconscious": round(float(u.mean()), 4), "recovery": round(float(rc.mean()), 4)}
    res["FD_trajectory"] = {
        "unc_minus_awake_p": round(float(wilcoxon(u, a)[1]), 5),
        "unc_minus_recovery_p": round(float(wilcoxon(u, rc)[1]), 5),
        "recovery_minus_awake_p": round(float(wilcoxon(rc, a)[1]), 5),
        "note": "If FD reverses (unc high, recovery returns to awake) like correspondence, motion could mimic the effect."}

    # merge with correspondence
    ps = pd.read_csv(PSUB)
    m = ps.merge(fd, on="subject", how="inner")
    res["n_merged"] = int(len(m))
    dcorr = (m["mantel_r_unconscious"] - m["mantel_r_awake"]).values
    dfd = (m["fd_unconscious"] - m["fd_awake"]).values

    # (2) does the correspondence gain track the motion gain?
    rho, p = spearmanr(dfd, dcorr)
    res["delta_corr_vs_delta_FD"] = {"spearman": round(float(rho), 3), "p": round(float(p), 4),
        "interpretation": "near-zero/ns => motion change does NOT explain correspondence change"}

    # (3) FD-adjusted state effect
    res["FD_adjusted_effect"] = ols_intercept_slope(dfd, dcorr)
    res["FD_adjusted_effect"]["interpretation"] = ("intercept = expected delta-correspondence at zero "
        "delta-FD (the motion-adjusted state effect); positive + significant => survives FD adjustment")
    res["raw_mean_delta_corr"] = round(float(np.mean(dcorr)), 4)
    json.dump(res, open(os.path.join(OUT, "motion_battery.json"), "w"), indent=2)
    print(json.dumps(res, indent=2))

if __name__ == "__main__":
    main()
