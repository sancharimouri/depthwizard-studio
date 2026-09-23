"""Step 1 of the DAv2 calibration-hardening pass: does a surface-offset
proxy (built_up% + tree_pct -- both raise the sensed surface above bare
ground, which is exactly what a DSM-vs-DTM mismatch would do) explain
icesat2_rmse_pct_of_range better than land-cover category alone?

Uses only existing data: calibration_results_with_range.csv (25 accepted
tiles) and content_audit_corrected_32.csv (the already-corrected 32-tile
WorldCover audit). No new dependency: R^2 for a single continuous
predictor via simple OLS equals Pearson r^2 (a standard identity); R^2
for a categorical predictor is computed directly via one-way ANOVA's
between/total sum-of-squares decomposition.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
CAL_CSV = ROOT / "data/sentinel2_benchmark/calibration_results_with_range.csv"
AUDIT_CSV = ROOT / "data/sentinel2_benchmark/content_audit_corrected_32.csv"


def r_squared_categorical(df, group_col, value_col):
    grand_mean = df[value_col].mean()
    ss_total = ((df[value_col] - grand_mean) ** 2).sum()
    ss_within = 0.0
    for _, group in df.groupby(group_col):
        ss_within += ((group[value_col] - group[value_col].mean()) ** 2).sum()
    ss_between = ss_total - ss_within
    return ss_between / ss_total


def main():
    cal = pd.read_csv(CAL_CSV)
    audit = pd.read_csv(AUDIT_CSV).set_index("tile_id")

    cal["tree_pct"] = cal["tile_id"].map(audit["tree_pct"])
    cal["builtup_pct"] = cal["tile_id"].map(audit["builtup_pct"])
    cal["offset_proxy"] = cal["tree_pct"] + cal["builtup_pct"]

    print("=== Per-tile proxy values ===")
    print(
        cal[["tile_id", "category", "tree_pct", "builtup_pct", "offset_proxy", "icesat2_rmse_pct_of_range"]]
        .sort_values("offset_proxy")
        .to_string(index=False)
    )

    print("\n=== 1. Simple correlation: offset_proxy vs icesat2_rmse_pct_of_range (n=25) ===")
    r, p = stats.pearsonr(cal["offset_proxy"], cal["icesat2_rmse_pct_of_range"])
    print(f"Pearson r = {r:+.4f}, p = {p:.4f}")
    rs, ps = stats.spearmanr(cal["offset_proxy"], cal["icesat2_rmse_pct_of_range"])
    print(f"Spearman r = {rs:+.4f}, p = {ps:.4f}")

    print("\n=== 2. Explanatory power: category (4-level) vs proxy (continuous) ===")
    r2_cat = r_squared_categorical(cal, "category", "icesat2_rmse_pct_of_range")
    r2_proxy = r ** 2  # single-predictor OLS R^2 == Pearson r^2
    print(f"R^2 (~ category, 4-level): {r2_cat:.4f}")
    print(f"R^2 (~ offset_proxy, 1 continuous predictor): {r2_proxy:.4f}")

    winner = "proxy" if r2_proxy > r2_cat else "category"
    print(f"\n=> {winner} explains more variance ({max(r2_proxy, r2_cat):.4f} vs {min(r2_proxy, r2_cat):.4f})")

    print("\n=== Category means, for reference ===")
    print(cal.groupby("category")["icesat2_rmse_pct_of_range"].agg(["mean", "std", "count"]))

    print("\n=== Confound check: is offset_proxy's correlation just riding on elevation range? ===")
    r_or, p_or = stats.pearsonr(cal["offset_proxy"], cal["dem_elev_range_m"])
    print(f"offset_proxy vs dem_elev_range_m: r={r_or:+.4f}, p={p_or:.4f}")
    r_re, p_re = stats.pearsonr(cal["dem_elev_range_m"], cal["icesat2_rmse_pct_of_range"])
    print(f"dem_elev_range_m vs icesat2_rmse_pct_of_range: r={r_re:+.4f}, p={p_re:.4f}")

    def resid(y, x):
        coef = np.polyfit(x, y, 1)
        return y - np.polyval(coef, x)

    log_range = np.log(cal["dem_elev_range_m"].values)
    resid_proxy = resid(cal["offset_proxy"].values, log_range)
    resid_target = resid(cal["icesat2_rmse_pct_of_range"].values, log_range)
    r_partial, p_partial = stats.pearsonr(resid_proxy, resid_target)
    print(f"PARTIAL correlation (offset_proxy, RMSE%) controlling for log(elev_range): r={r_partial:+.4f}, p={p_partial:.4f}")

    print("\n=== 3. Fallback harder proxy: held-out DEM residual magnitude ===")
    r_mae, p_mae = stats.pearsonr(cal["held_out_dem_mae_m"], cal["icesat2_rmse_pct_of_range"])
    print(f"held_out_dem_mae_m (absolute, m) vs icesat2_rmse_pct_of_range: r={r_mae:+.4f}, p={p_mae:.4f}, R^2={r_mae**2:.4f}")
    r_maepct, p_maepct = stats.pearsonr(cal["dem_rmse_pct_of_range"], cal["icesat2_rmse_pct_of_range"])
    print(f"dem_rmse_pct_of_range (normalized) vs icesat2_rmse_pct_of_range: r={r_maepct:+.4f}, p={p_maepct:.4f}, R^2={r_maepct**2:.4f}")

    cal.to_csv(ROOT / "data/sentinel2_benchmark/step1_offset_proxy.csv", index=False)
    return cal, r, p, r2_cat, r2_proxy


if __name__ == "__main__":
    main()
