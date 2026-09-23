"""Step 2 of the DAv2 calibration-hardening pass: add a second,
confidence-based tier on top of the existing sign-flip REJECTED/accepted
gate, using held_out_dem_pearson from calibrate_accepted_tiles.py's
output. Same "find a real gap in the data" method already used to justify
the sign-flip detector's a=0.0 threshold (sign-flip-detector.md).

Updates data/sentinel2_benchmark/sign_flip_detector_verdicts.csv in place
with a new `tier` column: REJECTED / LOW-CONFIDENCE / CONFIDENT.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
VERDICTS_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv"
CAL_CSV = ROOT / "data/sentinel2_benchmark/calibration_results.csv"


def main():
    verdicts = pd.read_csv(VERDICTS_CSV)
    cal = pd.read_csv(CAL_CSV).set_index("tile_id")

    accepted = cal[cal["status"] == "calibrated"].sort_values("held_out_dem_pearson")
    vals = accepted["held_out_dem_pearson"].values
    gaps = [(vals[i + 1] - vals[i], vals[i], vals[i + 1]) for i in range(len(vals) - 1)]
    gaps.sort(reverse=True)
    biggest_gap, lower, upper = gaps[0]
    threshold = (lower + upper) / 2
    print(f"Largest gap among {len(vals)} accepted tiles' held_out_dem_pearson: {biggest_gap:.4f}, between {lower:.4f} and {upper:.4f}")
    print(f"Chosen threshold (midpoint of gap): {threshold:.4f}")

    low_conf_tiles = accepted[accepted["held_out_dem_pearson"] <= lower].index.tolist()
    print(f"LOW-CONFIDENCE tiles (held_out_dem_pearson <= {lower:.4f}): {low_conf_tiles}")

    def tier(tile_id):
        row = cal.loc[tile_id]
        if row["status"] != "calibrated":
            return "REJECTED"
        return "LOW-CONFIDENCE" if row["held_out_dem_pearson"] <= lower else "CONFIDENT"

    verdicts["tier"] = verdicts["tile_id"].map(tier)
    verdicts["held_out_dem_pearson"] = verdicts["tile_id"].map(
        lambda t: cal.loc[t, "held_out_dem_pearson"] if cal.loc[t, "status"] == "calibrated" else float("nan")
    )

    print("\nTier counts:")
    print(verdicts["tier"].value_counts())
    print("\nFull tier table:")
    print(verdicts[["tile_id", "category", "flagged", "tier", "held_out_dem_pearson"]].sort_values(["tier", "held_out_dem_pearson"]).to_string(index=False))

    verdicts.to_csv(VERDICTS_CSV, index=False)
    print(f"\nUpdated {VERDICTS_CSV} with the new `tier` column.")


if __name__ == "__main__":
    main()
