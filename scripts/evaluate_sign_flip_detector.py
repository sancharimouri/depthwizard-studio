"""Threshold search + confusion-matrix validation for the sign-flip
detector built in detect_sign_flip.py. Ground truth (DAv2-vs-ICESat2 sign)
is used ONLY here, for validation -- never as a detector input.
"""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SIGNALS_CSV = ROOT / "data/sentinel2_benchmark/sign_flip_detector_signals.csv"


def confusion(df, thresh_a, informative_tree_pct, thresh_b):
    informative = df["tree_pct"] >= informative_tree_pct
    flag_a = df["dav2_vs_dem_pearson"] < thresh_a
    flag_b = informative & (df["dav2_vs_dinov3_pearson"] < thresh_b)
    flagged = flag_a | flag_b

    tp = int((flagged & df["true_inverted"]).sum())
    fp = int((flagged & ~df["true_inverted"]).sum())
    fn = int((~flagged & df["true_inverted"]).sum())
    tn = int((~flagged & ~df["true_inverted"]).sum())
    return tp, fp, fn, tn, flagged, flag_a, flag_b


def main():
    df = pd.read_csv(SIGNALS_CSV)

    print("=== Signal (a) alone: threshold sweep on dav2_vs_dem_pearson ===")
    for t in [0.05, 0.02, 0.0, -0.02, -0.05, -0.1, -0.2]:
        tp, fp, fn, tn, *_ = confusion(df, t, informative_tree_pct=101, thresh_b=-999)  # signal b disabled
        print(f"  thresh_a={t:+.2f}  TP={tp} FP={fp} FN={fn} TN={tn}  acc={(tp+tn)/32:.3f}")

    print("\n=== Signal (b) alone (informative>=5% tree, all thresh_a effectively off): threshold sweep on dav2_vs_dinov3_pearson ===")
    for t in [-0.05, -0.1, -0.15, -0.2, -0.3, -0.4]:
        tp, fp, fn, tn, *_ = confusion(df, thresh_a=-999, informative_tree_pct=5.0, thresh_b=t)
        print(f"  thresh_b={t:+.2f}  TP={tp} FP={fp} FN={fn} TN={tn}  acc={(tp+tn)/32:.3f}")

    print("\n=== Combined rule sensitivity: thresh_a=0.0 fixed, vary thresh_b (informative>=5% tree) ===")
    for t in [-0.1, -0.15, -0.2, -0.25, -0.3, -0.4, -0.5]:
        tp, fp, fn, tn, flagged, flag_a, flag_b = confusion(df, thresh_a=0.0, informative_tree_pct=5.0, thresh_b=t)
        extra_fp_from_b = df[flag_b & ~flag_a & ~df["true_inverted"]]["tile_id"].tolist()
        extra_tp_from_b = df[flag_b & ~flag_a & df["true_inverted"]]["tile_id"].tolist()
        print(
            f"  thresh_b={t:+.2f}  TP={tp} FP={fp} FN={fn} TN={tn}  acc={(tp+tn)/32:.3f}  "
            f"b-only-added-FP={extra_fp_from_b}  b-only-added-TP={extra_tp_from_b}"
        )

    print("\n=== FINAL CHOSEN RULE: thresh_a=0.0, informative if tree_pct>=5%, thresh_b=-0.3 ===")
    tp, fp, fn, tn, flagged, flag_a, flag_b = confusion(df, thresh_a=0.0, informative_tree_pct=5.0, thresh_b=-0.3)
    print(f"TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"accuracy={(tp+tn)/32:.4f}  precision={tp/(tp+fp) if (tp+fp) else float('nan'):.4f}  recall={tp/(tp+fn) if (tp+fn) else float('nan'):.4f}")

    df["flagged"] = flagged
    df["flag_a"] = flag_a
    df["flag_b"] = flag_b
    print("\nFull per-tile verdict table:")
    cols = ["tile_id", "category", "dav2_vs_dem_pearson", "dav2_vs_dinov3_pearson", "tree_pct", "flag_a", "flag_b", "flagged", "true_inverted"]
    print(df[cols].sort_values("true_inverted", ascending=False).to_string(index=False))

    mismatches = df[df["flagged"] != df["true_inverted"]]
    print("\nMismatches (false positives / false negatives):")
    print(mismatches[cols].to_string(index=False) if len(mismatches) else "  none")

    df.to_csv(ROOT / "data/sentinel2_benchmark/sign_flip_detector_verdicts.csv", index=False)


if __name__ == "__main__":
    main()
