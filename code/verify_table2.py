# Regenerates every number in Table II of the paper from the saved CSVs.
#
# WHICH FILE BACKS TABLE II
#   Table II reports the configuration WITH class-wise aggregation (alpha).
#   That is  results/fedobserve_ridegrouped.csv  (rows 1, 2, 5, 6)
#        and results/baselines_ridegrouped.csv   (rows 3, 4).
#
#   results/main_table_ridegrouped.csv is NOT Table II. It is the alpha-DROPPED
#   ablation arm, which reaches 0.0949 on train rather than 0.1289. It is kept
#   because the factorial in Sec. VI refers to it, and it is the reason the
#   headline number cannot be found by filename alone.
#
# Runs in under a second; it only reads CSVs, it does not retrain anything.
import os
import pandas as pd
from scipy import stats as sps

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(os.path.dirname(HERE), "results")

G = pd.read_csv(os.path.join(RES, "fedobserve_ridegrouped.csv"))
B = pd.read_csv(os.path.join(RES, "baselines_ridegrouped.csv"))

# paper row label -> (dataframe, method string in that file)
ROWS = [
    ("FedAvg, no mask",       G, "FedAvg"),
    ("FedAvg",                G, "FedAvg + mask"),
    ("FedProx",               B, "FedProx + mask"),
    ("FedRS",                 B, "FedRS + mask"),
    ("Rule, training only",   G, "FedObserve"),
    ("Rule + inference mask", G, "FedObserve + mask"),
]

PAPER = {  # what the printed table says: (train F1, macro F1)
    "FedAvg, no mask":       (0.0160, 0.5999),
    "FedAvg":                (0.0259, 0.6061),
    "FedProx":               (0.0254, 0.6035),
    "FedRS":                 (0.1089, 0.6173),
    "Rule, training only":   (0.1112, 0.6107),
    "Rule + inference mask": (0.1289, 0.6187),
}

print("=" * 76)
print("TABLE II  -  ride-grouped, ten seeds, paired")
print("=" * 76)
ok_all = True
for label, df, meth in ROWS:
    d = df[df.method == meth]
    assert len(d) == 10, "expected 10 seeds for %s, found %d" % (meth, len(d))
    tr, mac = d.f1_train.mean(), d.macro_f1.mean()
    ptr, pmac = PAPER[label]
    ok = (round(tr, 4) == ptr) and (round(mac, 4) == pmac)
    ok_all &= ok
    print("  %-22s train %.4f (paper %.4f)   macro %.4f (paper %.4f)   %s"
          % (label, tr, ptr, mac, pmac, "OK" if ok else "MISMATCH"))

print()
print("=" * 76)
print("PAIRED TESTS  -  Wilcoxon signed-rank, one-sided, ten paired seeds")
print("=" * 76)

ours = G[G.method == "FedObserve + mask"].sort_values("seed")


def paired(name, other):
    for metric, shown in [("f1_train", "train F1"), ("macro_f1", "macro F1")]:
        x = other[metric].to_numpy()
        y = ours[metric].to_numpy()
        _, p = sps.wilcoxon(y, x, alternative="greater")
        print("  vs %-16s %-9s  %.4f vs %.4f   wins %2d/10   p = %.4f"
              % (name, shown, y.mean(), x.mean(), int((y > x).sum()), p))


paired("FedAvg + mask",  G[G.method == "FedAvg + mask"].sort_values("seed"))
paired("FedProx + mask", B[B.method == "FedProx + mask"].sort_values("seed"))
paired("FedRS + mask",   B[B.method == "FedRS + mask"].sort_values("seed"))

print()
print("Paper text to check against:")
print("  +0.0127 macro F1 (9/10 seeds, p = 0.0322)")
print("  +0.1030 on train (9/10 seeds, p = 0.0020)")
print("  FedRS:  macro 0.6173 vs 0.6187 (5/10, p = 0.4229);")
print("          train 0.1089 vs 0.1289 (6/10, p = 0.1328)   <- reported as a tie")
print()
print("ALL TABLE II VALUES REPRODUCED" if ok_all else "MISMATCH FOUND - see above")
