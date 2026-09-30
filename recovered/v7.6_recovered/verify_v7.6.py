"""
Compare a rerun of v7.6 (under MRD_PROJ) against the original v7.6 results saved on 2026-08-09.

Two of the saved originals were later re-saved by Excel (";"-separated, decimal point dropped:
30.5 -> 305), so those are compared by content.
"""
import os, glob
import numpy as np, pandas as pd

REC  = os.environ.get("MRD_REC",  "/mnt/e/MRD analysis Apr_2026/30 May 2026_Claude new results/new 18.06.26")
PROJ = os.environ.get("MRD_PROJ", f"{REC}/proj_recovered")
O = f"{REC}/Nandhu_final_v7.6"
N = f"{PROJ}/results_soumya/Nandhu_final_v7.6"
K = ["unique_region", "Cell Id"]
ok = True

def report(name, passed, detail=""):
    global ok
    ok &= bool(passed)
    print(f"[{'PASS' if passed else 'FAIL'}] {name} {detail}")

a = pd.read_csv(f"{O}/csv/cell_assignments.csv", dtype={"Cell Id": str}).set_index(K)
b = pd.read_csv(f"{N}/csv/cell_assignments.csv", dtype={"Cell Id": str}).set_index(K)
report("same cells", a.index.sort_values().equals(b.index.sort_values()), f"({len(a)} vs {len(b)})")
b = b.loc[a.index]
for c in ["v76_label", "v76_coarse", "v76_label_int", "v2_final_label_integer", "machine", "Nuc X", "Nuc Y Inv"]:
    same = int(((a[c] == b[c]) | (a[c].isna() & b[c].isna())).sum())
    report(f"cell_assignments.{c}", same == len(a), f"{same}/{len(a)}")

x = pd.read_csv(f"{O}/csv/recovery_counts.csv"); y = pd.read_csv(f"{N}/csv/recovery_counts.csv")
report("recovery_counts.csv", x.astype(str).equals(y.astype(str)))

x = pd.read_csv(f"{O}/csv/final_label_integer_key.csv", sep=";"); y = pd.read_csv(f"{N}/csv/final_label_integer_key.csv")
report("final_label_integer_key.csv", x.astype(str).equals(y.astype(str)))

x = pd.read_csv(f"{O}/csv/recovery_quantification.csv", sep=";", dtype=str)
y = pd.read_csv(f"{N}/csv/recovery_quantification.csv", dtype=str)
for c in ["recovery_%", "purity_%"]:
    y[c] = y[c].str.replace(".", "", regex=False)
report("recovery_quantification.csv", x.equals(y), "(ignoring Excel's lost decimal point)")

worst = 0.0
for f in sorted(glob.glob(f"{O}/heatmaps_final/*.csv")):
    x = pd.read_csv(f, index_col=0); y = pd.read_csv(f.replace(O, N), index_col=0)
    assert x.index.equals(y.index) and x.columns.equals(y.columns), f
    worst = max(worst, float(np.nanmax(np.abs(x.values - y.values))))
report("heatmap matrices", worst < 1e-9, f"(max diff {worst:.1e})")

ro = sorted(os.listdir(f"{O}/roi_assignments_final")); rn = sorted(os.listdir(f"{N}/roi_assignments_final"))
bad = [f for f in ro if f not in rn or not pd.read_csv(f"{O}/roi_assignments_final/{f}").astype(str)
       .equals(pd.read_csv(f"{N}/roi_assignments_final/{f}").astype(str))]
report("roi_assignments_final", ro == rn and not bad, f"({len(ro)} files, {len(bad)} differ)")

print("\nALL CHECKS PASSED - v7.6 reproduced exactly." if ok else "\nSOME CHECKS FAILED.")
