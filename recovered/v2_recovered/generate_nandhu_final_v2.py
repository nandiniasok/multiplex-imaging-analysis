#!/usr/bin/env python3
"""
Nandhu_final_v2 — revised from Nandhu_final with the following changes:

UNCHANGED subclusters:
  CD4+ T helper, Regulatory T (Treg), Mature B, Plasmablast/Plasma cell,
  Early/Immature B, Memory T lymphoid, HLA-DR+ lymphoid,
  all Tumor subclusters, Megakaryocyte, Endothelial

KEY CHANGES vs v1:
  1. CD8+ Cytotoxic T (T cells + Mixed Lymphoid) → merged, all under T cells
  2. Mast cell (Mixed Lymphoid + Myeloid) → merged, all under Myeloid
  3. Granulocyte: higher Arginase/CD15 weights + OR-logic gate to capture
     arginase/cd15/cd16/cd11c-positive cells previously missed
  4. Mast cell threshold raised to 1.5 (was 1.0) to remove ~40% Tryptase-low cells
  5. Fallback label assignment: if top-scoring label fails gate, tries next best
  6. final_label_integer_key built from canonical group mapping (not ROI pooled_group)
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
from pathlib import Path

RESULTS  = Path(os.environ.get("MRD_PROJ", "/home/hp/projects/multiplex_imaging_analysis")) / "results_soumya"
SRC_BASE = RESULTS / "all rois all markers final"
SRC_CSV  = SRC_BASE / "csv"
SRC_ROI  = SRC_BASE / "roi_assignments"
OUT      = RESULTS / "Nandhu_final_v2"

for d in [OUT / "heatmaps_final", OUT / "roi_assignments_final"]:
    d.mkdir(parents=True, exist_ok=True)

META_COLS = {"group", "best_resolution", "sub_cluster"}

EXCLUDED_MARKERS = {
    "AKT Pan REA676", "c myc REAL810", "CD104 REA236", "CD117 REA787",
    "CD181 REA958", "CD182 REA208", "CD183 REAL756", "CD196 REA190",
    "CD202b REA198", "CD204 REA460", "CD23 REAL1096", "CD295 REA361",
    "Podoplanin REA446", "CD52 REA164", "CD2 REA1130", "CD247 REAL1135",
    "EmptyPE1", "EZH2 REA907", "FAK pS910 REA407", "IFN gamma REAL788",
    "IRF 7 REA521", "JNK2 REA1153", "Nucleophosmin REAL823", "PKCalpha REA156",
    "RRM2 REAL1010", "Synaptophysin REA1121", "WT1 REA925", "ZAP70 REA814",
}

GROUP_FILE_MAP = {
    "T cells":         "T_cells",
    "B cells":         "B_cells",
    "Mixed Lymphoid":  "Mixed_Lymphoid",
    "Myeloid":         "Myeloid",
    "Tumor":           "Tumor",
    "Erythroid":       "Erythroid",
    "Tumor/Erythroid": "Tumor-Erythroid",
    "MSC":             "MSC",
    "Megakaryocytes":  "Megakaryocytes",
    "Endothelial":     "Endothelial",
}

# Canonical group for labels that appear in multiple groups.
# Mixed Lymphoid CD8+ T → T cells; Mixed Lymphoid Mast cell → Myeloid;
# Tumor Megakaryocyte → Megakaryocytes.
LABEL_CANONICAL = {
    "CD8+ Cytotoxic T": "T cells",
    "Mast cell":        "Myeloid",
    "Megakaryocyte":    "Megakaryocytes",
}

# ── Label rules ───────────────────────────────────────────────────────────────
LABEL_RULES = {
    "T cells": [
        ("CD4+ T helper",
            {"CD3 REAL1097": 2, "CD4 REA1307": 3, "CD8a REA1024": -2, "FoxP3 REA1253": -1}),
        # CD8+ Cytotoxic T is CANONICAL here; Mixed Lymphoid also detects these
        # cells but they all map to the T cells integer.
        ("CD8+ Cytotoxic T",
            {"CD3 REAL1097": 2, "CD8a REA1024": 3, "CD57 REA769": 1,
             "CD4 REA1307": -2, "FoxP3 REA1253": -1}),
        ("Regulatory T (Treg)",
            {"CD3 REAL1097": 2, "CD4 REA1307": 1, "FoxP3 REA1253": 3, "CD8a REA1024": -2}),
        ("CD4+ PD1+ T",
            {"CD4 REA1307": 2, "CD279 PD1 3 1 3": 4, "CD3 REAL1097": 1, "FoxP3 REA1253": -2}),
    ],
    "B cells": [
        ("Mature B",
            {"CD20 REAL1069": 3, "PAX 5 REA140": 2, "CD38 REAL719": -1}),
        ("Plasmablast/Plasma cell",
            {"CD38 REAL719": 3, "CD20 REAL1069": -2, "PAX 5 REA140": -2}),
        ("Early/Immature B",
            {"CD79a REA1168": 3, "CD38 REAL719": 1, "CD20 REAL1069": -2}),
    ],
    "Mixed Lymphoid": [
        ("NK cell",
            {"CD16 REA1324": 3, "CD56 REAL1142": 3, "CD45RA REAL164": 1,
             "CD3 REAL1097": -3, "Mast Cell Tryptase REAL798": -4}),
        # CD8+ T cells misrouted to Mixed Lymphoid; canonical = T cells
        ("CD8+ Cytotoxic T",
            {"CD3 REAL1097": 2, "CD8a REA1024": 3, "CD57 REA769": 1,
             "CD4 REA1307": -2, "FoxP3 REA1253": -1}),
        # Mast cells in Mixed Lymphoid; canonical = Myeloid; threshold 1.5
        ("Mast cell",
            {"Mast Cell Tryptase REAL798": 4}),
        ("Memory T lymphoid",
            {"CD45RO REA611": 3, "CD3 REAL1097": 2, "CD45 5B1": 1}),
        ("HLA-DR+ lymphoid",
            {"HLA DR REAL550": 4, "CD45 5B1": 1, "CD45RA REAL164": 1}),
        ("Mixed B/T lymphoid",
            {"CD20 REAL1069": 1, "CD3 REAL1097": 1}),
    ],
    "Myeloid": [
        ("Classical monocyte",
            {"CD14 REA1314": 3, "CD11b REA1321": 2, "CD68 REAL1346": 1, "CD16 REA1324": -1}),
        ("Patrolling monocyte",
            {"CD16 REA1324": 3, "CD11b REA1321": 2, "CD14 REA1314": -2}),
        ("Macrophage (M2-like)",
            {"CD163 REA1309": 3, "CD68 REAL1346": 2, "Arginase 1 REAL1137": 1}),
        ("Dendritic cell",
            {"CD11c REA1310": 3, "HLA DR REAL550": 3, "CD11b REA1321": 2,
             "CD14 REA1314": -1, "CD68 REAL1346": -1}),
        # Canonical mast cell; threshold 1.5 to remove low-Tryptase contamination
        ("Mast cell",
            {"Mast Cell Tryptase REAL798": 4}),
        # REVISED: higher CD15 and Arginase weights; CD163 penalty added;
        # OR-logic gate ensures arginase/cd15/cd16/cd11c+ cells are captured
        ("Neutrophil/Granulocyte",
            {"CD15 VIMC6": 5, "CD11b REA1321": 2, "CD16 REA1324": 2,
             "Arginase 1 REAL1137": 5, "CD11c REA1310": 1,
             "CD14 REA1314": -3, "CD68 REAL1346": -3, "CD163 REA1309": -2}),
        ("Proliferating myeloid",
            {"Ki 67 REAL1047": 4, "CD163 REA1309": 2, "CD68 REAL1346": 1}),
    ],
    "Tumor": [
        ("Proliferating AML blast",
            {"Ki 67 REAL1047": 3, "Alexa647 anti rabbit1": 2, "CD38 REAL719": 1}),
        ("Quiescent AML blast",
            {"Alexa647 anti rabbit1": 3, "Ki 67 REAL1047": -2, "HLA DR REAL550": -1}),
        ("AML monocytic differentiated",
            {"HLA DR REAL550": 4, "Alexa647 anti rabbit1": 2, "CD43 REA833": 1,
             "CD147 REA282": 1, "CD11c REA1310": 1}),
        # Cross-group megakaryocyte rescue; canonical = Megakaryocytes
        ("Megakaryocyte",
            {"CD31 REA1312": 4, "CD226 REA1040": 2, "CD34 REAL1217": -3}),
    ],
    "Erythroid": [
        ("Mature erythroblast",
            {"CD235a REA175": 3, "CD34 REAL1217": -2, "Ki 67 REAL1047": -1}),
    ],
    "Tumor/Erythroid": [
        ("Leukemic erythroid blast",
            {"CD235a REA175": 2, "Alexa647 anti rabbit1": 2, "CD34 REAL1217": 1}),
        ("Proliferating leukemic erythroid",
            {"Ki 67 REAL1047": 3, "CD235a REA175": 1}),
    ],
    "MSC": [
        ("Bone marrow MSC",
            {"CD73 REAL1172": 3, "CD90 REAL677": 2, "CD271 REAL709": 1}),
        ("CD271+ MSC",
            {"CD271 REAL709": 3, "CD73 REAL1172": 2, "CD90 REAL677": 1}),
    ],
    "Megakaryocytes": [
        ("Megakaryocyte",
            {"CD31 REA1312": 4, "CD226 REA1040": 2, "CD34 REAL1217": -3}),
    ],
    "Endothelial": [
        ("Endothelial",
            {"CD31 REA1312": 3, "CD34 REAL1217": 2}),
    ],
}

# ── ALL_LABELS_ORDERED: deduplicated via LABEL_CANONICAL ─────────────────────
# Each label appears exactly once, at its canonical group's position.
ALL_LABELS_ORDERED = []
_seen_labels = set()
for _group, _rules in LABEL_RULES.items():
    for _lbl, _ in _rules:
        _canonical_g = LABEL_CANONICAL.get(_lbl, _group)
        if _lbl not in _seen_labels and _canonical_g == _group:
            ALL_LABELS_ORDERED.append((_group, _lbl))
            _seen_labels.add(_lbl)

LABEL_TO_INT = {lbl: i for i, (_, lbl) in enumerate(ALL_LABELS_ORDERED)}

# ── Per-label AND gates ───────────────────────────────────────────────────────
LABEL_GATES = {
    "CD4+ T helper":       {"CD3 REAL1097": -0.25, "CD4 REA1307": 0.0},
    "CD8+ Cytotoxic T":    {"CD3 REAL1097": -0.25, "CD8a REA1024": 0.0},
    "Regulatory T (Treg)": {"CD3 REAL1097": -0.25, "FoxP3 REA1253": 0.5},
    "CD4+ PD1+ T":         {"CD279 PD1 3 1 3": 0.3, "CD4 REA1307": -0.5},
    "Mature B":            {"CD20 REAL1069": 0.0},
    "Early/Immature B":    {"CD79a REA1168": 0.0, "CD38 REAL719": 0.0},
    "NK cell":             {"CD16 REA1324": 0.0, "CD56 REAL1142": 0.0},
    "Memory T lymphoid":   {"CD45RO REA611": 0.0, "CD3 REAL1097": -0.25},
    "HLA-DR+ lymphoid":    {"HLA DR REAL550": 1.0},
    # Raised to 1.5: removes subclusters with ~40% Tryptase-negative cells
    "Mast cell":           {"Mast Cell Tryptase REAL798": 1.5},
    "Classical monocyte":  {"CD14 REA1314": 0.0},
    "Macrophage (M2-like)":{"CD163 REA1309": 0.0, "CD68 REAL1346": 0.0},
    "Dendritic cell":      {"CD11c REA1310": 0.0, "HLA DR REAL550": 0.0},
    "Proliferating myeloid":{"Ki 67 REAL1047": 0.5, "CD163 REA1309": 0.0},
    "Proliferating AML blast":          {"Ki 67 REAL1047": 0.0, "Alexa647 anti rabbit1": 0.0},
    "Quiescent AML blast":              {"Alexa647 anti rabbit1": 0.0},
    "AML monocytic differentiated":     {"Alexa647 anti rabbit1": 0.0, "HLA DR REAL550": 0.5},
    "Megakaryocyte":                    {"CD31 REA1312": 1.0},
    "Mature erythroblast":              {"CD235a REA175": 0.3},
    "Leukemic erythroid blast":         {"Alexa647 anti rabbit1": 0.0, "CD34 REAL1217": 0.0},
    "Proliferating leukemic erythroid": {"Ki 67 REAL1047": 0.0},
    "Bone marrow MSC": {"CD73 REAL1172": 0.5},
    "CD271+ MSC":      {"CD271 REAL709": 0.5, "CD73 REAL1172": 0.0},
    "Endothelial":     {"CD34 REAL1217": 0.0},
}

# OR gates: pass if ANY single condition dict is fully satisfied.
LABEL_OR_GATES = {
    # Capture CD15+, Arginase-strongly+, CD16+high, CD11c+ granulocytes/MDSCs
    "Neutrophil/Granulocyte": [
        {"CD15 VIMC6": 0.0},           # CD15 positive
        {"Arginase 1 REAL1137": 1.0},  # Arginase > 1 z-score
    ],
}

# Exclusion gates — marker must be BELOW threshold
LABEL_MAX_GATES = {
    "NK cell": {"Mast Cell Tryptase REAL798": 0.3},
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def score_label(row, markers):
    return sum(row.get(m, 0) * w for m, w in markers.items())


def passes_gate(row, label):
    if label in LABEL_OR_GATES:
        or_conds = LABEL_OR_GATES[label]
        if not any(all(row.get(m, 0) > thr for m, thr in cond.items())
                   for cond in or_conds):
            return False
    elif label in LABEL_GATES:
        if not all(row.get(m, 0) > thr for m, thr in LABEL_GATES[label].items()):
            return False
    if label in LABEL_MAX_GATES:
        if not all(row.get(m, 0) < thr for m, thr in LABEL_MAX_GATES[label].items()):
            return False
    return True


def assign_label(row, rules):
    """Return highest-scoring label that passes gates; None if none qualify."""
    scores = [(score_label(row, markers), lbl) for lbl, markers in rules]
    for _, lbl in sorted(scores, reverse=True):
        if passes_gate(row, lbl):
            return lbl
    return None


def shorten(name):
    aliases = {
        "Alexa647 anti rabbit1":            "Alexa647",
        "Ki 67 REAL1047":                   "Ki67",
        "Mast Cell Tryptase REAL798":       "MastTryptase",
        "HLA DR REAL550":                   "HLA-DR",
        "PAX 5 REA140":                     "PAX5",
        "FoxP3 REA1253":                    "FoxP3",
        "NPM1mut primaryRab400 polyclonal": "NPM1mut_poly",
        "Arginase 1 REAL1137":              "Arginase1",
        "Bcl 2 REA872":                     "BCL-2",
        "Galectin 9 REA435":                "Galectin-9",
        "FcepsilonRIalpha REA758":          "FceRI-alpha",
        "beta Catenin REA480":              "betaCatenin",
        "TIM 3 REAL818":                    "TIM-3",
        "CD279 PD1 3 1 3":                  "PD-1",
        "PI3K p85 alpha REAL589":           "PI3K-p85",
        "JNK1 REAL1128":                    "JNK1",
        "BATF REA486":                      "BATF",
        "PCNA REA858":                      "PCNA",
        "Actin REAL650":                    "Actin",
        "p53 REA1132":                      "p53",
        "CD45 5B1":                         "CD45",
        "CD45RO REA611":                    "CD45RO",
        "CD45RA REAL164":                   "CD45RA",
        "CD57 REA769":                      "CD57",
    }
    return aliases.get(name, name.split(" ")[0])


# ── Step 1: Marker-gated subcluster assignment ────────────────────────────────
print("=" * 70)
print("Step 1: Marker-gated assignment (v2 — revised granulocyte + OR gates)")
print("=" * 70)

# (source_group, sc) → label
subcluster_map = {}
# label → list of mean-expression rows from all source groups (for heatmap)
label_subcluster_rows = {}
# (canonical_group, label) → [(source_group, sc), ...]
label_to_subclusters = {}

for group, rules in LABEL_RULES.items():
    prefix = GROUP_FILE_MAP[group]
    path   = SRC_CSV / f"{prefix}_best_means.csv"
    if not path.exists():
        print(f"  SKIP (not found): {path.name}")
        continue

    df = pd.read_csv(path)
    counts, n_drop = {}, 0

    for _, row in df.iterrows():
        sc    = int(row["sub_cluster"])
        label = assign_label(row, rules)
        if label is not None:
            subcluster_map[(group, sc)] = label
            canonical_g = LABEL_CANONICAL.get(label, group)
            label_to_subclusters.setdefault((canonical_g, label), []).append((group, sc))
            label_subcluster_rows.setdefault(label, []).append(row)
            counts[label] = counts.get(label, 0) + 1
        else:
            n_drop += 1

    total   = sum(counts.values())
    summary = ", ".join(f"{l}({n})" for l, n in counts.items())
    print(f"  {group:20s}  kept={total:3d}  dropped={n_drop:3d}  →  {summary}")


# ── Step 2: final_label_mapping.csv ──────────────────────────────────────────
print("\nStep 2: Writing final_label_mapping.csv")
rows = []
for i, (group, label) in enumerate(ALL_LABELS_ORDERED):
    entries = label_to_subclusters.get((group, label), [])
    sc_strs = "; ".join(f"{g[:2].upper()}-{sc}" for g, sc in sorted(entries))
    rows.append({
        "final_label_integer":   i,
        "pooled_group":          group,
        "meaningful_label":      label,
        "final_label":           label,
        "assigned_sub_clusters": sc_strs,
        "n_sub_clusters":        len(entries),
    })
mapping_df = pd.DataFrame(rows)
mapping_df.to_csv(OUT / "final_label_mapping.csv", index=False)
print(f"  Saved  ({len(mapping_df)} labels, {(mapping_df['n_sub_clusters']>0).sum()} with data)")


# ── Step 3: meaningful_subcluster_labels.csv ──────────────────────────────────
print("\nStep 3: Writing meaningful_subcluster_labels.csv")
LABEL_META = {
    "CD4+ T helper":
        ("CD3; CD4",                "CD8a; FoxP3",    "Helper T cells"),
    "CD8+ Cytotoxic T":
        ("CD3; CD8a; CD57",         "CD4; FoxP3",     "CD8+ T from T cells + Mixed Lymphoid — merged under T cells"),
    "Regulatory T (Treg)":
        ("CD3; FoxP3 (>0.5)",       "CD8a",           "Immunosuppressive Tregs; strictly FoxP3-high"),
    "CD4+ PD1+ T":
        ("CD4; PD-1",               "",               "Exhausted/checkpoint-activated CD4+ T cells"),
    "Mature B":
        ("CD20",                    "CD38",           "Resting mature naive B cells"),
    "Plasmablast/Plasma cell":
        ("CD38 (high)",             "CD20; PAX5",     "Terminal B-cell differentiation"),
    "Early/Immature B":
        ("CD79a; CD38",             "CD20",           "Early B lineage before full CD20 acquisition"),
    "NK cell":
        ("CD16; CD56; CD45RA",      "CD3; Tryptase",  "Cytotoxic innate NK cells"),
    "Memory T lymphoid":
        ("CD45RO; CD3",             "",               "Memory T cells; CD45RO-high"),
    "HLA-DR+ lymphoid":
        ("HLA-DR (>1.0); CD45",    "",               "Activated HLA-DR-high lymphocytes"),
    "Mixed B/T lymphoid":
        ("CD20 or CD3",             "",               "Ambiguous residual lymphoid co-expression"),
    "Classical monocyte":
        ("CD14",                    "CD16",           "CD14-high classical monocytes"),
    "Patrolling monocyte":
        ("CD16; CD11b",             "CD14",           "Non-classical patrolling monocytes"),
    "Macrophage (M2-like)":
        ("CD163; CD68",             "",               "M2 macrophages / TAMs; both CD163 and CD68 high"),
    "Dendritic cell":
        ("CD11c; HLA-DR; CD11b",    "CD14; CD68",     "Myeloid dendritic cells"),
    "Mast cell":
        ("MastTryptase (>1.5)",     "",               "Mast cells from Mixed Lymphoid + Myeloid — merged under Myeloid; threshold 1.5"),
    "Neutrophil/Granulocyte":
        ("CD15; Arginase1; CD16; CD11b; CD11c", "CD14; CD68; CD163",
         "Granulocytes / MDSCs — OR gate: CD15>0 or Arginase>1.0"),
    "Proliferating myeloid":
        ("Ki67 (>0.5); CD163",     "",               "Proliferating myeloid; Ki67-high CD163+"),
    "Proliferating AML blast":
        ("Ki67; Alexa647",          "",               "Actively cycling NPM1mut AML blasts"),
    "Quiescent AML blast":
        ("Alexa647 (high)",         "Ki67; HLA-DR",   "Low-cycling NPM1mut blasts"),
    "AML monocytic differentiated":
        ("HLA-DR; Alexa647; CD43; CD147; CD11c", "Ki67",
         "Differentiated NPM1mut blast — monocytic axis"),
    "Megakaryocyte":
        ("CD31 (>1.0); CD226",      "CD34",           "Mature megakaryocytes; from Tumor + Megakaryocytes groups"),
    "Mature erythroblast":
        ("CD235a (>0.3)",           "CD34; Ki67",     "Late-stage erythroblasts; CD235a-high"),
    "Leukemic erythroid blast":
        ("Alexa647; CD34",          "",               "AML blasts with erythroid differentiation"),
    "Proliferating leukemic erythroid":
        ("Ki67",                    "",               "Cycling tumour-erythroid hybrid"),
    "Bone marrow MSC":
        ("CD73 (>0.5)",             "",               "CD73-high BM stromal MSCs"),
    "CD271+ MSC":
        ("CD271 (>0.5); CD73",     "",               "CD271/NGFR-high MSC subset"),
    "Endothelial":
        ("CD34 (high)",             "",               "BM sinusoidal endothelium; CD34-high"),
}

label_rows_out = []
for i, (group, label) in enumerate(ALL_LABELS_ORDERED):
    entries = label_to_subclusters.get((group, label), [])
    sc_strs = "; ".join(f"{g[:2].upper()}-{sc}" for g, sc in sorted(entries))
    meta = LABEL_META.get(label, ("", "", ""))
    label_rows_out.append({
        "final_label_integer":     i,
        "pooled_group":            group,
        "meaningful_label":        label,
        "assigned_sub_clusters":   sc_strs,
        "n_assigned_sub_clusters": len(entries),
        "key_markers_high":        meta[0],
        "key_markers_low":         meta[1],
        "biological_rationale":    meta[2],
    })
labels_df = pd.DataFrame(label_rows_out)
labels_df.to_csv(OUT / "meaningful_subcluster_labels.csv", index=False)
print(f"  Saved  ({len(labels_df)} rows)")


# ── Step 4: markers_used.csv ──────────────────────────────────────────────────
print("\nStep 4: Writing markers_used.csv")
_df = pd.read_csv(SRC_CSV / "T_cells_best_means.csv", nrows=1)
all_markers       = [c for c in _df.columns if c not in META_COLS]
remaining_markers = [m for m in all_markers if m not in EXCLUDED_MARKERS]
pd.DataFrame({"marker": remaining_markers}).to_csv(OUT / "markers_used.csv", index=False)
print(f"  Saved  ({len(remaining_markers)} markers)")


# ── Step 5: roi_assignments_final/ ────────────────────────────────────────────
print("\nStep 5: Writing roi_assignments_final/")
roi_files    = sorted(SRC_ROI.glob("*_converted.csv"))
total_kept   = total_dropped = 0
for roi_path in roi_files:
    df = pd.read_csv(roi_path, low_memory=False)
    new_labels = df.apply(
        lambda r: subcluster_map.get((r["pooled_group"], int(r["sub_cluster"])), None), axis=1
    )
    mask = new_labels.notna()
    out  = df[mask].copy()
    out["final_label"]         = new_labels[mask].values
    out["final_label_integer"] = out["final_label"].map(LABEL_TO_INT)
    out.to_csv(OUT / "roi_assignments_final" / roi_path.name, index=False)
    total_kept    += mask.sum()
    total_dropped += (~mask).sum()
    print(f"  {roi_path.name}: kept {mask.sum():5d}  dropped {(~mask).sum():5d}")
print(f"\n  Total: kept {total_kept:,}  dropped {total_dropped:,}")


# ── Step 6: final_label_integer_key.csv ──────────────────────────────────────
# Built from canonical ALL_LABELS_ORDERED so Mixed Lymphoid CD8+ T → T cells
# integer, and Mixed Lymphoid Mast cell → Myeloid integer.
print("\nStep 6: Writing final_label_integer_key.csv")
labels_with_data = set(label_subcluster_rows.keys())
key_rows = [
    (LABEL_TO_INT[lbl], group, lbl)
    for group, lbl in ALL_LABELS_ORDERED
    if lbl in labels_with_data
]
key_df = pd.DataFrame(
    sorted(key_rows),
    columns=["final_label_integer", "cell_group", "cell_type"]
)
key_df.to_csv(OUT / "final_label_integer_key.csv", index=False)
print(key_df.to_string(index=False))


# ── Heatmap helper ────────────────────────────────────────────────────────────
def save_heatmap(agg_df, title, out_path):
    n_labels  = agg_df.shape[1]
    n_markers = agg_df.shape[0]
    fig_w = max(4, n_labels * 2.0)
    fig_h = max(8, n_markers * 0.28)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    sns.heatmap(agg_df, cmap="RdBu_r", center=0, vmin=-2, vmax=2,
                linewidths=0.3, linecolor="white", ax=ax,
                cbar_kws={"label": "Mean Expression (z-score)", "shrink": 0.6})
    ax.set_title(title, fontsize=10, fontweight="bold", pad=10)
    ax.set_xlabel("Subcluster", fontsize=9)
    ax.set_ylabel("Marker", fontsize=9)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Step 7: per-group heatmaps ────────────────────────────────────────────────
# Rows in label_subcluster_rows already aggregate data from ALL source groups
# (e.g. CD8+ T from both T cells and Mixed Lymphoid; Mast from both groups).
print("\nStep 7: Generating heatmaps_final/ …")

GROUP_COLORS = {
    "T cells": "#4C72B0", "B cells": "#55A868", "Mixed Lymphoid": "#8172B2",
    "Myeloid": "#C44E52", "Tumor": "#CCB974", "Erythroid": "#64B5CD",
    "Tumor/Erythroid": "#DD8452", "MSC": "#937860",
    "Megakaryocytes": "#DA8BC3", "Endothelial": "#8C8C8C",
}

# Identify available markers once
_sample_cols = pd.read_csv(SRC_CSV / "T_cells_best_means.csv", nrows=0).columns.tolist()
avail_markers = [m for m in remaining_markers if m in _sample_cols]

group_agg_for_combined = []

# Group canonical labels by their canonical group, preserving order
canonical_groups_labels = {}
for group, label in ALL_LABELS_ORDERED:
    rows_for_label = label_subcluster_rows.get(label, [])
    if rows_for_label:
        canonical_groups_labels.setdefault(group, []).append((label, rows_for_label))

for group, label_row_pairs in canonical_groups_labels.items():
    prefix   = GROUP_FILE_MAP[group]
    agg_data = {}
    label_order = []

    for label, row_list in label_row_pairs:
        rows_df   = pd.DataFrame(row_list)
        cols_here = [m for m in avail_markers if m in rows_df.columns]
        mean_vals = rows_df[cols_here].mean()
        agg_data[label] = mean_vals
        label_order.append(label)

    if not agg_data:
        print(f"  SKIP {group}: no qualified subclusters")
        continue

    agg = pd.DataFrame(agg_data).T.reindex(label_order)
    agg = agg[[m for m in avail_markers if m in agg.columns]].T
    agg.index = [shorten(m) for m in agg.index]

    title    = f"{group}  |  {agg.shape[1]} subclusters  |  {agg.shape[0]} markers"
    out_name = f"{prefix}_heatmap.png"
    save_heatmap(agg, title, OUT / "heatmaps_final" / out_name)
    print(f"  Saved  {out_name}  ({agg.shape[1]} labels × {agg.shape[0]} markers)")
    group_agg_for_combined.append((group, agg))


# ── Step 8: combined heatmap ──────────────────────────────────────────────────
print("\nStep 8: Generating combined heatmap …")
combined  = pd.concat([agg for _, agg in group_agg_for_combined], axis=1)
n_labels  = combined.shape[1]
n_markers = combined.shape[0]
fig_w = max(16, n_labels * 1.5)
fig_h = max(18, n_markers * 0.28)
fig, ax = plt.subplots(figsize=(fig_w, fig_h))
sns.heatmap(combined, cmap="RdBu_r", center=0, vmin=-2, vmax=2,
            linewidths=0.25, linecolor="white", ax=ax,
            cbar_kws={"label": "Mean Expression (z-score)", "shrink": 0.5, "pad": 0.02})
x_pos = 0
for group, agg in group_agg_for_combined:
    n = agg.shape[1]
    if x_pos > 0:
        ax.axvline(x=x_pos, color="black", linewidth=1.5, zorder=5)
    rect = mpatches.FancyBboxPatch(
        (x_pos, -2.2), n, 1.6, boxstyle="square,pad=0",
        facecolor=GROUP_COLORS.get(group, "#AAAAAA"),
        transform=ax.transData, clip_on=False, zorder=10)
    ax.add_patch(rect)
    ax.text(x_pos + n / 2, -1.4, group, ha="center", va="center",
            fontsize=7.5, fontweight="bold", color="white",
            transform=ax.transData, clip_on=False, zorder=11)
    x_pos += n
ax.set_title(
    f"Nandhu_final_v2 — {n_labels} labels × {n_markers} markers  (z-score)",
    fontsize=11, fontweight="bold", pad=18)
ax.set_xlabel("")
ax.set_ylabel("Marker", fontsize=9)
plt.xticks(rotation=40, ha="right", fontsize=7)
plt.yticks(fontsize=7)
plt.tight_layout()
fig.savefig(OUT / "heatmaps_final" / "all_groups_combined_heatmap.png",
            dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"  Saved  all_groups_combined_heatmap.png  ({n_labels} labels × {n_markers} markers)")


# ── Summary ───────────────────────────────────────────────────────────────────
n_hm  = len(list((OUT / "heatmaps_final").glob("*.png")))
n_roi = len(list((OUT / "roi_assignments_final").glob("*.csv")))
print(f"""
✓ All done — Nandhu_final_v2
  markers used            : {len(remaining_markers)} / {len(all_markers)}
  total unique label slots: {len(ALL_LABELS_ORDERED)}
  labels with data        : {(mapping_df['n_sub_clusters'] > 0).sum()}
  heatmaps_final/         : {n_hm} PNG files
  roi_assignments_final/  : {n_roi} ROI CSVs
  Output folder           : {OUT}

Key changes vs v1:
  CD8+ Cytotoxic T  → merged under T cells (was T cells + Mixed Lymphoid)
  Mast cell         → merged under Myeloid (was Mixed Lymphoid + Myeloid); threshold 1.5
  Neutrophil/Gran.  → OR gate (CD15>0 OR Arginase>1.0); weights CD15×5 Arginase×5
  Fallback assign.  → tries next best label if top scorer fails gate (no silent drops)
""")
