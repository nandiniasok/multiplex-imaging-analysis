"""
Spatial analysis re-run on v7.6 cell types with the ORIGINAL (June/July) tumour definition RESTORED.

Why: the July-3/June analyses defined tumour = v2 'pooled_group in {Tumor, Tumor/Erythroid}' (32,706 cells).
In v7.6 those same cells are scattered (NPM1_blasts 12k, Mixed progenitors 12k, Erythroid 3k, Unassigned 2.2k,
Artifact 0.5k, ...). Using only v7.6 'NPM1_blasts' as tumour lost half the tumour and dropped 13 ROIs.
FIX: tumour = the original v2 tumour cells (by Cell Id) UNION any v7.6 Unassigned/Artifact cell that is
NPM1_mut+ (Alexa647 z>0.5). => 38,382 tumour cells, 47/49 ROIs usable (same 2 skipped as the original).

PART 1  (= 1-June proximity analysis): cell-type composition of NON-tumour cells by distance-from-tumour
        zone (5 zones: <15 touching, 15-30 immediate, 30-60 local, 60-100 intermediate, 100+ far),
        Non-relapse (G1) vs Relapse (G2), Mann-Whitney + BH per zone. Cell types = v7.6 labels.
PART 2  (= July-3 immune pathway analysis): immune-cell pathway modules vs nearest-tumour zone
        (4 zones: Contact/Near/Mid/Far), per-machine z, per-zone G1 vs G2 + contact-minus-far gradient.
"""
import pandas as pd, numpy as np, os, glob, warnings
import scanpy as sc
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
warnings.filterwarnings("ignore")
PROJ="/tmp/claude-1000/-mnt-e-multiplex-imaging-analysis/fde39b44-8ec7-46e0-ba78-02a6ea34a165/scratchpad/proj"; RS=PROJ
OUT="/tmp/claude-1000/-mnt-e-multiplex-imaging-analysis/fde39b44-8ec7-46e0-ba78-02a6ea34a165/scratchpad/spatial_check"; os.makedirs(OUT,exist_ok=True)
UM=0.325; C_G1,C_G2="#2980B9","#C0392B"
def star(p): return "" if pd.isna(p) else ("***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")
GROUP1_ROIS={ # non-relapse
 "Machine A_Slides1_ROI_12","Machine A_Slides1_ROI_13","Machine A_Slides1_ROI_14","Machine A_Slides1_ROI_15",
 "Machine A_Slides1_ROI_16","Machine A_Slides2_ROI_10","Machine A_Slides2_ROI_11","Machine A_Slides2_ROI_12",
 "Machine A_Slides2_ROI_13","Machine A_Slides2_ROI_14","Machine A_Slides2_ROI_15","Machine A_Slides2_ROI_16",
 "Machine B_SN177_ROI_06","Machine B_SN177_ROI_07","Machine B_SN177_ROI_08","Machine B_SN177_ROI_09",
 "Machine B_SN177_ROI_10","Machine B_SN177_ROI_11","Machine B_SN177_ROI_12","Machine B_SN177_ROI_13",
 "Machine B_SN177_ROI_14","Machine B_SN177_ROI_15","Machine B_SN177_ROI_16"}
def grp(r): return "Group 1" if r in GROUP1_ROIS else "Group 2"
KEY=["unique_region","Cell Id"]

# ---- v7.6 labels + coords ----
m=pd.read_csv("/mnt/e/multiplex-imaging-analysis/data/output/Nandhu_version_final/csv/cell_assignments.csv"); m["v76_label"]=m["final_label"]; m["Cell Id"]=m["Cell Id"].astype(str)
m["roi"]=m.unique_region.str.replace("/","_",regex=False); m["group"]=m.roi.map(grp)
# ---- expression (Alexa647 for extra tumour + pathway module markers) ----
ad=sc.read_h5ad(f"{PROJ}/experiments/exp-04-03-2026/stage6_subclustering/final_adata.h5ad"); o=ad.obs.copy(); o["Cell Id"]=o["Cell Id"].astype(str)
aidx=pd.MultiIndex.from_arrays([o.unique_region,o["Cell Id"].astype(str)],names=KEY)
def col(n): return np.asarray(ad[:,n].X).ravel() if n in ad.var_names else pd.to_numeric(o[n],errors="coerce").values
MODULES={"T-cell exhaustion":["CD279 PD1 3 1 3","TIM 3 REAL818","CD223 REA351","BATF REA486","CD305 REA447"],
 "PI3K-AKT survival":["AKT Pan REA676","PI3K p85 alpha REAL589"],"MAPK-JNK stress":["JNK1 REAL1128","JNK2 REA1153"],
 "FAK adhesion/migration":["FAK pS910 REA407","CD44 REA690"],"TCR proximal signalling":["ZAP70 REA814","CD247 REAL1135"],
 "Proliferation":["Ki 67 REAL1047","PCNA REA858","RRM2 REAL1010","c myc REAL810"],"Wnt / beta-catenin":["beta Catenin REA480"],
 "p53 stress":["p53 REA1132"],"Type-I IFN / inflammation":["IRF 7 REA521","IFN gamma REAL788"],
 "Anti-apoptotic (Bcl-2)":["Bcl 2 REA872"],"Antigen presentation (HLA-DR)":["HLA DR REAL550"],"Epigenetic (EZH2)":["EZH2 REA907"]}
mods=sorted({x for v in MODULES.values() for x in v}|{"Alexa647 anti rabbit1"})
E=pd.DataFrame({mk:col(mk) for mk in mods}); E["unique_region"]=o.unique_region.values; E["Cell Id"]=o["Cell Id"].values
full=m.merge(E,on=KEY,how="left"); full["machine"]=full.machine.fillna("machine_A")
for mk in mods: full[f"z_{mk}"]=full.groupby("machine")[mk].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
# ---- tumour = v2 tumour UNION Alexa647+ in v7.6 Unassigned/Artifact ----
full["is_tumor"]=(full.v76_label=="NPM1_blasts").values
print("tumour (final NPM1_blasts):",int(full.is_tumor.sum()))
print(f"ROIs: {full.roi.nunique()} | G1 {full[full.group=='Group 1'].roi.nunique()} G2 {full[full.group=='Group 2'].roi.nunique()}")

TCELL={"CD4+ T","CD8+ T"}; BCELL={"Plasma cells"}
MYELOID={"Granulocyte","Macrophages","Monocytes","Mast cells","HLA-DR-hi APC (CD1c+)","Immature myeloid CD11c+"}
IMMUNE=TCELL|BCELL|MYELOID

# ============ PART 1 — proximity composition (5 zones, June) ============
Z5=[(0,15,"<15 µm\n(touching)"),(15,30,"15–30 µm\n(immediate)"),(30,60,"30–60 µm\n(local)"),(60,100,"60–100 µm\n(intermediate)"),(100,None,"100+ µm\n(far)")]
Z5L=[z[2] for z in Z5]
def zlab5(d):
    for lo,hi,l in Z5:
        if hi is None or d<hi: return l
    return Z5L[-1]
rows=[]
for roi,df in full.groupby("roi"):
    tum=df[df.is_tumor]; non=df[~df.is_tumor]
    if len(tum)<1 or len(non)<10: print("  skip",roi,"tum",len(tum)); continue
    d,_=cKDTree(tum[["Nuc X","Nuc Y Inv"]].values).query(non[["Nuc X","Nuc Y Inv"]].values,k=1)
    z=[zlab5(x*UM) for x in d]
    t=pd.DataFrame({"ct":non.v76_label.values,"zone":z})
    for (ct,zn),g in t.groupby(["ct","zone"]): rows.append({"roi":roi,"group":grp(roi),"cell_type":ct,"zone":zn,"n":len(g)})
det=pd.DataFrame(rows)
zt=det.groupby(["roi","zone"]).n.sum().rename("zt").reset_index(); det=det.merge(zt,on=["roi","zone"]); det["frac"]=det.n/det.zt
agg=det.groupby(["group","zone","cell_type"]).frac.agg(mean="mean",sem="sem",n="count").reset_index()
sr=[]
for (z,ct),s in det.groupby(["zone","cell_type"]):
    g1=s[s.group=="Group 1"].frac; g2=s[s.group=="Group 2"].frac
    if len(g1)>=4 and len(g2)>=4:
        _,p=mannwhitneyu(g1,g2,alternative="two-sided")
        sr.append({"zone":z,"cell_type":ct,"mean_G1_%":g1.mean()*100,"mean_G2_%":g2.mean()*100,"diff_G1_G2_%":(g1.mean()-g2.mean())*100,"p":p,"n_G1":len(g1),"n_G2":len(g2)})
cs=pd.DataFrame(sr); out=[]
for z,zz in cs.groupby("zone"):
    zz=zz.sort_values("p").reset_index(drop=True); n=len(zz); zz["p_adj"]=(zz.p*n/(np.arange(n)+1)).clip(upper=1.0); zz["sig"]=zz.p_adj<0.05; out.append(zz)
cs=pd.concat(out); cs.to_csv(f"{OUT}/PART1_composition_stats.csv",index=False); agg.to_csv(f"{OUT}/PART1_composition_agg.csv",index=False)
print("\nPART 1 — significant composition differences (v7.6 types, recovered tumour, BH<0.05):")
for _,r in cs[cs.sig].sort_values(["zone","p_adj"]).iterrows():
    hi="G1 non-relapse" if r["diff_G1_G2_%"]>0 else "G2 relapse"
    print(f"  {r.zone.split(chr(10))[0]:14s} {r.cell_type:34s} G1={r['mean_G1_%']:5.1f}% G2={r['mean_G2_%']:5.1f}% diff={r['diff_G1_G2_%']:+5.1f} p_adj={r.p_adj:.4f} higher={hi}")
print("ROIs used PART1:",det.roi.nunique())

# ============ PART 2 — immune pathway modules (4 zones, July3) ============
ZE=np.array([0,15,30,50,100.]); NZ=4; Z4L=["Contact\n<15 µm","Near\n15–30 µm","Mid/local\n30–50 µm","Far\n50–100 µm"]
prows=[]; grows=[]
for roi,df in full.groupby("roi"):
    tum=df[df.is_tumor]; ic=df[df.v76_label.isin(IMMUNE)].copy()
    if len(tum)<1 or len(ic)<20: continue
    d,_=cKDTree(tum[["Nuc X","Nuc Y Inv"]].values).query(ic[["Nuc X","Nuc Y Inv"]].values,k=1)
    ic["um"]=d*UM; ic=ic[ic.um<ZE[-1]]; ic["zi"]=np.clip(np.digitize(ic.um,ZE[1:-1]),0,NZ-1)
    for mod,mks in MODULES.items():
        zc=[f"z_{x}" for x in mks if f"z_{x}" in ic.columns]; ic["_s"]=ic[zc].mean(axis=1); pz={}
        for zi in range(NZ):
            c=ic[ic.zi==zi]
            if len(c)>=15: pz[zi]=c._s.mean(); prows.append({"roi":roi,"group":grp(roi),"module":mod,"zone":Z4L[zi],"zone_idx":zi,"score":c._s.mean(),"n":len(c)})
        if 0 in pz and NZ-1 in pz: grows.append({"roi":roi,"group":grp(roi),"module":mod,"gradient":pz[0]-pz[NZ-1]})
per=pd.DataFrame(prows); grad=pd.DataFrame(grows)
per.to_csv(f"{OUT}/PART2_pathway_per_roi_zone.csv",index=False); grad.to_csv(f"{OUT}/PART2_gradient_per_roi.csv",index=False)
pr=[]
for (z,mod),s in per.groupby(["zone","module"]):
    g1=s[s.group=="Group 1"].score; g2=s[s.group=="Group 2"].score
    if len(g1)>=4 and len(g2)>=4:
        _,p=mannwhitneyu(g1,g2,alternative="two-sided")
        pr.append({"zone":z,"module":mod,"mean_G1":g1.mean(),"mean_G2":g2.mean(),"diff_G1_G2":g1.mean()-g2.mean(),"p":p,"n_G1":len(g1),"n_G2":len(g2)})
pzs=pd.DataFrame(pr); out=[]
for z,zz in pzs.groupby("zone"):
    zz=zz.sort_values("p").reset_index(drop=True); n=len(zz); zz["p_adj"]=(zz.p*n/(np.arange(n)+1)).clip(upper=1.0); zz["sig"]=zz.p_adj<0.05; out.append(zz)
pzs=pd.concat(out); pzs.to_csv(f"{OUT}/PART2_pathway_stats_perzone.csv",index=False)
gr=[]
for mod,s in grad.groupby("module"):
    g1=s[s.group=="Group 1"].gradient; g2=s[s.group=="Group 2"].gradient
    if len(g1)>=4 and len(g2)>=4:
        _,p=mannwhitneyu(g1,g2,alternative="two-sided"); gr.append({"module":mod,"grad_G1":g1.mean(),"grad_G2":g2.mean(),"diff":g1.mean()-g2.mean(),"p":p,"n_G1":len(g1),"n_G2":len(g2)})
grs=pd.DataFrame(gr).sort_values("p").reset_index(drop=True); n=len(grs); grs["p_adj"]=(grs.p*n/(np.arange(n)+1)).clip(upper=1.0); grs["sig"]=grs.p_adj<0.05
grs.to_csv(f"{OUT}/PART2_gradient_stats.csv",index=False)
pc=pzs[pzs.zone.str.startswith("Contact")].sort_values("p_adj")
print("\nPART 2 — CONTACT-zone immune modules, G1 vs G2 (recovered tumour, all ROIs):")
for _,r in pc.iterrows():
    print(f"  {r.module:30s} G1={r.mean_G1:+.3f} G2={r.mean_G2:+.3f} diff={r.diff_G1_G2:+.3f} p={r.p:.4f} p_adj={r.p_adj:.4f} {star(r.p_adj)}")
print("ROIs used PART2:",per.roi.nunique(),"| G1",per[per.group=='Group 1'].roi.nunique(),"G2",per[per.group=='Group 2'].roi.nunique())
print("__DONE__")
