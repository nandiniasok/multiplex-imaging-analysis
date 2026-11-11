# %% [markdown]
# # UMAP projections and cell-type composition — Nandhu_version_final
# * The embedding uses `final_adata.X`: the 25 selected markers, double z-normalised per machine. These are the same values
#   used for the original clustering. Steps: PCA (15 PCs) → Harmony correction for machine → UMAP (umap-learn, 15 neighbours,
#   min_dist 0.3, random_state 0). The uncorrected UMAP is kept for comparison in fig 4.
#   **No clustering is run.** Cells are only coloured by their Nandhu_version_final labels.
# * Plots: UMAP by cell type, Relapse vs Non-relapse, machine (a batch check), and selected markers; cell-type composition
#   per ROI, per group, and per patient (`../roi_patient_mapping.csv`).

# %%
import os, warnings
import numpy as np, pandas as pd, scanpy as sc
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
REC=os.environ.get("MRD_REC","/mnt/e/MRD analysis Apr_2026/30 May 2026_Claude new results/new 18.06.26")
PROJ=os.environ.get("MRD_PROJ",f"{REC}/proj_recovered")
ADATA=f"{PROJ}/experiments/exp-04-03-2026/stage6_subclustering/final_adata.h5ad"
BASE="/mnt/e/multiplex-imaging-analysis/data/output"; FINAL=f"{BASE}/Nandhu_version_final/csv/cell_assignments.csv"
OUT=os.environ.get("MRD_OUT",f"{BASE}/UMAP_Graphs"); os.makedirs(OUT,exist_ok=True)
C_NR,C_R="#2a78d6","#e34948"; GN,GR="Non-relapse","Relapse"
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":200,
                     "axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
NONREL={"Machine A_Slides1_ROI_12","Machine A_Slides1_ROI_13","Machine A_Slides1_ROI_14","Machine A_Slides1_ROI_15",
 "Machine A_Slides1_ROI_16","Machine A_Slides2_ROI_10","Machine A_Slides2_ROI_11","Machine A_Slides2_ROI_12",
 "Machine A_Slides2_ROI_13","Machine A_Slides2_ROI_14","Machine A_Slides2_ROI_15","Machine A_Slides2_ROI_16",
 "Machine B_SN177_ROI_06","Machine B_SN177_ROI_07","Machine B_SN177_ROI_08","Machine B_SN177_ROI_09",
 "Machine B_SN177_ROI_10","Machine B_SN177_ROI_11","Machine B_SN177_ROI_12","Machine B_SN177_ROI_13",
 "Machine B_SN177_ROI_14","Machine B_SN177_ROI_15","Machine B_SN177_ROI_16"}
def grp(r): return GN if r in NONREL else GR
def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); r=np.empty(n); r[o]=np.arange(1,n+1)
    q=p*n/r; qs=np.minimum.accumulate(q[o][::-1])[::-1]; out=np.empty(n); out[o]=np.clip(qs,0,1); return out
def star(p): return "" if pd.isna(p) else ("***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")
COL={"CD4+ T":"#1c5cab","CD8+ T":"#3987e5","Plasma cells":"#86b6ef","Granulocyte":"#eb6834","Macrophages":"#e34948",
     "Monocytes":"#e87ba4","Mast cells":"#eda100","HLA-DR-hi APC (CD1c+)":"#6250d6","Immature myeloid CD11c+":"#9085e9",
     "Erythroid":"#8c2b2b","NPM1_blasts":"#1a1a19","Mixed progenitors":"#a8a79f","Endothelial":"#1baf7a","MSC":"#008300",
     "Megakaryocyte":"#8a5a00","Unassigned-Leiden: CD68+CD3+Tryptase+":"#c3c2b7","Unassigned-Leiden: Ki67+CD147+CD45RA+":"#77756e",
     "Unassigned/Other":"#e6e5e0","Artifact":"#f3f2ee"}

# %% [markdown]
# ## Embedding
# %%
ad=sc.read_h5ad(ADATA); ad.obs["Cell Id"]=ad.obs["Cell Id"].astype(str)
lab=pd.read_csv(FINAL,dtype={"Cell Id":str}).set_index(["unique_region","Cell Id"])
key=pd.MultiIndex.from_arrays([ad.obs.unique_region.values,ad.obs["Cell Id"].values])
ad.obs["cell_type"]=lab.final_label.reindex(key).values; assert ad.obs.cell_type.notna().all()
ad.obs["roi"]=ad.obs.unique_region.str.replace("/","_",regex=False).values; ad.obs["group"]=ad.obs.roi.map(grp).values
UMAP_CSV=f"{OUT}/umap_coordinates.csv"
if os.path.exists(UMAP_CSV):   # reuse the saved embedding so reruns give identical figures
    u=pd.read_csv(UMAP_CSV,dtype={"Cell Id":str}).set_index(["unique_region","Cell Id"]).reindex(key)
    ad.obsm["X_umap"]=u[["UMAP1","UMAP2"]].values; ad.obsm["X_umap_raw"]=u[["UMAP1_uncorrected","UMAP2_uncorrected"]].values
else:
    # The uncorrected UMAP separates Machine A and Machine B cells (see fig4), and machine is partly confounded with
    # relapse status. The main UMAP is therefore built on Harmony-corrected PCs (batch = machine). This changes ONLY the
    # 2-D picture; no labels are touched and nothing is clustered.
    # umap-learn is called directly on a C-contiguous matrix: scanpy's neighbors→umap route hits a numba contiguity bug
    # with umap-learn 0.6 / numba 0.68 at this data size.
    import umap, harmonypy
    sc.pp.pca(ad,n_comps=15,random_state=0)
    ho=harmonypy.run_harmony(ad.obsm["X_pca"],ad.obs[["machine"]],"machine",random_state=0)
    Zh=np.asarray(ho.Z_corr); Zh=Zh.T if Zh.shape[0]!=ad.n_obs else Zh
    fit=lambda M: umap.UMAP(n_neighbors=15,min_dist=0.3,random_state=0).fit_transform(np.ascontiguousarray(M,dtype=np.float32))
    ad.obsm["X_umap"]=fit(Zh); ad.obsm["X_umap_raw"]=fit(ad.obsm["X_pca"])
    pd.DataFrame({"unique_region":ad.obs.unique_region.values,"Cell Id":ad.obs["Cell Id"].values,"cell_type":ad.obs.cell_type.values,
                  "group":ad.obs.group.values,"machine":ad.obs.machine.values,"UMAP1":ad.obsm["X_umap"][:,0],"UMAP2":ad.obsm["X_umap"][:,1],
                  "UMAP1_uncorrected":ad.obsm["X_umap_raw"][:,0],"UMAP2_uncorrected":ad.obsm["X_umap_raw"][:,1]}).to_csv(UMAP_CSV,index=False)
U=ad.obsm["X_umap"]; ct=ad.obs.cell_type.values; g=ad.obs.group.values
rng=np.random.default_rng(0); perm=rng.permutation(len(U))   # random draw order: no cell type/group is painted on top systematically
print(f"{ad.n_obs:,} cells embedded")

def frame(ax):
    ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False); ax.set_xlabel("UMAP1"); ax.set_ylabel("UMAP2")

# %% [markdown]
# ## UMAP by cell type (direct labels at each type's median position)
# %%
fig,ax=plt.subplots(figsize=(11,9.5))
ax.scatter(U[perm,0],U[perm,1],s=0.6,c=[COL[x] for x in ct[perm]],linewidths=0,rasterized=True)
for t in COL:
    mk=ct==t
    if mk.sum()<200 or t in ("Unassigned/Other","Artifact"): continue
    xy=np.median(U[mk],axis=0); ax.text(xy[0],xy[1],t.replace("Unassigned-Leiden: ","Leiden: "),fontsize=7.5,ha="center",va="center",
        bbox=dict(boxstyle="round,pad=0.2",fc="white",ec=COL[t],lw=1,alpha=0.85))
for t in COL:
    if (ct==t).any(): ax.scatter([],[],s=30,c=COL[t],label=f"{t} ({(ct==t).sum():,})")
ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=7.5,markerscale=1)
frame(ax); ax.set_title(f"UMAP of {len(U):,} cells, coloured by cell type (Nandhu_version_final)",fontsize=10.5,loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_umap_celltype.png",bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## UMAP Relapse vs Non-relapse
# %%
fig,axs=plt.subplots(1,3,figsize=(18,6.2))
cols=np.where(g[perm]==GR,C_R,C_NR); axs[0].scatter(U[perm,0],U[perm,1],s=0.4,c=cols,linewidths=0,rasterized=True)
axs[0].scatter([],[],c=C_NR,s=30,label=f"{GN} ({(g==GN).sum():,} cells, 23 ROIs)"); axs[0].scatter([],[],c=C_R,s=30,label=f"{GR} ({(g==GR).sum():,} cells, 26 ROIs)")
axs[0].legend(loc="upper left",frameon=False,fontsize=8); axs[0].set_title("Both groups (random draw order)",fontsize=10,loc="left")
for ax,gg,c in [(axs[1],GN,C_NR),(axs[2],GR,C_R)]:
    ax.scatter(U[:,0],U[:,1],s=0.3,c="#e6e5e0",linewidths=0,rasterized=True); mk=g==gg
    ax.scatter(U[mk,0],U[mk,1],s=0.4,c=c,linewidths=0,rasterized=True); ax.set_title(f"{gg} only (grey = all cells)",fontsize=10,loc="left",color=c)
for ax in axs: frame(ax)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_umap_relapse_vs_nonrelapse.png",bbox_inches="tight"); plt.close(fig)

# density per group (same bins) and the difference
H=[np.histogram2d(U[g==gg,0],U[g==gg,1],bins=150,range=[[U[:,0].min(),U[:,0].max()],[U[:,1].min(),U[:,1].max()]])[0] for gg in (GN,GR)]
H=[h/h.sum() for h in H]; Dd=(H[1]-H[0]); lim=np.percentile(np.abs(Dd[Dd!=0]),99)
fig,ax=plt.subplots(figsize=(8,7)); ext=[U[:,0].min(),U[:,0].max(),U[:,1].min(),U[:,1].max()]
im=ax.imshow(Dd.T,origin="lower",extent=ext,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto"); frame(ax)
cb=fig.colorbar(im,ax=ax,shrink=0.7); cb.set_label("Relapse − Non-relapse cell density\nred = over-represented in relapse")
ax.set_title("Where relapse and non-relapse cells differ on the UMAP",fontsize=10,loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_umap_density_difference.png",bbox_inches="tight"); plt.close(fig)

# machine (batch check)
fig,axs=plt.subplots(2,2,figsize=(12,11))
for row,(E,lbl) in enumerate([(ad.obsm["X_umap_raw"],"uncorrected PCs"),(U,"Harmony-corrected PCs (used in all other figures)")]):
    for ax,mm,c in [(axs[row,0],"machine_A","#2a78d6"),(axs[row,1],"machine_B","#eb6834")]:
        ax.scatter(E[:,0],E[:,1],s=0.3,c="#e6e5e0",linewidths=0,rasterized=True); mk=ad.obs.machine.values==mm
        ax.scatter(E[mk,0],E[mk,1],s=0.4,c=c,linewidths=0,rasterized=True); frame(ax); ax.set_title(f"{lbl}\n{mm} ({mk.sum():,} cells)",fontsize=9.5,loc="left")
fig.suptitle("Batch check: cells from each machine before and after Harmony (grey = all cells)",fontsize=10.5,x=0.01,ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_umap_machine_check.png",bbox_inches="tight"); plt.close(fig)

# marker feature plots
MK=[("Alexa647 anti rabbit1","NPM1mut (Alexa647)"),("CD34 REAL1217","CD34"),("CD235a REA175","CD235a"),("CD3 REAL1097","CD3"),
    ("CD8a REA1024","CD8a"),("CD68 REAL1346","CD68"),("CD15 VIMC6","CD15"),("HLA DR REAL550","HLA-DR"),
    ("CD11c REA1310","CD11c"),("Ki 67 REAL1047","Ki-67"),("CD31 REA1312","CD31"),("Mast Cell Tryptase REAL798","Tryptase")]
Xm=pd.DataFrame(np.asarray(ad.X),columns=list(ad.var_names))
fig,axs=plt.subplots(3,4,figsize=(16,12))
for ax,(mk,nm) in zip(axs.flat,MK):
    v=Xm[mk].values; lo,hi=np.percentile(v,[2,98]); o=np.argsort(v)
    sca=ax.scatter(U[o,0],U[o,1],s=0.3,c=np.clip(v[o],lo,hi),cmap="Blues",linewidths=0,rasterized=True); frame(ax); ax.set_title(nm,fontsize=10)
    ax.set_xlabel(""); ax.set_ylabel("")
fig.colorbar(sca,ax=axs,shrink=0.5,label="marker value (double z-norm, 2–98th pct)")
fig.suptitle("Marker expression on the UMAP (high values drawn on top)",fontsize=11,x=0.01,ha="left")
fig.savefig(f"{OUT}/fig5_umap_markers.png",bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## Cell-type composition per ROI and per group
# %%
CTS=[c for c in COL if c in set(ct)]
comp=pd.crosstab(ad.obs.roi,ad.obs.cell_type,normalize="index").reindex(columns=CTS,fill_value=0)*100
comp.insert(0,"group",comp.index.map(grp)); comp.insert(1,"n_cells",ad.obs.roi.value_counts().reindex(comp.index).values)
comp=comp.sort_values(["group","NPM1_blasts"],ascending=[True,False]); comp.round(3).to_csv(f"{OUT}/celltype_composition_per_ROI_percent.csv")
def stacked(ax,M,labels):
    bottom=np.zeros(len(M)); xs=np.arange(len(M))
    for t in CTS:
        ax.bar(xs,M[t],bottom=bottom,color=COL[t],edgecolor="white",linewidth=0.3,width=0.85,label=t); bottom+=M[t].values
    ax.set_xticks(xs); ax.set_xticklabels(labels,rotation=90,fontsize=6.5); ax.set_ylim(0,108); ax.set_ylabel("% of cells")
fig,ax=plt.subplots(figsize=(16,5.6)); stacked(ax,comp,[r.replace("Machine ","").replace("_ROI_"," R") for r in comp.index])
nnr=(comp.group==GN).sum(); ax.axvline(nnr-0.5,color="#0b0b0b",lw=1.2)
ax.text((nnr-1)/2,103,GN,ha="center",color=C_NR,fontweight="bold"); ax.text(nnr+(len(comp)-nnr-1)/2,103,GR,ha="center",color=C_R,fontweight="bold")
ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=7); ax.set_title("Cell-type composition per ROI (sorted by % NPM1 blasts within each group)",fontsize=10,loc="left",pad=14)
fig.tight_layout(); fig.savefig(f"{OUT}/fig6_composition_per_ROI.png",bbox_inches="tight"); plt.close(fig)

st=[]
for t in CTS:
    a,b=comp.loc[comp.group==GN,t],comp.loc[comp.group==GR,t]
    st.append({"cell_type":t,"nonrelapse_mean_%":a.mean(),"relapse_mean_%":b.mean(),"nonrelapse_median_%":a.median(),"relapse_median_%":b.median(),"p":mannwhitneyu(a,b,alternative="two-sided")[1]})
st=pd.DataFrame(st); st["p_adj"]=bh(st.p); st["sig"]=st.p_adj<0.05; st.to_csv(f"{OUT}/celltype_composition_group_stats.csv",index=False)
print(st.round(4).to_string(index=False))
fig,axs=plt.subplots(4,5,figsize=(17,12.5))
for ax,t in zip(axs.flat,CTS):
    data=[comp.loc[comp.group==gg,t].values for gg in (GN,GR)]
    bp=ax.boxplot(data,widths=0.55,patch_artist=True,showfliers=False,medianprops=dict(color="#0b0b0b"))
    for patch,c in zip(bp["boxes"],[C_NR,C_R]): patch.set_facecolor(c); patch.set_alpha(0.3); patch.set_edgecolor(c)
    for k,(d,c) in enumerate(zip(data,[C_NR,C_R])): ax.scatter(np.random.default_rng(k).normal(k+1,0.06,len(d)),d,s=10,color=c,zorder=3,linewidths=0)
    r=st[st.cell_type==t].iloc[0]; ax.set_title(f"{t.replace('Unassigned-Leiden: ','Leiden: ')}\nBH p={r.p_adj:.3f} {star(r.p_adj)}",fontsize=8)
    ax.set_xticks([1,2]); ax.set_xticklabels([GN,GR],fontsize=7.5); ax.set_ylabel("% of ROI cells",fontsize=7)
for ax in list(axs.flat)[len(CTS):]: ax.axis("off")
fig.suptitle("Cell-type abundance per ROI, Non-relapse vs Relapse (Mann–Whitney, BH across cell types)",fontsize=11,x=0.01,ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig7_composition_group_boxplots.png",bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## Per patient: `../roi_patient_mapping.csv` (the ROI → patient table supplied on 2026-11-11)
# 11 patients: 6 non-relapse (NR-P1…P6, 23 ROIs) and 5 relapse (R-P1…P5, 26 ROIs). Patient composition = all cells of the
# patient's ROIs pooled. The patient-level test uses one value per patient (the mean of its ROI percentages), 6 vs 5.
# %%
mp=pd.read_csv(f"{BASE}/roi_patient_mapping.csv")
assert set(mp.roi)==set(comp.index) and all(mp.group==mp.roi.map(grp)), "mapping does not match ROI groups"
ad.obs["patient"]=ad.obs.roi.map(dict(zip(mp.roi,mp.patient))).values
PORDER=sorted(mp.patient.unique(),key=lambda x:(x.startswith("R"),int(x.split("P")[1])))
pc=pd.crosstab(ad.obs.patient,ad.obs.cell_type,normalize="index").reindex(index=PORDER,columns=CTS,fill_value=0)*100
info=mp.groupby("patient").agg(group=("group","first"),stratification=("stratification","first"),acquisition=("acquisition","first"),n_ROIs=("roi","size"))
pc=info.reindex(PORDER).join(ad.obs.patient.value_counts().rename("n_cells")).join(pc); pc.round(3).to_csv(f"{OUT}/celltype_composition_per_patient_percent.csv")
fig,ax=plt.subplots(figsize=(13,6)); stacked(ax,pc,[f"{p}\n{int(r.n_ROIs)} ROIs\n{int(r.n_cells):,} cells" for p,r in pc.iterrows()])
ax.tick_params(axis="x",rotation=0,labelsize=7.5); nnr=(pc.group==GN).sum(); ax.axvline(nnr-0.5,color="#0b0b0b",lw=1.2)
ax.text((nnr-1)/2,103,GN,ha="center",color=C_NR,fontweight="bold"); ax.text(nnr+(len(pc)-nnr-1)/2,103,GR,ha="center",color=C_R,fontweight="bold")
ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=7)
ax.set_title("Cell-type composition per patient (all cells of the patient's ROIs pooled)",fontsize=10,loc="left",pad=14)
fig.tight_layout(); fig.savefig(f"{OUT}/fig8_composition_per_patient.png",bbox_inches="tight"); plt.close(fig)

# patient-level statistics: one value per patient = mean of its ROI percentages
rp=comp[CTS].join(mp.set_index("roi").patient); pm=rp.groupby("patient")[CTS].mean().reindex(PORDER); pm["group"]=info.reindex(PORDER).group
pst=[]
for t in CTS:
    a,b=pm.loc[pm.group==GN,t],pm.loc[pm.group==GR,t]
    pst.append({"cell_type":t,"nonrelapse_mean_%":a.mean(),"relapse_mean_%":b.mean(),"n_NR_patients":len(a),"n_R_patients":len(b),"p":mannwhitneyu(a,b,alternative="two-sided")[1]})
pst=pd.DataFrame(pst); pst["p_adj"]=bh(pst.p); pst["sig"]=pst.p_adj<0.05; pst.to_csv(f"{OUT}/celltype_composition_patient_level_stats.csv",index=False)
print(pst.round(4).to_string(index=False))
fig,axs=plt.subplots(4,5,figsize=(17,12.5))
for ax,t in zip(axs.flat,CTS):
    for k,(gg,c) in enumerate([(GN,C_NR),(GR,C_R)]):
        sub=pm[pm.group==gg]; ax.scatter(np.full(len(sub),k+1)+np.linspace(-0.12,0.12,len(sub)),sub[t],s=28,color=c,zorder=3)
        ax.hlines(sub[t].median(),k+0.75,k+1.25,color="#0b0b0b",lw=1.5)
        for x,(pid,v) in zip(np.full(len(sub),k+1)+np.linspace(-0.12,0.12,len(sub)),sub[t].items()): ax.text(x+0.05,v,pid.split("-")[1],fontsize=5.5,color="#52514e",va="center")
    r=pst[pst.cell_type==t].iloc[0]; ax.set_title(f"{t.replace('Unassigned-Leiden: ','Leiden: ')}\np={r.p:.3f}, BH={r.p_adj:.3f} {star(r.p_adj)}",fontsize=8)
    ax.set_xticks([1,2]); ax.set_xticklabels([f"{GN}\n(6 pts)",f"{GR}\n(5 pts)"],fontsize=7.5); ax.set_xlim(0.5,2.5); ax.set_ylabel("% of cells",fontsize=7)
for ax in list(axs.flat)[len(CTS):]: ax.axis("off")
fig.suptitle("Patient-level cell-type abundance (one dot per patient = mean of its ROIs; line = median; Mann–Whitney 6 vs 5, BH)",fontsize=11,x=0.01,ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig9_composition_patient_level.png",bbox_inches="tight"); plt.close(fig)

# UMAP per patient (small multiples)
fig,axs=plt.subplots(3,4,figsize=(16,12))
for ax,pid in zip(axs.flat,PORDER):
    mk=ad.obs.patient.values==pid; ax.scatter(U[:,0],U[:,1],s=0.2,c="#e6e5e0",linewidths=0,rasterized=True)
    ax.scatter(U[mk,0],U[mk,1],s=0.4,c=[COL[x] for x in ct[mk]],linewidths=0,rasterized=True); frame(ax); ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_title(f"{pid} ({mk.sum():,} cells)",fontsize=9.5,color=C_NR if pid.startswith("NR") else C_R,loc="left")
axs.flat[-1].axis("off")
for t in CTS: axs.flat[-1].scatter([],[],c=COL[t],s=25,label=t)
axs.flat[-1].legend(loc="center",frameon=False,fontsize=7)
fig.suptitle("UMAP per patient, coloured by cell type (grey = all cells)",fontsize=11,x=0.01,ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig10_umap_per_patient.png",bbox_inches="tight"); plt.close(fig)
print("__DONE__")
