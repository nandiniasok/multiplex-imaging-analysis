# %% [markdown]
# # Spatial proximity + ligand–receptor analysis — Nandhu_version_final
# Cell types = `Nandhu_version_final/csv/cell_assignments.csv` (`final_label`). **Tumour = NPM1_blasts** (43,493 cells).
# Groups: Non-relapse = the 23 ROIs used in every earlier analysis; Relapse = the other 26. One ROI = one replicate.
#
# * **PART 1, proximity screen.** For each cell type: % of its cells within 32.5 µm (100 px) of the nearest tumour cell.
#   This is the same single-threshold screen as the June report.
# * **PART 2, distance zones.** The non-tumour composition in 5 zones (<15 touching, 15–30 immediate, 30–60 local,
#   60–100 intermediate, 100+ far), as in the June report.
# * **PART 3, ligand–receptor (July 3 method).** Tumour ligand → immune receptor:
#   PD-L1→PD-1, Galectin-9→TIM-3, Galectin-9→CD44, HLA-DR→LAG-3. For every tumour–immune pair < 100 µm, binned into
#   Contact <15 / Near 15–30 / Mid 30–50 / Far 50–100 µm: SIS = mean(max(z_lig,0)·max(z_rec,0)).
#   This is scored for all immune receivers and again with the receiver restricted to T cells.
# * **PART 4, immune pathway modules vs distance (July 3 part B).** 12 modules = mean per-machine z of their member markers.
#   Each immune cell takes the zone of its nearest tumour cell. Tested per zone, and as a gradient (Contact − Far).
# * **PART 5, lineage-resolved modules (July 3 part C).** T cells / Myeloid / Plasma cells separately.
#
# Statistics: two-sided Mann–Whitney U on per-ROI values, with Benjamini–Hochberg correction inside each family
# (within zone, or within lineage × zone). Unassigned/Other and Artifact are left out of every composition: they are
# not cell types. Distances are nucleus centre to centre at 0.325 µm/px.

# %%
import os, warnings
import numpy as np, pandas as pd, scanpy as sc
from scipy.spatial import cKDTree
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
REC=os.environ.get("MRD_REC","/mnt/e/MRD analysis Apr_2026/30 May 2026_Claude new results/new 18.06.26")
PROJ=os.environ.get("MRD_PROJ",f"{REC}/proj_recovered")
ADATA=f"{PROJ}/experiments/exp-04-03-2026/stage6_subclustering/final_adata.h5ad"
BASE="/mnt/e/multiplex-imaging-analysis/data/output"
FINAL=f"{BASE}/Nandhu_version_final/csv/cell_assignments.csv"
OUT=os.environ.get("MRD_OUT",f"{BASE}/Spatial_proximity_analysis_final")
os.makedirs(f"{OUT}/figures",exist_ok=True); os.makedirs(f"{OUT}/csv",exist_ok=True)
UM=0.325; KEY=["unique_region","Cell Id"]
C_NR,C_R="#2a78d6","#e34948"; GN,GR="Non-relapse","Relapse"
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"axes.edgecolor":"#52514e",
                     "axes.labelcolor":"#0b0b0b","xtick.color":"#52514e","ytick.color":"#52514e","savefig.dpi":200,
                     "axes.grid":True,"grid.color":"#e6e5e0","grid.linewidth":0.6,"axes.axisbelow":True})
NONREL={"Machine A_Slides1_ROI_12","Machine A_Slides1_ROI_13","Machine A_Slides1_ROI_14","Machine A_Slides1_ROI_15",
 "Machine A_Slides1_ROI_16","Machine A_Slides2_ROI_10","Machine A_Slides2_ROI_11","Machine A_Slides2_ROI_12",
 "Machine A_Slides2_ROI_13","Machine A_Slides2_ROI_14","Machine A_Slides2_ROI_15","Machine A_Slides2_ROI_16",
 "Machine B_SN177_ROI_06","Machine B_SN177_ROI_07","Machine B_SN177_ROI_08","Machine B_SN177_ROI_09",
 "Machine B_SN177_ROI_10","Machine B_SN177_ROI_11","Machine B_SN177_ROI_12","Machine B_SN177_ROI_13",
 "Machine B_SN177_ROI_14","Machine B_SN177_ROI_15","Machine B_SN177_ROI_16"}
def grp(r): return GN if r in NONREL else GR
def star(p): return "" if pd.isna(p) else ("***" if p<0.001 else "**" if p<0.01 else "*" if p<0.05 else "")
def bh(p):
    p=np.asarray(p,float); n=len(p); o=np.argsort(p); r=np.empty(n); r[o]=np.arange(1,n+1)
    q=p*n/r; qs=np.minimum.accumulate(q[o][::-1])[::-1]; out=np.empty(n); out[o]=np.clip(qs,0,1); return out
def mw(a,b):
    a,b=np.asarray(a,float),np.asarray(b,float)
    return mannwhitneyu(a,b,alternative="two-sided")[1] if len(a)>=4 and len(b)>=4 else np.nan

TUMOUR="NPM1_blasts"; DROP={"Unassigned/Other","Artifact"}
TCELL=["CD4+ T","CD8+ T"]; BCELL=["Plasma cells"]
MYELOID=["Granulocyte","Macrophages","Monocytes","Mast cells","HLA-DR-hi APC (CD1c+)","Immature myeloid CD11c+"]
LINEAGE={"T cells":TCELL,"Myeloid":MYELOID,"Plasma cells":BCELL}; IMMUNE=TCELL+BCELL+MYELOID

m=pd.read_csv(FINAL,dtype={"Cell Id":str}); m["roi"]=m.unique_region.str.replace("/","_",regex=False); m["group"]=m.roi.map(grp)
m["is_tumor"]=m.final_label.eq(TUMOUR)
print(f"cells {len(m):,} | tumour {int(m.is_tumor.sum()):,} | ROIs {m.roi.nunique()} "
      f"({m[m.group==GN].roi.nunique()} non-relapse, {m[m.group==GR].roi.nunique()} relapse)")

# %% [markdown]
# ## PART 1 — proximity screen (% of each cell type within 32.5 µm of tumour)
# %%
NEAR=100*UM
rows=[]
for roi,df in m.groupby("roi"):
    tum=df[df.is_tumor]; non=df[~df.is_tumor & ~df.final_label.isin(DROP)]
    if len(tum)==0 or len(non)==0: continue
    d,_=cKDTree(tum[["Nuc X","Nuc Y Inv"]].values).query(non[["Nuc X","Nuc Y Inv"]].values,k=1)
    non=non.assign(um=d*UM)
    for ct,g in non.groupby("final_label"):
        if len(g)>=5: rows.append({"roi":roi,"group":grp(roi),"cell_type":ct,"n":len(g),"pct_near":100*(g.um<NEAR).mean(),"median_dist_um":g.um.median()})
near=pd.DataFrame(rows); near.to_csv(f"{OUT}/csv/P1_pct_near_tumour_per_roi.csv",index=False)
st=[]
for ct,s in near.groupby("cell_type"):
    a,b=s[s.group==GN].pct_near,s[s.group==GR].pct_near
    st.append({"cell_type":ct,"mean_nonrelapse_%":a.mean(),"sem_nonrelapse":a.sem(),"mean_relapse_%":b.mean(),"sem_relapse":b.sem(),
               "diff_NR_minus_R":a.mean()-b.mean(),"n_NR":len(a),"n_R":len(b),"p":mw(a,b)})
p1=pd.DataFrame(st).dropna(subset=["p"]); p1["p_adj"]=bh(p1.p); p1["sig"]=p1.p_adj<0.05
p1=p1.sort_values("diff_NR_minus_R"); p1.to_csv(f"{OUT}/csv/P1_pct_near_tumour_stats.csv",index=False)
print(p1[["cell_type","mean_nonrelapse_%","mean_relapse_%","p","p_adj"]].round(4).to_string(index=False))

fig,ax=plt.subplots(figsize=(8,0.42*len(p1)+1.6)); y=np.arange(len(p1)); h=0.38
ax.barh(y+h/2,p1["mean_nonrelapse_%"],h,xerr=p1.sem_nonrelapse,color=C_NR,label=GN,error_kw=dict(lw=0.8,ecolor="#52514e"))
ax.barh(y-h/2,p1["mean_relapse_%"],h,xerr=p1.sem_relapse,color=C_R,label=GR,error_kw=dict(lw=0.8,ecolor="#52514e"))
for i,(_,r) in enumerate(p1.iterrows()):
    s=star(r.p_adj)
    if s: ax.text(max(r["mean_nonrelapse_%"]+r.sem_nonrelapse,r["mean_relapse_%"]+r.sem_relapse)+1.5,i,s,va="center",fontsize=10,fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(p1.cell_type); ax.set_xlabel("% of the cell type within 32.5 µm of an NPM1 blast (mean ± SEM over ROIs)")
ax.set_title("Proximity to NPM1 blasts — Non-relapse vs Relapse\n* BH-adjusted p<0.05 (Mann–Whitney, ROIs as replicates)",fontsize=10,loc="left")
ax.legend(loc="lower right",frameon=False); ax.grid(axis="y",visible=False); fig.tight_layout()
fig.savefig(f"{OUT}/figures/fig1_pct_near_tumour.png"); plt.close(fig)

# %% [markdown]
# ## PART 2 — distance-zone composition (5 zones)
# %%
Z5=[(0,15,"<15 µm touching"),(15,30,"15–30 µm immediate"),(30,60,"30–60 µm local"),(60,100,"60–100 µm intermediate"),(100,np.inf,"100+ µm far")]
Z5L=[z[2] for z in Z5]; ZE5=np.array([0,15,30,60,100])
rows=[]
for roi,df in m.groupby("roi"):
    tum=df[df.is_tumor]; non=df[~df.is_tumor & ~df.final_label.isin(DROP)]
    if len(tum)==0 or len(non)<10: continue
    d,_=cKDTree(tum[["Nuc X","Nuc Y Inv"]].values).query(non[["Nuc X","Nuc Y Inv"]].values,k=1)
    zi=np.clip(np.digitize(d*UM,ZE5[1:]),0,4)
    t=pd.DataFrame({"ct":non.final_label.values,"zone":np.array(Z5L)[zi]})
    for (ct,zn),g in t.groupby(["ct","zone"]): rows.append({"roi":roi,"group":grp(roi),"cell_type":ct,"zone":zn,"n":len(g)})
det=pd.DataFrame(rows)
# complete the grid with zeros: a cell type absent from a zone that has cells is a true 0 %
zt=det.groupby(["roi","zone"]).n.sum().rename("zone_total").reset_index()
cts=sorted(det.cell_type.unique())
grid=zt.merge(pd.DataFrame({"cell_type":cts}),how="cross")
det=grid.merge(det[["roi","zone","cell_type","n"]],on=["roi","zone","cell_type"],how="left").fillna({"n":0})
det["group"]=det.roi.map(grp); det["pct"]=100*det.n/det.zone_total
det=det[det.zone_total>=10]; det.to_csv(f"{OUT}/csv/P2_zone_composition_per_roi.csv",index=False)
out=[]
for zn in Z5L:
    s=det[det.zone==zn]; rr=[]
    for ct,c in s.groupby("cell_type"):
        a,b=c[c.group==GN].pct,c[c.group==GR].pct
        rr.append({"zone":zn,"cell_type":ct,"mean_nonrelapse_%":a.mean(),"mean_relapse_%":b.mean(),"diff_NR_minus_R":a.mean()-b.mean(),
                   "n_NR":len(a),"n_R":len(b),"p":mw(a,b)})
    rr=pd.DataFrame(rr).dropna(subset=["p"]); rr["p_adj"]=bh(rr.p); out.append(rr)
p2=pd.concat(out); p2["sig"]=p2.p_adj<0.05; p2.to_csv(f"{OUT}/csv/P2_zone_composition_stats.csv",index=False)
print("\nPART 2 — significant (BH<0.05 within zone):")
print(p2[p2.sig][["zone","cell_type","mean_nonrelapse_%","mean_relapse_%","p_adj"]].round(4).to_string(index=False))

# fig2: difference heatmap (cell type × zone), stars = BH<0.05
D=p2.pivot(index="cell_type",columns="zone",values="diff_NR_minus_R")[Z5L]; Q=p2.pivot(index="cell_type",columns="zone",values="p_adj")[Z5L]
D=D.loc[D.abs().max(axis=1).sort_values(ascending=False).index]; Q=Q.loc[D.index]
fig,ax=plt.subplots(figsize=(7.8,0.36*len(D)+2)); lim=np.nanmax(np.abs(D.values))
im=ax.imshow(D.values,cmap="RdBu",vmin=-lim,vmax=lim,aspect="auto")  # blue = higher in non-relapse
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        v=D.values[i,j]
        if not np.isnan(v): ax.text(j,i,f"{v:+.1f}{star(Q.values[i,j])}",ha="center",va="center",fontsize=7,color="white" if abs(v)>0.6*lim else "#0b0b0b")
ax.set_xticks(range(5)); ax.set_xticklabels(["<15 µm\ntouching","15–30 µm\nimmediate","30–60 µm\nlocal","60–100 µm\nintermediate","100+ µm\nfar"],fontsize=8); ax.set_yticks(range(len(D))); ax.set_yticklabels(D.index)
ax.grid(False); cb=fig.colorbar(im,ax=ax,shrink=0.6); cb.set_label("Non-relapse − Relapse (% points)")
ax.set_title("Composition around NPM1 blasts by distance zone\nNon-relapse − Relapse (% of non-tumour cells)\nblue = higher in non-relapse; * BH<0.05 within zone",fontsize=10,loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig2_zone_difference_heatmap.png"); plt.close(fig)

# fig3: zone line plots for every cell type significant in ≥1 zone
sigct=sorted(p2[p2.sig].cell_type.unique()) or list(D.index[:4])
nc=min(3,len(sigct)); nr=int(np.ceil(len(sigct)/nc))
fig,axs=plt.subplots(nr,nc,figsize=(4.2*nc,3.1*nr),squeeze=False)
for ax,ct in zip(axs.flat,sigct):
    for g,c in [(GN,C_NR),(GR,C_R)]:
        s=det[(det.cell_type==ct)&(det.group==g)].groupby("zone").pct.agg(["mean","sem"]).reindex(Z5L)
        ax.errorbar(range(5),s["mean"],yerr=s["sem"],color=c,lw=2,marker="o",ms=5,capsize=2,label=g)
    for j,zn in enumerate(Z5L):
        q=p2[(p2.zone==zn)&(p2.cell_type==ct)].p_adj
        if len(q) and star(q.iloc[0]): ax.text(j,ax.get_ylim()[1]*0.95,star(q.iloc[0]),ha="center",fontsize=11,fontweight="bold")
    ax.set_xticks(range(5)); ax.set_xticklabels(["<15","15–30","30–60","60–100","100+"]); ax.set_title(ct,fontsize=9.5)
    ax.set_xlabel("distance to nearest NPM1 blast (µm)",fontsize=8); ax.set_ylabel("% of non-tumour cells",fontsize=8)
for ax in list(axs.flat)[len(sigct):]: ax.axis("off")
axs.flat[0].legend(frameon=False,fontsize=8); fig.suptitle("Cell types that differ in at least one zone (mean ± SEM; * BH<0.05)",fontsize=10,x=0.01,ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig3_zone_lineplots_significant.png"); plt.close(fig)

# fig4: stacked composition per zone (mean over ROIs), non-relapse vs relapse
COL={"CD4+ T":"#1c5cab","CD8+ T":"#3987e5","Plasma cells":"#86b6ef","Granulocyte":"#eb6834","Macrophages":"#e34948",
     "Monocytes":"#e87ba4","Mast cells":"#eda100","HLA-DR-hi APC (CD1c+)":"#6250d6","Immature myeloid CD11c+":"#9085e9",
     "Erythroid":"#8c2b2b","NPM1_blasts":"#1a1a19","Mixed progenitors":"#a8a79f","Endothelial":"#1baf7a","MSC":"#008300",
     "Megakaryocyte":"#8a5a00","Unassigned-Leiden: CD68+CD3+Tryptase+":"#c3c2b7","Unassigned-Leiden: Ki67+CD147+CD45RA+":"#77756e",
     "Unassigned/Other":"#e0dfda","Artifact":"#f0efec"}
ORDER=[c for c in COL if c in cts]
mean=det.groupby(["group","zone","cell_type"]).pct.mean().unstack("cell_type").reindex(columns=ORDER).fillna(0)
fig,axs=plt.subplots(1,2,figsize=(12,5.2),sharey=True)
for ax,g in zip(axs,[GN,GR]):
    M=mean.loc[g].reindex(Z5L); M=M.div(M.sum(axis=1),axis=0)*100; bottom=np.zeros(5)
    for ct in ORDER:
        ax.bar(range(5),M[ct],bottom=bottom,color=COL[ct],edgecolor="white",linewidth=0.8,width=0.75,label=ct); bottom+=M[ct].values
    ax.set_xticks(range(5)); ax.set_xticklabels(["<15","15–30","30–60","60–100","100+"]); ax.set_xlabel("distance to nearest NPM1 blast (µm)")
    ax.set_title(g,fontsize=10,color=C_NR if g==GN else C_R,fontweight="bold"); ax.grid(axis="x",visible=False)
axs[0].set_ylabel("% of non-tumour cells (mean over ROIs)")
axs[1].legend(loc="center left",bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=7.5)
fig.suptitle("Non-tumour composition by distance from NPM1 blasts",fontsize=10.5,x=0.01,ha="left"); fig.tight_layout()
fig.savefig(f"{OUT}/figures/fig4_zone_stacked_composition.png",bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## Expression for parts 3–5 (per-machine z, as in July 3 / Aug 9)
# %%
MODULES={"T-cell exhaustion":["CD279 PD1 3 1 3","TIM 3 REAL818","CD223 REA351","BATF REA486","CD305 REA447"],
 "PI3K-AKT survival":["AKT Pan REA676","PI3K p85 alpha REAL589"],"MAPK-JNK stress":["JNK1 REAL1128","JNK2 REA1153"],
 "FAK adhesion/migration":["FAK pS910 REA407","CD44 REA690"],"TCR proximal signalling":["ZAP70 REA814","CD247 REAL1135"],
 "Proliferation":["Ki 67 REAL1047","PCNA REA858","RRM2 REAL1010","c myc REAL810"],"Wnt / beta-catenin":["beta Catenin REA480"],
 "p53 stress":["p53 REA1132"],"Type-I IFN / inflammation":["IRF 7 REA521","IFN gamma REAL788"],
 "Anti-apoptotic (Bcl-2)":["Bcl 2 REA872"],"Antigen presentation (HLA-DR)":["HLA DR REAL550"],"Epigenetic (EZH2)":["EZH2 REA907"]}
LR_PAIRS=[("PD-L1 → PD-1","CD274 REA1308","CD279 PD1 3 1 3"),("Galectin-9 → TIM-3","Galectin 9 REA435","TIM 3 REAL818"),
          ("Galectin-9 → CD44","Galectin 9 REA435","CD44 REA690"),("HLA-DR → LAG-3","HLA DR REAL550","CD223 REA351")]
ad=sc.read_h5ad(ADATA); o=ad.obs.copy(); o["Cell Id"]=o["Cell Id"].astype(str)
mk_all=sorted({x for v in MODULES.values() for x in v}|{p[1] for p in LR_PAIRS}|{p[2] for p in LR_PAIRS})
miss=[mk for mk in mk_all if mk not in ad.var_names and mk not in o.columns]; assert not miss, miss
E=pd.DataFrame({mk:(np.asarray(ad[:,mk].X).ravel() if mk in ad.var_names else pd.to_numeric(o[mk],errors="coerce").values) for mk in mk_all})
E["unique_region"]=o.unique_region.values; E["Cell Id"]=o["Cell Id"].values
full=m.merge(E,on=KEY,how="left")
for mk in mk_all: full[f"z_{mk}"]=full.groupby("machine")[mk].transform(lambda s:(s-s.mean())/(s.std()+1e-9))
for mod,mks in MODULES.items(): full[f"M_{mod}"]=full[[f"z_{x}" for x in mks]].mean(axis=1)
ZE=np.array([0,15,30,50,100.]); Z4=["Contact <15 µm","Near 15–30 µm","Mid 30–50 µm","Far 50–100 µm"]; Z4S=["Contact","Near","Mid","Far"]

# %% [markdown]
# ## PART 3 — ligand–receptor spatial interaction score (SIS)
# %%
lr=[]
for roi,df in full.groupby("roi"):
    tum=df[df.is_tumor]
    for rname,labs in [("all immune",IMMUNE),("T cells",TCELL)]:
        ic=df[df.final_label.isin(labs)]
        if len(tum)<5 or len(ic)<5: continue
        sdm=cKDTree(ic[["Nuc X","Nuc Y Inv"]].values).sparse_distance_matrix(cKDTree(tum[["Nuc X","Nuc Y Inv"]].values),max_distance=ZE[-1]/UM,output_type="coo_matrix")
        if sdm.nnz==0: continue
        dum=sdm.data*UM; ok=dum<ZE[-1]; ii,ti,zb=sdm.row[ok],sdm.col[ok],np.digitize(dum[ok],ZE[1:-1])
        for label,lig,rec in LR_PAIRS:
            L=tum[f"z_{lig}"].to_numpy()[ti]; R=ic[f"z_{rec}"].to_numpy()[ii]
            sis=np.clip(L,0,None)*np.clip(R,0,None); dp=(L>0.5)&(R>0.5)
            for zi in range(4):
                sel=zb==zi
                if sel.sum()>=10: lr.append({"roi":roi,"group":grp(roi),"receiver":rname,"pair":label,"zone":Z4[zi],"zone_idx":zi,
                                            "n_pairs":int(sel.sum()),"sis":float(sis[sel].mean()),"dp_frac":float(dp[sel].mean())})
lrdf=pd.DataFrame(lr); lrdf.to_csv(f"{OUT}/csv/P3_LR_SIS_per_roi.csv",index=False)
out=[]
for (rname,zn),s in lrdf.groupby(["receiver","zone"]):
    rr=[]
    for pair,c in s.groupby("pair"):
        a,b=c[c.group==GN].sis,c[c.group==GR].sis
        rr.append({"receiver":rname,"zone":zn,"pair":pair,"mean_nonrelapse":a.mean(),"mean_relapse":b.mean(),"diff_R_minus_NR":b.mean()-a.mean(),
                   "n_NR":len(a),"n_R":len(b),"p":mw(a,b)})
    rr=pd.DataFrame(rr).dropna(subset=["p"]); rr["p_adj"]=bh(rr.p); out.append(rr)
p3=pd.concat(out); p3["sig"]=p3.p_adj<0.05; p3["zone_idx"]=p3.zone.map({z:i for i,z in enumerate(Z4)})
p3=p3.sort_values(["receiver","pair","zone_idx"]); p3.to_csv(f"{OUT}/csv/P3_LR_SIS_stats.csv",index=False)
print("\nPART 3 — LR SIS (BH within receiver × zone):"); print(p3[["receiver","pair","zone","mean_nonrelapse","mean_relapse","p","p_adj"]].round(4).to_string(index=False))

for rname,fname in [("all immune","fig5_LR_SIS_vs_distance_all_immune.png"),("T cells","fig6_LR_SIS_vs_distance_Tcell_receiver.png")]:
    fig,axs=plt.subplots(1,4,figsize=(15,3.6))
    for ax,(label,_,_) in zip(axs,LR_PAIRS):
        s=lrdf[(lrdf.receiver==rname)&(lrdf.pair==label)]
        for g,c in [(GN,C_NR),(GR,C_R)]:
            q=s[s.group==g].groupby("zone_idx").sis.agg(["mean","sem"]).reindex(range(4))
            ax.errorbar(range(4),q["mean"],yerr=q["sem"],color=c,lw=2,marker="o",ms=5,capsize=2,label=g)
        for zi in range(4):
            r=p3[(p3.receiver==rname)&(p3.pair==label)&(p3.zone_idx==zi)]
            if len(r): ax.text(zi,ax.get_ylim()[1]*0.97,star(r.p_adj.iloc[0]) or ("†" if r.p.iloc[0]<0.05 else ""),ha="center",fontsize=11,fontweight="bold")
        ax.set_xticks(range(4)); ax.set_xticklabels(Z4S); ax.set_title(label,fontsize=9.5); ax.set_ylabel("SIS (mean ± SEM)",fontsize=8)
    axs[0].legend(frameon=False,fontsize=8)
    fig.suptitle(f"Tumour ligand → {rname} receptor interaction score by distance  (* BH<0.05, † raw p<0.05)",fontsize=10,x=0.01,ha="left")
    fig.tight_layout(); fig.savefig(f"{OUT}/figures/{fname}"); plt.close(fig)

# fig7: contact-zone boxplots (all immune + T-cell receiver)
fig,axs=plt.subplots(2,4,figsize=(14,6.4))
for row,rname in enumerate(["all immune","T cells"]):
    for ax,(label,_,_) in zip(axs[row],LR_PAIRS):
        s=lrdf[(lrdf.receiver==rname)&(lrdf.pair==label)&(lrdf.zone_idx==0)]
        data=[s[s.group==g].sis.values for g in (GN,GR)]
        bp=ax.boxplot(data,widths=0.55,patch_artist=True,showfliers=False,medianprops=dict(color="#0b0b0b"))
        for patch,c in zip(bp["boxes"],[C_NR,C_R]): patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
        for k,(d,c) in enumerate(zip(data,[C_NR,C_R])):
            ax.scatter(np.random.default_rng(0).normal(k+1,0.06,len(d)),d,s=14,color=c,zorder=3,linewidths=0)
        r=p3[(p3.receiver==rname)&(p3.pair==label)&(p3.zone_idx==0)]
        pv=f"p={r.p.iloc[0]:.3f}, BH={r.p_adj.iloc[0]:.3f}" if len(r) else ""
        ax.set_xticks([1,2]); ax.set_xticklabels([GN,GR]); ax.set_title(f"{label} — {rname}\n{pv}",fontsize=8.5); ax.grid(axis="x",visible=False)
fig.suptitle("Contact zone (<15 µm): ligand–receptor SIS per ROI",fontsize=10,x=0.01,ha="left"); fig.tight_layout()
fig.savefig(f"{OUT}/figures/fig7_LR_contact_boxplots.png"); plt.close(fig)

# %% [markdown]
# ## PART 4 — immune pathway modules vs distance to tumour
# %%
def zone_scores(cells_by_roi,minn):
    rows=[]
    for roi,(tum,ic) in cells_by_roi.items():
        if len(tum)<1 or len(ic)<minn: continue
        d,_=cKDTree(tum[["Nuc X","Nuc Y Inv"]].values).query(ic[["Nuc X","Nuc Y Inv"]].values,k=1)
        ic=ic.assign(um=d*UM); ic=ic[ic.um<ZE[-1]]; ic["zi"]=np.clip(np.digitize(ic.um,ZE[1:-1]),0,3)
        for mod in MODULES:
            for zi in range(4):
                c=ic[ic.zi==zi]
                if len(c)>=minn: rows.append({"roi":roi,"group":grp(roi),"module":mod,"zone":Z4[zi],"zone_idx":zi,"score":c[f"M_{mod}"].mean(),"n":len(c)})
    return pd.DataFrame(rows)
def zone_stats(per,by):
    out=[]
    for key,s in per.groupby(by):
        rr=[]
        for mod,c in s.groupby("module"):
            a,b=c[c.group==GN].score,c[c.group==GR].score
            rr.append(dict(zip(by,key if isinstance(key,tuple) else (key,)))|{"module":mod,"mean_nonrelapse":a.mean(),"mean_relapse":b.mean(),
                      "diff_R_minus_NR":b.mean()-a.mean(),"n_NR":len(a),"n_R":len(b),"p":mw(a,b)})
        rr=pd.DataFrame(rr).dropna(subset=["p"]); rr["p_adj"]=bh(rr.p); out.append(rr)
    r=pd.concat(out); r["sig"]=r.p_adj<0.05; return r
byroi={roi:(df[df.is_tumor],df[df.final_label.isin(IMMUNE)]) for roi,df in full.groupby("roi")}
p4=zone_scores(byroi,15); p4.to_csv(f"{OUT}/csv/P4_pathway_per_roi_zone.csv",index=False)
p4s=zone_stats(p4,["zone"]); p4s["zone_idx"]=p4s.zone.map({z:i for i,z in enumerate(Z4)}); p4s.to_csv(f"{OUT}/csv/P4_pathway_stats_per_zone.csv",index=False)
gr=p4.pivot_table(index=["roi","group","module"],columns="zone_idx",values="score").reset_index()
gr=gr.dropna(subset=[0,3]); gr["gradient"]=gr[0]-gr[3]; gr[["roi","group","module","gradient"]].to_csv(f"{OUT}/csv/P4_gradient_per_roi.csv",index=False)
gs=[]
for mod,s in gr.groupby("module"):
    a,b=s[s.group==GN].gradient,s[s.group==GR].gradient
    gs.append({"module":mod,"gradient_nonrelapse":a.mean(),"sem_NR":a.sem(),"gradient_relapse":b.mean(),"sem_R":b.sem(),"n_NR":len(a),"n_R":len(b),"p":mw(a,b)})
gs=pd.DataFrame(gs); gs["p_adj"]=bh(gs.p); gs["sig"]=gs.p_adj<0.05; gs.to_csv(f"{OUT}/csv/P4_gradient_stats.csv",index=False)
print("\nPART 4 — per zone (BH within zone), p<0.05:"); print(p4s[p4s.p<0.05][["zone","module","mean_nonrelapse","mean_relapse","p","p_adj"]].round(4).to_string(index=False))
print("PART 4 — gradient:"); print(gs.sort_values("p")[["module","gradient_nonrelapse","gradient_relapse","p","p_adj"]].round(4).to_string(index=False))

def diff_heatmap(st,title,fname,rows_order=None):
    Dm=st.pivot(index="module",columns="zone",values="diff_R_minus_NR")[Z4]; Qm=st.pivot(index="module",columns="zone",values="p_adj")[Z4]
    if rows_order is not None: Dm=Dm.reindex(rows_order); Qm=Qm.reindex(rows_order)
    lim=max(0.3,np.nanmax(np.abs(Dm.values)))
    fig,ax=plt.subplots(figsize=(7,0.42*len(Dm)+1.8))
    im=ax.imshow(Dm.values,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto")
    for i in range(Dm.shape[0]):
        for j in range(4):
            v=Dm.values[i,j]
            if not np.isnan(v): ax.text(j,i,f"{v:+.2f}{star(Qm.values[i,j])}",ha="center",va="center",fontsize=7.5,color="white" if abs(v)>0.6*lim else "#0b0b0b")
    ax.set_xticks(range(4)); ax.set_xticklabels([z.replace(" ","\n",1) for z in Z4]); ax.set_yticks(range(len(Dm))); ax.set_yticklabels(Dm.index); ax.grid(False)
    cb=fig.colorbar(im,ax=ax,shrink=0.7); cb.set_label("Relapse − Non-relapse (mean z)\nred = higher in relapse")
    ax.set_title(title,fontsize=10,loc="left"); fig.tight_layout(); fig.savefig(f"{OUT}/figures/{fname}"); plt.close(fig)
MOD_ORDER=list(MODULES)
diff_heatmap(p4s,"Immune-cell pathway modules by distance to NPM1 blasts\n* BH<0.05 within zone","fig8_pathway_zone_heatmap.png",MOD_ORDER)

gsp=gs.set_index("module").reindex(MOD_ORDER[::-1]); y=np.arange(len(gsp))
fig,ax=plt.subplots(figsize=(7.5,5.2))
ax.errorbar(gsp.gradient_nonrelapse,y+0.15,xerr=gsp.sem_NR,fmt="o",color=C_NR,ms=6,capsize=2,label=GN)
ax.errorbar(gsp.gradient_relapse,y-0.15,xerr=gsp.sem_R,fmt="o",color=C_R,ms=6,capsize=2,label=GR)
ax.axvline(0,color="#52514e",lw=0.8)
for i,(mod,r) in enumerate(gsp.iterrows()):
    ax.text(ax.get_xlim()[1],i,f" p={r.p:.3f}{' '+star(r.p_adj) if star(r.p_adj) else ''}",va="center",fontsize=7.5,color="#52514e")
ax.set_yticks(y); ax.set_yticklabels(gsp.index); ax.set_xlabel("Contact − Far module score (mean ± SEM over ROIs)\n> 0 = higher next to blasts than 50–100 µm away")
ax.set_title("Proximity gradient of immune pathway modules",fontsize=10,loc="left"); ax.legend(frameon=False,loc="lower left"); fig.tight_layout()
fig.savefig(f"{OUT}/figures/fig9_gradient_forest.png",bbox_inches="tight"); plt.close(fig)

fig,axs=plt.subplots(3,4,figsize=(15,9.5))
for ax,mod in zip(axs.flat,MOD_ORDER):
    for g,c in [(GN,C_NR),(GR,C_R)]:
        q=p4[(p4.module==mod)&(p4.group==g)].groupby("zone_idx").score.agg(["mean","sem"]).reindex(range(4))
        ax.errorbar(range(4),q["mean"],yerr=q["sem"],color=c,lw=2,marker="o",ms=5,capsize=2,label=g)
    for zi in range(4):
        r=p4s[(p4s.module==mod)&(p4s.zone_idx==zi)]
        if len(r) and star(r.p_adj.iloc[0]): ax.text(zi,ax.get_ylim()[1]*0.95 if ax.get_ylim()[1]>0 else ax.get_ylim()[1],star(r.p_adj.iloc[0]),ha="center",fontsize=11,fontweight="bold")
    ax.set_xticks(range(4)); ax.set_xticklabels(Z4S); ax.set_title(mod,fontsize=9.5); ax.axhline(0,color="#c3c2b7",lw=0.7)
axs.flat[0].legend(frameon=False,fontsize=8)
fig.suptitle("Immune-cell module score vs distance to nearest NPM1 blast (mean ± SEM over ROIs; * BH<0.05)",fontsize=10.5,x=0.01,ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig10_pathway_lineplots.png"); plt.close(fig)

# %% [markdown]
# ## PART 5 — lineage-resolved modules (T cells / Myeloid / Plasma cells)
# %%
parts=[]
for lin,labs in LINEAGE.items():
    br={roi:(df[df.is_tumor],df[df.final_label.isin(labs)]) for roi,df in full.groupby("roi")}
    pl=zone_scores(br,10); pl["lineage"]=lin; parts.append(pl)
p5=pd.concat(parts); p5.to_csv(f"{OUT}/csv/P5_lineage_module_per_roi_zone.csv",index=False)
p5s=zone_stats(p5,["lineage","zone"]); p5s["zone_idx"]=p5s.zone.map({z:i for i,z in enumerate(Z4)}); p5s.to_csv(f"{OUT}/csv/P5_lineage_module_stats.csv",index=False)
print("\nPART 5 — lineage × zone, BH<0.05:"); print(p5s[p5s.sig][["lineage","zone","module","mean_nonrelapse","mean_relapse","p","p_adj"]].round(4).to_string(index=False))

fig,axs=plt.subplots(1,3,figsize=(17,5.6))
lim=max(0.3,np.nanmax(np.abs(p5s.diff_R_minus_NR)))
for ax,lin in zip(axs,LINEAGE):
    s=p5s[p5s.lineage==lin]
    Dm=s.pivot(index="module",columns="zone",values="diff_R_minus_NR").reindex(index=MOD_ORDER,columns=Z4)
    Qm=s.pivot(index="module",columns="zone",values="p_adj").reindex(index=MOD_ORDER,columns=Z4)
    im=ax.imshow(Dm.values,cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto")
    for i in range(Dm.shape[0]):
        for j in range(4):
            v=Dm.values[i,j]
            if not np.isnan(v): ax.text(j,i,f"{v:+.2f}{star(Qm.values[i,j])}",ha="center",va="center",fontsize=7,color="white" if abs(v)>0.6*lim else "#0b0b0b")
    ax.set_xticks(range(4)); ax.set_xticklabels(Z4S); ax.set_yticks(range(len(MOD_ORDER))); ax.grid(False)
    ax.set_yticklabels(MOD_ORDER if lin=="T cells" else []); ax.set_title(lin,fontsize=10,fontweight="bold")
cb=fig.colorbar(im,ax=axs,shrink=0.7); cb.set_label("Relapse − Non-relapse (mean z)\nred = higher in relapse")
fig.suptitle("Lineage-resolved pathway modules by distance zone (* BH<0.05 within lineage × zone)",fontsize=10.5,x=0.01,ha="left")
fig.savefig(f"{OUT}/figures/fig11_lineage_zone_heatmaps.png",bbox_inches="tight"); plt.close(fig)

KEYMODS=["Proliferation","Anti-apoptotic (Bcl-2)","Antigen presentation (HLA-DR)","Wnt / beta-catenin"]
fig,axs=plt.subplots(len(LINEAGE),len(KEYMODS),figsize=(13,9))
for i,lin in enumerate(LINEAGE):
    for j,mod in enumerate(KEYMODS):
        ax=axs[i,j]; s=p5[(p5.lineage==lin)&(p5.module==mod)&(p5.zone_idx==0)]
        data=[s[s.group==g].score.values for g in (GN,GR)]
        bp=ax.boxplot(data,widths=0.55,patch_artist=True,showfliers=False,medianprops=dict(color="#0b0b0b"))
        for patch,c in zip(bp["boxes"],[C_NR,C_R]): patch.set_facecolor(c); patch.set_alpha(0.35); patch.set_edgecolor(c)
        for k,(d,c) in enumerate(zip(data,[C_NR,C_R])): ax.scatter(np.random.default_rng(1).normal(k+1,0.06,len(d)),d,s=12,color=c,zorder=3,linewidths=0)
        r=p5s[(p5s.lineage==lin)&(p5s.module==mod)&(p5s.zone_idx==0)]
        ax.set_title(f"{lin}: {mod}\n" + (f"p={r.p.iloc[0]:.3f}, BH={r.p_adj.iloc[0]:.3f} {star(r.p_adj.iloc[0])}" if len(r) else "n too small"),fontsize=8)
        ax.set_xticks([1,2]); ax.set_xticklabels([GN,GR],fontsize=8); ax.grid(axis="x",visible=False); ax.axhline(0,color="#c3c2b7",lw=0.7)
fig.suptitle("Contact zone (<15 µm): key modules per immune lineage, one dot per ROI",fontsize=10.5,x=0.01,ha="left")
fig.tight_layout(); fig.savefig(f"{OUT}/figures/fig12_key_modules_contact_by_lineage.png"); plt.close(fig)

summary={"n_cells":len(m),"n_tumour":int(m.is_tumor.sum()),"rois_P1":near.roi.nunique(),"rois_P2":det.roi.nunique(),
         "rois_P3":lrdf.roi.nunique(),"rois_P4":p4.roi.nunique(),"rois_P5":p5.roi.nunique()}
pd.Series(summary).to_csv(f"{OUT}/csv/run_summary.csv"); print(summary); print("__DONE__")

# %% [markdown]
# ## PART 6 — patient-level check of the key results
# Several ROIs come from the same patient (11 patients: 6 non-relapse, 5 relapse; `../roi_patient_mapping.csv`). For each
# key endpoint the per-ROI values are averaged within each patient, then tested 6 vs 5 (Mann–Whitney, exact). The smallest
# possible two-sided p with 6 vs 5 is 0.0043. BH is applied across the endpoints listed here.
# %%
mp=pd.read_csv(f"{BASE}/roi_patient_mapping.csv"); PAT=dict(zip(mp.roi,mp.patient))
assert all(mp.group==mp.roi.map(grp))
EP=[]
for zn in ["<15 µm touching","15–30 µm immediate","30–60 µm local","60–100 µm intermediate","100+ µm far"]:
    EP.append((f"Endothelial % — {zn}",det[(det.cell_type=="Endothelial")&(det.zone==zn)][["roi","pct"]]))
for ctn in ["Immature myeloid CD11c+","CD8+ T","CD4+ T"]:
    EP.append((f"{ctn} % — <15 µm touching",det[(det.cell_type==ctn)&(det.zone=="<15 µm touching")][["roi","pct"]]))
EP.append(("All immune: Proliferation — Contact",p4[(p4.module=="Proliferation")&(p4.zone_idx==0)][["roi","score"]]))
for lin,mod in [("T cells","Proliferation"),("T cells","Type-I IFN / inflammation"),("T cells","Anti-apoptotic (Bcl-2)"),
                ("Myeloid","Proliferation"),("Myeloid","Wnt / beta-catenin"),("Myeloid","Antigen presentation (HLA-DR)")]:
    EP.append((f"{lin}: {mod} — Contact",p5[(p5.lineage==lin)&(p5.module==mod)&(p5.zone_idx==0)][["roi","score"]]))
rows=[]
for name,d in EP:
    d=d.rename(columns={d.columns[1]:"v"}).assign(patient=lambda x:x.roi.map(PAT),group=lambda x:x.roi.map(grp))
    pm=d.groupby(["patient","group"]).v.mean().reset_index()
    a,b=pm[pm.group==GN].v,pm[pm.group==GR].v
    rows.append({"endpoint":name,"nonrelapse_mean":a.mean(),"relapse_mean":b.mean(),"n_NR_patients":len(a),"n_R_patients":len(b),
                 "p_patient":mannwhitneyu(a,b,alternative="two-sided")[1] if len(a)>=3 and len(b)>=3 else np.nan})
p6=pd.DataFrame(rows); ok=p6.p_patient.notna(); p6.loc[ok,"p_adj"]=bh(p6.loc[ok,"p_patient"]); p6.to_csv(f"{OUT}/csv/P6_patient_level_key_results.csv",index=False)
print("\nPART 6 — patient-level (6 vs 5):"); print(p6.round(4).to_string(index=False))
