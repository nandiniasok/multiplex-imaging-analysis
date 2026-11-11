# %% [markdown]
# # Cellular neighbourhoods (CN) — Nandhu_version_final
# Method: Schürch et al., Cell 2020 — github.com/nolanlab/NeighborhoodCoordination (neighbourhood identification).
# 1. For every cell, take a **window** of its **k nearest cells** (including itself) in the same ROI. Run with k = 10 and k = 20.
# 2. Count the cell types in each window and turn the counts into fractions.
# 3. Cluster all windows with **MiniBatchKMeans, n = 10 neighbourhoods** (random_state 0). The cluster is the cell's CN.
#
# Cell types are the `final_label` values in Nandhu_version_final (tumour = NPM1_blasts). **Unassigned/Other and Artifact are excluded**:
# they are neither counted in windows nor given a CN. They appear as -1 in the MACSiQ View import files.
# CNs are numbered by size (CN0 = largest) and named after their two most enriched cell types among those making up ≥5% of the CN.
# Groups: Non-relapse = 23 ROIs, Relapse = 26 ROIs, with one ROI as one replicate. Statistics: Mann–Whitney U, BH across the 10 CNs.

# %%
import os, csv, warnings
import numpy as np, pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import MiniBatchKMeans
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
BASE="/mnt/e/multiplex-imaging-analysis/data/output"
FINAL=f"{BASE}/Nandhu_version_final/csv/cell_assignments.csv"
OUT=os.environ.get("MRD_OUT",f"{BASE}/CN_final")
N_CN=10; KS=[10,20]; SEED=0; EXCLUDE={"Unassigned/Other","Artifact"}
X,Y="Nuc X","Nuc Y Inv"; C_NR,C_R="#2a78d6","#e34948"; GN,GR="Non-relapse","Relapse"
plt.rcParams.update({"font.size":9,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":200,
                     "axes.edgecolor":"#52514e","xtick.color":"#52514e","ytick.color":"#52514e"})
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
COL={"CD4+ T":"#1c5cab","CD8+ T":"#3987e5","Plasma cells":"#86b6ef","Granulocyte":"#eb6834","Macrophages":"#e34948",
     "Monocytes":"#e87ba4","Mast cells":"#eda100","HLA-DR-hi APC (CD1c+)":"#6250d6","Immature myeloid CD11c+":"#9085e9",
     "Erythroid":"#8c2b2b","NPM1_blasts":"#1a1a19","Mixed progenitors":"#a8a79f","Endothelial":"#1baf7a","MSC":"#008300",
     "Megakaryocyte":"#8a5a00","Unassigned-Leiden: CD68+CD3+Tryptase+":"#c3c2b7","Unassigned-Leiden: Ki67+CD147+CD45RA+":"#77756e"}
# 10 CN colours, fixed by CN number (not by rank) across both k
CN_COL=["#2a78d6","#eb6834","#1baf7a","#eda100","#e87ba4","#008300","#6250d6","#e34948","#1a1a19","#a8a79f"]

allc=pd.read_csv(FINAL,dtype={"Cell Id":str}); allc["roi"]=allc.unique_region.str.replace("/","_",regex=False); allc["group"]=allc.roi.map(grp)
cells=allc[~allc.final_label.isin(EXCLUDE)].reset_index(drop=True)
CTS=[c for c in COL if c in set(cells.final_label)]
print(f"{len(allc):,} cells; {len(cells):,} used for CN ({len(allc)-len(cells):,} Unassigned/Artifact excluded); {len(CTS)} cell types; ROIs {cells.roi.nunique()}")

# %% [markdown]
# ## Neighbourhood identification for k = 10 and k = 20
# %%
def windows(k):
    oh=pd.get_dummies(cells.final_label).reindex(columns=CTS,fill_value=0).values.astype(np.float32)
    win=np.zeros_like(oh)
    for reg,idx in cells.groupby("unique_region").indices.items():
        idx=np.asarray(idx); kk=min(k,len(idx)); xy=cells.loc[idx,[X,Y]].values
        _,nbr=NearestNeighbors(n_neighbors=kk).fit(xy).kneighbors(xy)
        win[idx]=oh[idx][nbr].sum(axis=1)
    return win/win.sum(axis=1,keepdims=True)

results={}
for k in KS:
    D=f"{OUT}/k{k}"; os.makedirs(f"{D}/cn_maps",exist_ok=True)
    W=windows(k)
    km=MiniBatchKMeans(n_clusters=N_CN,random_state=SEED,n_init=3,batch_size=1000).fit(W)
    order=pd.Series(km.labels_).value_counts().index.tolist(); remap={o:i for i,o in enumerate(order)}
    cn=np.array([remap[x] for x in km.labels_]); centers=km.cluster_centers_[order]
    overall=W.mean(axis=0); enr=np.log2((centers+1e-4)/(overall+1e-4))
    enrdf=pd.DataFrame(enr,columns=CTS,index=[f"CN{i}" for i in range(N_CN)])
    comp=pd.DataFrame(centers,columns=CTS,index=enrdf.index)
    # name = the 2 most enriched cell types among those making up >=5% of the CN (pure enrichment favours rare types)
    def cn_name(i):
        ok=comp.columns[comp.loc[f"CN{i}"]>=0.05]
        return " + ".join(enrdf.loc[f"CN{i}",ok].sort_values(ascending=False).index[:2])
    names={i:cn_name(i) for i in range(N_CN)}
    enrdf.to_csv(f"{D}/CN_enrichment_log2.csv"); (comp*100).round(2).to_csv(f"{D}/CN_composition_percent.csv")
    pd.DataFrame({"CN":list(enrdf.index),"name":[names[i] for i in range(N_CN)],"n_cells":[int((cn==i).sum()) for i in range(N_CN)]}).to_csv(f"{D}/CN_names.csv",index=False)
    ck=cells[["unique_region","Cell Id","final_label",X,Y,"machine","roi","group"]].copy(); ck["CN"]=cn
    ck.to_csv(f"{D}/cells_with_CN.csv",index=False); results[k]=(ck,names)
    print(f"\n=== k={k} ==="); [print(f"  CN{i}: {names[i]:60s} n={int((cn==i).sum()):,}") for i in range(N_CN)]

    # --- enrichment heatmap ---
    M=enrdf.T; lim=3
    fig,ax=plt.subplots(figsize=(11,0.38*len(CTS)+2.4))
    im=ax.imshow(M.values.clip(-lim,lim),cmap="RdBu_r",vmin=-lim,vmax=lim,aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v=M.values[i,j]
            if v>1: ax.text(j,i,f"{v:.1f}",ha="center",va="center",fontsize=6.5,color="white" if v>2 else "#0b0b0b")
    ax.set_xticks(range(N_CN)); ax.set_xticklabels([f"CN{i}\n{(cn==i).sum():,}" for i in range(N_CN)],fontsize=8)
    ax.set_yticks(range(len(CTS))); ax.set_yticklabels(CTS,fontsize=8); ax.spines[:].set_visible(False)
    cb=fig.colorbar(im,ax=ax,shrink=0.6); cb.set_label("log2 enrichment vs tissue mean")
    ax.set_title(f"Cellular neighbourhoods, k={k}: cell-type enrichment (numbers shown where log2 > 1)",fontsize=10,loc="left")
    fig.tight_layout(); fig.savefig(f"{D}/fig1_CN_enrichment_heatmap.png"); plt.close(fig)

    # --- cell-type composition within each CN (stacked %) ---
    within=pd.crosstab(ck.CN,ck.final_label,normalize="index").reindex(columns=CTS,fill_value=0)*100
    within.round(3).to_csv(f"{D}/celltype_within_CN_percent.csv")
    fig,ax=plt.subplots(figsize=(12,5.5)); bottom=np.zeros(N_CN)
    for ct in CTS:
        ax.bar(range(N_CN),within[ct],bottom=bottom,color=COL[ct],edgecolor="white",linewidth=0.7,width=0.78,label=ct); bottom+=within[ct].values
    ax.set_xticks(range(N_CN)); ax.set_xticklabels([f"CN{i}\n{names[i].replace(' + ','\n')}" for i in range(N_CN)],fontsize=6.5)
    ax.set_ylabel("% of cells in the CN"); ax.set_ylim(0,100); ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=7)
    ax.set_title(f"Cell-type composition of each neighbourhood (k={k})",fontsize=10,loc="left"); fig.tight_layout()
    fig.savefig(f"{D}/fig2_celltype_within_CN.png",bbox_inches="tight"); plt.close(fig)

    # --- CN frequency per ROI (stacked %), ROIs grouped by relapse status ---
    freq=pd.crosstab(ck.roi,ck.CN,normalize="index").reindex(columns=range(N_CN),fill_value=0)*100
    freq.insert(0,"group",freq.index.map(grp)); freq=freq.sort_values(["group",0],ascending=[True,False])
    freq.round(3).to_csv(f"{D}/CN_frequency_per_ROI_percent.csv")
    fig,ax=plt.subplots(figsize=(15,5.4)); bottom=np.zeros(len(freq)); xs=np.arange(len(freq))
    for i in range(N_CN):
        ax.bar(xs,freq[i],bottom=bottom,color=CN_COL[i],edgecolor="white",linewidth=0.4,width=0.85,label=f"CN{i}: {names[i]}"); bottom+=freq[i].values
    nnr=(freq.group==GN).sum(); ax.axvline(nnr-0.5,color="#0b0b0b",lw=1.2)
    ax.text((nnr-1)/2,103,GN,ha="center",color=C_NR,fontweight="bold"); ax.text(nnr+(len(freq)-nnr-1)/2,103,GR,ha="center",color=C_R,fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([r.replace("Machine ","").replace("_ROI_"," R") for r in freq.index],rotation=90,fontsize=6.5)
    ax.set_ylabel("% of cells in ROI"); ax.set_ylim(0,108); ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=7)
    ax.set_title(f"Neighbourhood frequency per ROI (k={k})",fontsize=10,loc="left",pad=16); fig.tight_layout()
    fig.savefig(f"{D}/fig3_CN_frequency_per_ROI.png",bbox_inches="tight"); plt.close(fig)

    # --- CN abundance Non-relapse vs Relapse (per-ROI %, MW + BH) ---
    rows=[]
    for i in range(N_CN):
        a=freq.loc[freq.group==GN,i]; b=freq.loc[freq.group==GR,i]
        rows.append({"CN":f"CN{i}","name":names[i],"nonrelapse_%":a.mean(),"sem_NR":a.sem(),"relapse_%":b.mean(),"sem_R":b.sem(),
                     "diff_NR_minus_R":a.mean()-b.mean(),"p":mannwhitneyu(a,b,alternative="two-sided")[1]})
    stt=pd.DataFrame(rows); stt["p_adj"]=bh(stt.p); stt["sig"]=stt.p_adj<0.05; stt.to_csv(f"{D}/CN_group_stats.csv",index=False)
    fig,ax=plt.subplots(figsize=(12,4.8)); w=0.38; xs=np.arange(N_CN)
    for k2,(g,c,col,sem) in enumerate([(GN,C_NR,"nonrelapse_%","sem_NR"),(GR,C_R,"relapse_%","sem_R")]):
        ax.bar(xs+(k2-0.5)*w,stt[col],w,yerr=stt[sem],color=c,label=g,error_kw=dict(lw=0.8,ecolor="#52514e"))
        for i in range(N_CN):
            v=freq.loc[freq.group==g,i].values; ax.scatter(np.full(len(v),i+(k2-0.5)*w)+np.random.default_rng(i).normal(0,0.04,len(v)),v,s=6,color="#0b0b0b",alpha=0.35,linewidths=0,zorder=3)
    for i,r in stt.iterrows():
        s=star(r.p_adj) or ("†" if r.p<0.05 else "")
        if s: ax.text(i,max(freq[i].max(),1)+1,s,ha="center",fontsize=11,fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels([f"CN{i}\n{names[i].replace(' + ','\n')}" for i in range(N_CN)],fontsize=6.5)
    ax.set_ylabel("% of cells in ROI (mean ± SEM; dots = ROIs)"); ax.legend(frameon=False)
    ax.set_title(f"Neighbourhood abundance, Non-relapse vs Relapse (k={k}); * BH<0.05, † raw p<0.05",fontsize=10,loc="left")
    fig.tight_layout(); fig.savefig(f"{D}/fig4_CN_abundance_group.png",bbox_inches="tight"); plt.close(fig)
    print(stt[["CN","name","nonrelapse_%","relapse_%","p","p_adj"]].round(4).to_string(index=False))

    # --- spatial CN maps, one per ROI ---
    for reg,g in ck.groupby("unique_region"):
        fig,ax=plt.subplots(figsize=(8,7)); ps=max(1.2,min(6,2200/np.sqrt(len(g))))
        for i in range(N_CN):
            mm=g.CN.values==i
            if mm.any(): ax.scatter(g[X].values[mm],g[Y].values[mm],s=ps,c=CN_COL[i],linewidths=0,label=f"CN{i}: {names[i]}",rasterized=True)
        ax.set_aspect("equal"); ax.invert_yaxis(); ax.set_xticks([]); ax.set_yticks([]); ax.spines[:].set_visible(False)
        ax.set_title(f"{reg} — {grp(reg.replace('/','_'))} — CN k={k}",fontsize=9,loc="left")
        ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),fontsize=6.5,markerscale=3,frameon=False)
        fig.tight_layout(); fig.savefig(f"{D}/cn_maps/{reg.replace('/','_').replace(' ','_')}_CN_k{k}.png",dpi=120,bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## Cell-type fraction, Non-relapse vs Relapse (single stacked bar per group)
# %%
fr=pd.crosstab(cells.roi,cells.final_label,normalize="index").reindex(columns=CTS,fill_value=0)*100
fr["group"]=fr.index.map(grp); gm=fr.groupby("group")[CTS].mean().reindex([GN,GR])
gm.T.round(3).to_csv(f"{OUT}/celltype_fraction_by_group_percent.csv")
st=[]
for ct in CTS:
    a,b=fr.loc[fr.group==GN,ct],fr.loc[fr.group==GR,ct]; st.append({"cell_type":ct,"nonrelapse_%":a.mean(),"relapse_%":b.mean(),"p":mannwhitneyu(a,b,alternative="two-sided")[1]})
st=pd.DataFrame(st); st["p_adj"]=bh(st.p); st.to_csv(f"{OUT}/celltype_fraction_by_group_stats.csv",index=False)
fig,ax=plt.subplots(figsize=(6.5,6)); bottom=np.zeros(2)
for ct in CTS:
    ax.bar([0,1],gm[ct],bottom=bottom,color=COL[ct],edgecolor="white",linewidth=0.8,width=0.6,label=ct)
    for j in range(2):
        if gm[ct].iloc[j]>=4: ax.text(j,bottom[j]+gm[ct].iloc[j]/2,f"{gm[ct].iloc[j]:.0f}%",ha="center",va="center",fontsize=7,color="white")
    bottom+=gm[ct].values
ax.set_xticks([0,1]); ax.set_xticklabels([f"{GN}\n(23 ROIs)",f"{GR}\n(26 ROIs)"]); ax.set_ylabel("% of annotated cells (mean over ROIs)"); ax.set_ylim(0,100)
ax.legend(loc="center left",bbox_to_anchor=(1.01,0.5),frameon=False,fontsize=7); ax.set_title("Cell-type composition by group",fontsize=10,loc="left")
fig.tight_layout(); fig.savefig(f"{OUT}/fig_celltype_fraction_by_group.png",bbox_inches="tight"); plt.close(fig)

# %% [markdown]
# ## MACSiQ View import files: one CSV per ROI
# Every cell of the ROI, keyed by the original MACSiQ `Cell Id`. Values are numeric so MACSiQ View can gate on them:
# `CN_k10` / `CN_k20` = 0–9, or −1 for excluded (Unassigned/Artifact) cells; `CellType_int` = `final_label_int`.
# %%
MQ=f"{OUT}/MACSiQ_import"; os.makedirs(MQ,exist_ok=True)
lab=allc[["unique_region","Cell Id",X,Y,"final_label","final_label_int"]].copy()
for k in KS:
    ck,_=results[k]; lab=lab.merge(ck[["unique_region","Cell Id","CN"]].rename(columns={"CN":f"CN_k{k}"}),on=["unique_region","Cell Id"],how="left")
    lab[f"CN_k{k}"]=lab[f"CN_k{k}"].fillna(-1).astype(int)
lab["Relapse"]=(lab.unique_region.str.replace("/","_",regex=False).map(grp)==GR).astype(int)
for reg,g in lab.groupby("unique_region"):
    g[["Cell Id",X,Y,"CN_k10","CN_k20","final_label_int","Relapse"]].rename(columns={"final_label_int":"CellType_int"}).to_csv(
        f"{MQ}/{reg.replace('/','_').replace(' ','_')}_CN.csv",index=False,quoting=csv.QUOTE_NONNUMERIC)
legend=[]
for k in KS:
    _,names=results[k]; legend+= [{"column":f"CN_k{k}","value":i,"meaning":f"CN{i}: {names[i]}"} for i in range(N_CN)]+[{"column":f"CN_k{k}","value":-1,"meaning":"not assigned (Unassigned/Other or Artifact)"}]
key=allc.groupby("final_label_int").final_label.first()
legend+=[{"column":"CellType_int","value":int(i),"meaning":v} for i,v in key.items()]+[{"column":"Relapse","value":0,"meaning":GN},{"column":"Relapse","value":1,"meaning":GR}]
pd.DataFrame(legend).to_csv(f"{MQ}/_LEGEND_value_meanings.csv",index=False)
print(f"wrote {lab.unique_region.nunique()} MACSiQ CSVs to {MQ}")
print("__DONE__")

# %% [markdown]
# ## Patient-level CN abundance (one value per patient = mean of its ROI percentages; 6 vs 5 patients)
# %%
mp=pd.read_csv(f"{BASE}/roi_patient_mapping.csv"); PAT=dict(zip(mp.roi,mp.patient))
for k in KS:
    ck,names=results[k]
    fr=pd.crosstab(ck.roi,ck.CN,normalize="index").reindex(columns=range(N_CN),fill_value=0)*100
    fr["patient"]=fr.index.map(PAT); fr["group"]=fr.index.map(grp); pm=fr.groupby(["patient","group"])[list(range(N_CN))].mean().reset_index()
    rows=[]
    for i in range(N_CN):
        a,b=pm[pm.group==GN][i],pm[pm.group==GR][i]
        rows.append({"CN":f"CN{i}","name":names[i],"nonrelapse_%":a.mean(),"relapse_%":b.mean(),"p_patient":mannwhitneyu(a,b,alternative="two-sided")[1]})
    r=pd.DataFrame(rows); r["p_adj"]=bh(r.p_patient); r.to_csv(f"{OUT}/k{k}/CN_group_stats_patient_level.csv",index=False)
    pm.round(3).to_csv(f"{OUT}/k{k}/CN_frequency_per_patient_percent.csv",index=False)
    print(f"\nk={k} patient-level:"); print(r.round(4).to_string(index=False))
