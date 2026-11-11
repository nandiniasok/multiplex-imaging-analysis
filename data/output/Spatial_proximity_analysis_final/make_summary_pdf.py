"""Summary PDF for Spatial_proximity_analysis_final. Tables are read from csv/; run after spatial_proximity_final.py."""
import os, pandas as pd
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, PageBreak, Spacer, KeepTogether
from PIL import Image as PImage
D=os.path.dirname(os.path.abspath(__file__)); F=f"{D}/figures"; C=f"{D}/csv"
ss=getSampleStyleSheet()
H1=ParagraphStyle("H1",parent=ss["Heading1"],fontSize=17,textColor=colors.HexColor("#1c3f6e"),spaceAfter=4)
H2=ParagraphStyle("H2",parent=ss["Heading2"],fontSize=12.5,textColor=colors.HexColor("#1c5cab"),spaceBefore=10,spaceAfter=4)
B=ParagraphStyle("B",parent=ss["BodyText"],fontSize=9.4,leading=13.2,spaceAfter=5)
BL=ParagraphStyle("BL",parent=B,leftIndent=12,bulletIndent=2)
SM=ParagraphStyle("SM",parent=B,fontSize=8,leading=10.5,textColor=colors.HexColor("#52514e"))
def P(t,s=B): return Paragraph(t,s)
def bul(t): return Paragraph("• "+t,BL)
def img(name,w=17.5*cm,maxh=21*cm):
    p=f"{F}/{name}"; iw,ih=PImage.open(p).size; W,H=w,w*ih/iw
    if H>maxh: H,W=maxh,maxh*iw/ih
    return Image(p,width=W,height=H)
def tbl(rows,colw,fs=7.8):
    t=Table(rows,colWidths=colw,repeatRows=1)
    t.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),fs),("GRID",(0,0),(-1,-1),0.3,colors.HexColor("#c3c2b7")),
        ("BACKGROUND",(0,0),(-1,0),colors.HexColor("#1c5cab")),("TEXTCOLOR",(0,0),(-1,0),colors.white),
        ("VALIGN",(0,0),(-1,-1),"MIDDLE"),("TOPPADDING",(0,0),(-1,-1),2),("BOTTOMPADDING",(0,0),(-1,-1),2)])); return t
def f(x,d=1): return f"{x:.{d}f}"
def pf(x): return "<0.001" if x<0.001 else f"{x:.3f}"
rs=pd.read_csv(f"{C}/run_summary.csv",index_col=0).iloc[:,0]
p1=pd.read_csv(f"{C}/P1_pct_near_tumour_stats.csv").sort_values("p_adj")
p2=pd.read_csv(f"{C}/P2_zone_composition_stats.csv"); p3=pd.read_csv(f"{C}/P3_LR_SIS_stats.csv")
p4=pd.read_csv(f"{C}/P4_pathway_stats_per_zone.csv"); g4=pd.read_csv(f"{C}/P4_gradient_stats.csv").sort_values("p")
p5=pd.read_csv(f"{C}/P5_lineage_module_stats.csv")

st=[]
st+= [P("Spatial proximity &amp; ligand–receptor analysis",H1),
      P(f"Nandhu_version_final cell types · 49 ROIs (23 non-relapse, 26 relapse) · {int(rs.n_cells):,} cells · "
        f"tumour = NPM1_blasts ({int(rs.n_tumour):,} cells) · 2026-11-11",SM),
      P("What was done",H2),
      bul("<b>Tumour.</b> Every cell labelled NPM1_blasts in Nandhu_version_final. This includes all v2 Tumor and Tumor/Erythroid cells."),
      bul("<b>Distance.</b> The nucleus-centre distance from each cell to its nearest NPM1 blast in the same ROI, at 0.325 µm per pixel. "
          "<15 µm means the cells are touching."),
      bul("<b>Part 1, proximity screen</b> (June method). The % of each cell type within 32.5 µm of a blast."),
      bul("<b>Part 2, distance zones</b> (June method). The cell-type make-up of non-tumour cells in 5 zones: "
          "&lt;15, 15–30, 30–60, 60–100 and 100+ µm."),
      bul("<b>Part 3, ligand–receptor</b> (July 3 method). For every blast–immune cell pair closer than 100 µm, the spatial "
          "interaction score is SIS = mean(max(z<sub>ligand</sub>,0) × max(z<sub>receptor</sub>,0)). The four pairs are "
          "PD-L1→PD-1, Galectin-9→TIM-3, Galectin-9→CD44 and HLA-DR→LAG-3. Receivers are scored as all immune cells and as T cells only."),
      bul("<b>Parts 4–5, pathway modules</b> (July 3 method). 12 marker modules are scored per immune cell as the mean "
          "per-machine z of their markers. They are compared by distance zone (Contact &lt;15, Near 15–30, Mid 30–50, "
          "Far 50–100 µm) for all immune cells together and then separately for T cells, Myeloid and Plasma cells."),
      bul("<b>Statistics.</b> One ROI is one replicate. Two-sided Mann–Whitney U, with Benjamini–Hochberg correction inside each "
          "family (within zone, or within lineage × zone). Unassigned/Other and Artifact are not cell types, so they are left "
          "out of every composition."),
      P("Key findings",H2),
      bul("<b>Relapse marrow is dominated by blasts, so everything sits closer to them.</b> NPM1 blasts are 27.9% of annotated cells in "
          "relapse ROIs versus 10.3% in non-relapse ROIs (BH p = 0.003). As a result, every cell type is 1.5–2× more often within 32.5 µm of a "
          "blast in relapse (all 14 types BH &lt; 0.05; Fig 1). This is mostly a consequence of tumour burden. It is not on its own evidence of "
          "targeted recruitment, which is why the zone analysis below compares <i>composition</i> at a fixed distance."),
      bul("<b>Blood vessels mark the non-relapse niche (the strongest result).</b> Endothelial cells are about 3× more common around "
          "blasts in non-relapse at <b>every</b> distance, from touching (8.6% vs 2.9%) to far (8.7% vs 1.7%), with BH p ≤ 0.004 in all "
          "5 zones (Figs 2–3). Non-relapse blasts sit in vascular-rich marrow. Relapse blasts sit in vessel-poor tissue."),
      bul("<b>The v2 dendritic-cell population (now 'Immature myeloid CD11c+') touches blasts only in non-relapse.</b> It is 3.6% of "
          "cells touching blasts in non-relapse versus 0.16% in relapse (BH 0.034). It is also enriched at 60–100 µm (BH 0.041). This "
          "reproduces the June dendritic-cell finding with the corrected labels."),
      bul("<b>T cells are packed against the blasts in relapse,</b> but they are not proliferating. In the touching zone, CD8+ T are 7.9% "
          "vs 3.0% (BH 0.011) and CD4+ T are 5.3% vs 2.8% (BH 0.022). The Ki67+CD147+CD45RA+ Leiden population is "
          "3.1% vs 0% (BH 0.007)."),
      bul("<b>Immune proliferation is switched off at the blast surface in relapse (ROI level).</b> This replicates the July 3 headline. For all "
          "immune cells, the Proliferation module (Ki-67/PCNA/RRM2/c-MYC) at contact is lower in relapse (BH 0.040; Figs 8, 10). By lineage: "
          "T cells at contact BH 0.013 and near BH 0.049, Myeloid at contact BH 0.044 (Figs 11–12)."),
      bul("<b>Wnt/β-catenin is raised in myeloid cells touching blasts in relapse</b> (BH 0.044), and it trends up across all zones. "
          "<b>Type-I IFN (IRF7/IFN-γ) is lower in T cells at contact in relapse</b> (BH 0.023)."),
      bul("<b>Patient-level check (Part 6).</b> Several ROIs come from the same patient, so the key results were re-tested with one value per "
          "patient (6 non-relapse vs 5 relapse). <b>The endothelial result holds</b>: BH 0.043 in 3 of 5 zones, raw p ≤ 0.017 in all 5. "
          "<b>The immune-state results (proliferation, IFN, Wnt) and the CD11c+/T-cell contact differences point the same way but are not "
          "significant per patient</b> (p 0.15–0.55). Treat them as ROI-level findings that need more patients."),
      bul("<b>No ligand–receptor pair separates the groups</b> (all BH &gt; 0.35; Figs 5–7). PD-L1→PD-1 on T cells is not higher in relapse "
          "at contact (0.14 vs 0.17)."),
      bul("<b>Not reproduced with these labels:</b> the July findings of lower T-cell Bcl-2 at contact (now raw p 0.031, BH 0.10) and "
          "lower myeloid HLA-DR (p 0.25). The far-zone macrophage excess in relapse is also no longer significant (p 0.22). Read these as trends only."),
      PageBreak()]

st+=[P("Part 1 — proximity screen",H2),img("fig1_pct_near_tumour.png",15*cm,15*cm)]
rows=[["Cell type","Non-relapse %","Relapse %","p","BH p"]]+[[r.cell_type,f(r["mean_nonrelapse_%"]),f(r["mean_relapse_%"]),pf(r.p),pf(r.p_adj)] for _,r in p1.iterrows()]
st+=[tbl(rows,[6.5*cm,2.6*cm,2.4*cm,2*cm,2*cm]),PageBreak()]
st+=[P("Part 2 — composition by distance zone",H2),img("fig2_zone_difference_heatmap.png",14*cm,17*cm)]
sig=p2[p2.sig].copy(); sig["zi"]=sig.zone.map({z:i for i,z in enumerate(["<15 µm touching","15–30 µm immediate","30–60 µm local","60–100 µm intermediate","100+ µm far"])})
rows=[["Zone","Cell type","Non-relapse %","Relapse %","BH p","Higher in"]]+[[r.zone,r.cell_type,f(r["mean_nonrelapse_%"],2),f(r["mean_relapse_%"],2),pf(r.p_adj),
      "Non-relapse" if r.diff_NR_minus_R>0 else "Relapse"] for _,r in sig.sort_values(["zi","p_adj"]).iterrows()]
st+=[P("Significant zone differences (BH &lt; 0.05 within zone):",SM),tbl(rows,[3.6*cm,5.6*cm,2.2*cm,2*cm,1.6*cm,2.2*cm]),PageBreak(),
     img("fig3_zone_lineplots_significant.png"),img("fig4_zone_stacked_composition.png"),PageBreak()]
st+=[P("Part 3 — ligand–receptor interaction scores",H2),img("fig5_LR_SIS_vs_distance_all_immune.png"),img("fig6_LR_SIS_vs_distance_Tcell_receiver.png"),
     img("fig7_LR_contact_boxplots.png",17*cm,9*cm)]
c=p3[p3.zone_idx==0]
rows=[["Receiver","Pair","Non-relapse SIS","Relapse SIS","p","BH p"]]+[[r.receiver,r.pair,f(r.mean_nonrelapse,3),f(r.mean_relapse,3),pf(r.p),pf(r.p_adj)] for _,r in c.iterrows()]
st+=[P("Contact zone (&lt;15 µm):",SM),tbl(rows,[2.6*cm,4*cm,2.8*cm,2.6*cm,1.8*cm,1.8*cm]),PageBreak()]
st+=[P("Part 4 — immune pathway modules vs distance",H2),img("fig8_pathway_zone_heatmap.png",14*cm,10*cm),img("fig10_pathway_lineplots.png",17.5*cm,12*cm),PageBreak(),
     img("fig9_gradient_forest.png",15*cm,10*cm)]
rows=[["Module","Gradient NR","Gradient R","p","BH p"]]+[[r.module,f(r.gradient_nonrelapse,3),f(r.gradient_relapse,3),pf(r.p),pf(r.p_adj)] for _,r in g4.iterrows()]
st+=[P("Contact − Far gradient. Nothing survives correction. The strongest raw trends (Type-I IFN, Proliferation, MAPK-JNK and EZH2) "
       "all rise towards the blasts in relapse and fall in non-relapse.",SM),tbl(rows,[6*cm,2.6*cm,2.6*cm,2*cm,2*cm]),PageBreak()]
st+=[P("Part 5 — lineage-resolved modules",H2),img("fig11_lineage_zone_heatmaps.png",17.5*cm,8*cm),img("fig12_key_modules_contact_by_lineage.png",17.5*cm,12*cm)]
s5=p5[p5.sig]
rows=[["Lineage","Zone","Module","Non-relapse","Relapse","BH p"]]+[[r.lineage,r.zone,r.module,f(r.mean_nonrelapse,2),f(r.mean_relapse,2),pf(r.p_adj)] for _,r in s5.iterrows()]
st+=[tbl(rows,[2.6*cm,3*cm,5*cm,2.2*cm,2*cm,1.8*cm]),PageBreak()]

p6=pd.read_csv(f"{C}/P6_patient_level_key_results.csv")
rows=[["Endpoint","NR mean","R mean","n pts NR/R","p (patient)","BH"]]+[[r.endpoint,f(r.nonrelapse_mean,2),f(r.relapse_mean,2),f"{int(r.n_NR_patients)}/{int(r.n_R_patients)}",pf(r.p_patient),pf(r.p_adj)] for _,r in p6.iterrows()]
st+=[P("Part 6 — patient-level check of the key results",H2),
     P("Per-ROI values are averaged within each patient, then compared 6 vs 5 (Mann–Whitney, exact; the smallest possible p is 0.0043). "
       "Contact-zone endpoints have 5 non-relapse patients because one patient has too few blasts for a contact zone.",SM),
     tbl(rows,[7.2*cm,1.8*cm,1.8*cm,2*cm,1.9*cm,1.6*cm]),PageBreak()]
st+=[P("Biological interpretation",H2),
     P("<b>1. The non-relapse blast lives in a vascular, immune-active niche.</b> Endothelium is consistently ~3× more abundant around "
       "non-relapse blasts at every distance, and the CD11c+ myeloid cells (the v2 'dendritic cells') make direct contact with blasts only "
       "in non-relapse. Together with proliferating immune cells at the contact zone, this describes an interface where antigen-presenting "
       "cells reach the blasts and immune cells are still expanding."),
     P("<b>2. Relapse blasts may build an 'immune-cold shell' (ROI-level evidence).</b> In relapse marrow the blast surface is crowded with CD4+ and CD8+ T cells, yet "
       "those T cells and the neighbouring myeloid cells are proliferation-arrested, T cells have lower type-I interferon signalling, and "
       "myeloid cells switch on the tolerogenic Wnt/β-catenin programme. The T cells are present but not expanding. This is "
       "a locally enforced suppression, strongest exactly at contact, not a lack of immune infiltration. These module differences are significant "
       "across ROIs but not yet across patients, so they need confirmation in more patients."),
     P("<b>3. No single checkpoint explains it.</b> None of PD-L1→PD-1, Galectin-9→TIM-3/CD44 or HLA-DR→LAG-3 differs between the "
       "groups. The relapse phenotype is a coordinated change in immune-cell <i>state</i> next to the blast, rather than one ligand–receptor axis."),
     P("Clinical interpretation",H2),
     bul("<b>Possible spatial relapse biomarker.</b> Three features separate relapse from non-relapse in MRD-positive remission marrow "
         "while the blast count alone does not capture them: endothelial depletion around blasts (patient-level confirmed), loss of CD11c+ myeloid contact, and "
         "immune proliferation arrest at the blast surface. If they are validated in an independent cohort, a combined score could stratify relapse risk beyond MRD transcript levels."),
     bul("<b>Therapy rationale.</b> PD-1/PD-L1 blockade alone is unlikely to be enough. The data point to strategies that "
         "restore immune proliferation and priming (for example, combinations that relieve local suppression or target Wnt/β-catenin in myeloid cells), "
         "and possibly to the vascular niche as a factor in disease control."),
     bul("<b>Caveats.</b>"),
     P("– 11 patients (6 non-relapse, 5 relapse) and 49 ROIs. ROI-level tests treat ROIs from the same patient as independent. "
       "Part 6 repeats the key endpoints per patient; only the endothelial result is confirmed there.",BL),
     P("– Machine is confounded with group: non-relapse is 12 Machine A + 11 Machine B ROIs, relapse is 22 A + 4 B. Module markers "
       "are z-scored within machine to reduce this.",BL),
     P("– The data are protein co-localisation, not signalling. Distances are nucleus centre to centre.",BL),
     P("– Relapse ROIs have about 2.7× more blasts, which inflates every proximity measure. This is why the zone-composition and "
       "module results, which are measured at a fixed distance, carry the biology.",BL),
     P("Files",H2),
     P("<b>spatial_proximity_final.py / .ipynb</b> is the full analysis (identical code). <b>csv/</b> holds per-ROI values and every "
       "statistics table (P1–P5). <b>figures/</b> holds figs 1–12. <b>make_summary_pdf.py</b> builds this report.",SM)]
SimpleDocTemplate(f"{D}/Spatial_proximity_analysis_final_summary.pdf",pagesize=A4,leftMargin=1.6*cm,rightMargin=1.6*cm,topMargin=1.5*cm,bottomMargin=1.5*cm,
                  title="Spatial proximity & ligand–receptor analysis — Nandhu_version_final").build(st)
print("wrote PDF")
