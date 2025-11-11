#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ============================================================
# Setup & Shared Utilities (Future-proof + labeled Neurons)
# ============================================================

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import Image, display

# ------------------------------------------------------------
# Global Visualization Settings
# ------------------------------------------------------------
sns.set(style="whitegrid", context="talk")

# ------------------------------------------------------------
# Folder Configuration
# ------------------------------------------------------------
DIR1 = "Composite6"   # Neuron 1
DIR2 = "Composite8"   # Neuron 2
LABEL1 = "Neuron 1"
LABEL2 = "Neuron 2"

OUT_DIR = Path("Comparison_Neuron1_vs_Neuron2")
OUT_DIR.mkdir(exist_ok=True)

# ------------------------------------------------------------
# Utility Functions
# ------------------------------------------------------------
def load_csv_safe(folder, relative_path):
    """
    Safely load a CSV file if it exists. Returns None if not found.
    """
    path = Path(folder) / relative_path
    if not path.exists():
        print(f"⚠️ Missing file: {path}")
        return None
    try:
        return pd.read_csv(path)
    except Exception as e:
        print(f"❌ Error reading {path.name}: {e}")
        return None

def show_thumbnail(path, width=500):
    """
    Display an image thumbnail inline in Jupyter (if file exists).
    """
    try:
        display(Image(filename=str(path), width=width))
    except Exception as e:
        print(f"⚠️ Could not display thumbnail for {path.name}: {e}")

def violin_compare(df1, df2, label1, label2, metric, title, out_path):
    """
    Create a violin plot comparing a numeric metric across two conditions.
    - Handles duplicate indices (fixes Step 5 ValueError)
    - Fixes Seaborn 0.14+ hue/palette deprecation warning
    - Fully compatible with all Step 3–7 metrics
    """
    if metric not in df1.columns or metric not in df2.columns:
        print(f"⚠️ Skipping {metric} — not found in both datasets.")
        return

    # --- Clean and ensure numeric data ---
    df1 = df1.reset_index(drop=True)
    df2 = df2.reset_index(drop=True)
    df1[metric] = pd.to_numeric(df1[metric], errors="coerce")
    df2[metric] = pd.to_numeric(df2[metric], errors="coerce")

    # --- Combine with descriptive neuron labels ---
    combined = pd.concat([
        df1.assign(Condition=label1),
        df2.assign(Condition=label2)
    ], ignore_index=True)

    # --- Violin Plot ---
    plt.figure(figsize=(6, 5))
    sns.violinplot(
        data=combined,
        x="Condition",
        y=metric,
        hue="Condition",       # ✅ prevents palette warning
        dodge=False,
        inner="quartile",
        palette=["#1f77b4", "#ff7f0e"],  # Neuron 1 (blue), Neuron 2 (orange)
        legend=False
    )
    plt.title(title)
    plt.tight_layout()

    # --- Save & Display ---
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"📈 Saved → {out_path.name}")
    show_thumbnail(out_path, width=450)


# In[4]:


# ============================================================
# Step 1 – Lysosome Count per Frame
# ============================================================
def compare_step1_lyso_count():
    csv_rel = "Step1_Lyso_Count_Outputs/Lysosome_Counts.csv"
    df1, df2 = load_csv_safe(DIR1, csv_rel), load_csv_safe(DIR2, csv_rel)

    if df1 is None or df2 is None:
        print("⚠️ Step 1 data missing.")
        return

    plt.figure(figsize=(7,5))
    plt.plot(df1["Frame"], df1["Lysosome_Count"], "-o", label="Neuron 1")
    plt.plot(df2["Frame"], df2["Lysosome_Count"], "-o", label="Neuron 2")
    plt.xlabel("Frame"); plt.ylabel("Lysosome Count")
    plt.title("Lysosome Count per Frame")
    plt.legend(); plt.tight_layout()
    out_path = OUT_DIR / "Step1_LysoCount_Comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("✅ Step 1 comparison complete.")
    show_thumbnail(out_path)

compare_step1_lyso_count()


# In[37]:


# ============================================================
# Step 2 – Morphology (Elongated vs Punctate)
# ============================================================
def compare_step2_morphology():
    csv_rel = "Step2_Morphology_Outputs/Step2_Morphology_Summary.csv"
    m1, m2 = load_csv_safe(DIR1, csv_rel), load_csv_safe(DIR2, csv_rel)

    if m1 is None or m2 is None:
        print("⚠️ Step 2 data missing.")
        return

    summary = pd.concat([
        m1.assign(Condition=DIR1), 
        m2.assign(Condition=DIR2)])
    melted = summary.melt(id_vars=["Frame","Condition"],
                          value_vars=["Elongated","Punctate"],
                          var_name="Type", value_name="Count")

    plt.figure(figsize=(7,5))
    sns.lineplot(data=melted, x="Frame", y="Count", hue="Type",
                 style="Condition", markers=True)
    plt.title("Mitochondrial Morphology per Frame")
    plt.tight_layout()
    out_path = OUT_DIR / "Step2_Morphology_Comparison.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("✅ Step 2 morphology comparison complete.")
    show_thumbnail(out_path)

compare_step2_morphology()


# In[38]:


# ============================================================
# Step 3 – Mitochondrial Shape Metrics
# ============================================================
def compare_step3_shape():
    csv_rel = "Step3_Shape_Analysis_Outputs/Step3_Mito_ShapeMetrics.csv"
    s1, s2 = load_csv_safe(DIR1, csv_rel), load_csv_safe(DIR2, csv_rel)

    if s1 is None or s2 is None:
        print("⚠️ Step 3 data missing.")
        return

    metrics = ["Circularity","Solidity","Aspect_Ratio","Eccentricity","Area"]
    for m in metrics:
        out_path = OUT_DIR / f"Step3_{m}_Comparison.png"
        violin_compare(s1, s2, "Neuron 1", "Neuron 2", m, f"Step 3: {m}", out_path)

    print("✅ Step 3 shape metrics comparison complete.")

compare_step3_shape()


# In[39]:


# ============================================================
# Step 5 – Motility Comparison
# ============================================================
def compare_step5_motility():
    csv_rel = "Step5_Motility_Outputs/Step5_Motility_Summary.csv"
    mot1, mot2 = load_csv_safe(DIR1, csv_rel), load_csv_safe(DIR2, csv_rel)

    if mot1 is None or mot2 is None:
        print("⚠️ Step 5 data missing.")
        return

    for metric in ["Mean_Velocity","Total_Displacement"]:
        out_path = OUT_DIR / f"Step5_{metric}_Comparison.png"
        violin_compare(mot1, mot2, "Neuron 1", "Neuron 2", metric, f"Step 5: {metric}", out_path)

    print("✅ Step 5 motility comparison complete.")

compare_step5_motility()


# In[40]:


# ============================================================
# Step 6 – Colocalization Metrics
# ============================================================
def compare_step6_colocalization():
    csv_rel = "Step6_Colocalization_Outputs/Step6_Colocalization.csv"
    c1, c2 = load_csv_safe(DIR1, csv_rel), load_csv_safe(DIR2, csv_rel)

    if c1 is None or c2 is None:
        print("⚠️ Step 6 data missing.")
        return

    metrics = ["Manders_M1","Manders_M2","Pearson_r","Percent_Overlap"]
    for m in metrics:
        out_path = OUT_DIR / f"Step6_{m}_Comparison.png"
        violin_compare(c1, c2, "Neuron 1", "Neuron 2", m, f"Step 6: {m}", out_path)

    print("✅ Step 6 colocalization comparison complete.")

compare_step6_colocalization()


# In[41]:


# ============================================================
# Step 7 – Integrated Correlation Matrix
# ============================================================
def compare_step7_integrated():
    csv_rel = "Step7_Integrated_Summary_Outputs/Step7_Integrated_CorrelationMatrix.csv"
    i1, i2 = load_csv_safe(DIR1, csv_rel), load_csv_safe(DIR2, csv_rel)

    if i1 is None or i2 is None:
        print("⚠️ Step 7 data missing.")
        return

    diff = i1.set_index(i1.columns[0]).subtract(
        i2.set_index(i2.columns[0]), fill_value=0
    )

    plt.figure(figsize=(14, 12))
    ax = sns.heatmap(diff, cmap="coolwarm", center=0)
    ax.set_xlabel("")     # ✅ removes unnamed horizontal label
    ax.set_ylabel("")     # ✅ removes unnamed vertical label
    plt.title("Step 7: Correlation Difference (Neuron 1 – Neuron 2)")
    plt.tight_layout()

    out_path = OUT_DIR / "Step7_CorrelationMatrix_Difference_Neuron1_vs_Neuron2.png"
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("✅ Step 7 integrated correlation comparison complete.")
    show_thumbnail(out_path, width=600)

compare_step7_integrated()


# In[ ]:




