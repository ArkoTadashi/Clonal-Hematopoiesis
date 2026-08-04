import re
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("final_combined_data.csv")

# Ensure string types for text columns
text_cols = ["mutations", "variants", "STAG2"]
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).replace("nan", np.nan)


# ==========================================
# 1. MUTATION SPECTRUM (Functional Impact)
# ==========================================
def classify_variant(val):
    if pd.isna(val) or val in ["0", "NA", "none", "None"]:
        return "Wildtype / No Mutation"
    val_lower = str(val).lower()

    if any(
        x in val_lower
        for x in ["nonsense", "stop", "*", "frame", "fs", "splice", "truncat"]
    ):
        return "Truncating / LoF (Nonsense/Frameshift/Splice)"
    elif any(x in val_lower for x in ["missense", "inframe", "subst"]):
        return "Missense / In-Frame"
    elif "deletion" in val_lower or "del" in val_lower:
        return "Deletion"
    elif "amplification" in val_lower or "amp" in val_lower:
        return "Amplification"
    else:
        return "Other / Unclassified"


# Combine variant/mutation text sources
combined_mut = df["mutations"].fillna("") + " " + df["variants"].fillna("")
df["mutation_type"] = combined_mut.apply(classify_variant)

print("--- STAG2 Mutation Spectrum ---")
print(df["mutation_type"].value_counts())
print("\n" + "=" * 50 + "\n")


# ==========================================
# 2. PROTEIN POSITION CLUSTERING (Hotspots)
# ==========================================
def extract_amino_acid_pos(text):
    """Extracts numeric protein position from notation like p.R216* or p.450_451del"""
    match = re.search(r"p\.[A-Za-z]+(\d+)", str(text))
    if match:
        return int(match.group(1))
    return np.nan


df["protein_position"] = combined_mut.apply(extract_amino_acid_pos)

mutated_samples = df[df["protein_position"].notna()]
print(
    f"Extracted amino acid positions for {len(mutated_samples)} mutation entries."
)
if not mutated_samples.empty:
    print("Summary of STAG2 mutation positions (Amino Acid 1-1231):")
    print(mutated_samples["protein_position"].describe())


# ==========================================
# 3. VISUALIZATION: Spectrum & Hotspots
# ==========================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=300)

# Plot 1: Mutation Spectrum Bar Chart
spectrum_counts = df["mutation_type"].value_counts()
sns.barplot(
    x=spectrum_counts.values,
    y=spectrum_counts.index,
    ax=axes[0],
    palette="viridis",
)
axes[0].set_title("STAG2 Mutation Spectrum", fontsize=12)
axes[0].set_xlabel("Sample Count")

# Plot 2: Histogram of Protein Positions (Hotspots)
if not mutated_samples.empty:
    sns.histplot(
        mutated_samples["protein_position"],
        bins=30,
        ax=axes[1],
        color="crimson",
        kde=True,
    )
    axes[1].set_xlim(0, 1250)  # STAG2 length ~1231 aa
    axes[1].set_title("STAG2 Mutation Distribution Along Protein", fontsize=12)
    axes[1].set_xlabel("Amino Acid Position (aa)")
    axes[1].set_ylabel("Mutation Count")
else:
    axes[1].text(
        0.5,
        0.5,
        "No explicit p.X123 protein positions found\nin text columns",
        ha="center",
        va="center",
    )

plt.tight_layout()
plt.savefig("stag2_mutation_spectrum.png", dpi=300)
plt.show()


# ==========================================
# 4. CO-OCCURRENCE / CORRELATION WITH ALTERED
# ==========================================
print("\n--- Co-occurrence with Copy Number Alterations (CNA) ---")
if "CNA" in df.columns and "Altered" in df.columns:
    cna_ct = pd.crosstab(
        df["Altered"], df["CNA"], margins=True, normalize="index"
    )
    print(cna_ct)