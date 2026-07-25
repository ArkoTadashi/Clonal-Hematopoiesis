import sys
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import matplotlib.pyplot as plt

GENE = "STAG2"

FILES = {
    "mutations":     "Data/mutations.txt",        # required
    "cna":           "Data/cna.txt",               # required for the hemizygous-loss check
    "structural_variants": None,              # optional, set path if you have it
    "sample_matrix": "Data/sample_matrix.txt", # cBioPortal's own 1/0 altered flag, for cross-checking
    "mrna_zscore":   None,                    # optional, e.g. "mrna_zscore.txt"
    "clinical":      None,                    # optional, download separately from "Clinical Data" tab
}

OUTPUT_CSV = "stag2_combined_status.csv"
LOG_TXT = "stag2_analysis_log.txt"

TRUNCATING_CLASSES = {
    "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
    "Splice_Site", "Nonstop_Mutation", "Translation_Start_Site",
}


class _Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)

    def flush(self):
        for s in self.streams:
            s.flush()


_log_file = open(LOG_TXT, "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_file)



def inspect(path, label):
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        print(f"[{label}] file not found at '{path}' -- skipping")
        return None
    df = pd.read_csv(p, sep="\t", comment="#")
    print(f"\n[{label}] shape={df.shape}")
    print(f"[{label}] columns: {list(df.columns)}")
    #print(df.head(3).to_string())
    return df


print("=" * 70)
print("STEP 1: Inspecting raw files")
print("=" * 70)
raw = {name: inspect(path, name) for name, path in FILES.items()}


# ------------------------------------------------------------------------
# 2. GENERIC LOADER -- handles matrix or transposed-matrix orientation
# ------------------------------------------------------------------------
def load_gene_matrix(df, gene=GENE):
    """Return a tidy 2-column frame: SAMPLE_ID, value -- from either
    orientation of a single-gene cBioPortal download."""
    sample_cols = [c for c in df.columns if "SAMPLE" in c.upper()]
    if sample_cols:
        sample_col = sample_cols[0]
        gene_cols = [c for c in df.columns if c.upper() == gene.upper()]
        if not gene_cols:
            raise ValueError(
                f"Gene column '{gene}' not found. Columns present: {list(df.columns)}"
            )
        out = df[[sample_col, gene_cols[0]]].copy()
        out.columns = ["SAMPLE_ID", "value"]
        return out
    else:
        id_col = df.columns[0]
        row = df[df[id_col].astype(str).str.upper() == gene.upper()]
        if row.empty:
            raise ValueError(f"Gene '{gene}' not found in first column of this file.")
        out = row.drop(columns=[id_col]).T.reset_index()
        out.columns = ["SAMPLE_ID", "value"]
        return out


def classify_variant_classification(vc):
    if vc in TRUNCATING_CLASSES:
        return "Truncating"
    if vc == "Missense_Mutation":
        return "Missense"
    if vc in {"In_Frame_Del", "In_Frame_Ins"}:
        return "In-frame indel"
    return "Other"


def classify_protein_change(call):
    """Crude heuristic for protein-change notation (e.g. 'R216fs*45').
    Less reliable than Variant_Classification -- used only as a fallback."""
    if pd.isna(call) or str(call).strip() == "":
        return "WT"
    parts = [p.strip() for p in str(call).split(",") if p.strip()]
    for p in parts:
        pl = p.lower()
        if "fs" in pl or p.rstrip().endswith("*") or "splice" in pl:
            return "Truncating"
    return "Missense/Other"


def load_mutation_data(df, gene=GENE):
    """Handles either a long-format MAF-like file (with Variant_Classification)
    or the simplified single-gene matrix download."""
    cols_upper = {c.upper(): c for c in df.columns}
    if "VARIANT_CLASSIFICATION" in cols_upper:
        gene_col = cols_upper.get("HUGO_SYMBOL")
        sample_col = cols_upper.get("TUMOR_SAMPLE_BARCODE") or cols_upper.get("SAMPLE_ID")
        vc_col = cols_upper["VARIANT_CLASSIFICATION"]
        sub = df[df[gene_col].astype(str).str.upper() == gene.upper()].copy()
        sub = sub.rename(columns={sample_col: "SAMPLE_ID", vc_col: "variant_classification"})
        sub["mut_type"] = sub["variant_classification"].apply(classify_variant_classification)
        return sub[["SAMPLE_ID", "variant_classification", "mut_type"]].drop_duplicates("SAMPLE_ID")
    else:
        tidy = load_gene_matrix(df, gene).rename(columns={"value": "protein_change"})
        tidy["has_mutation"] = tidy["protein_change"].notna() & (
            tidy["protein_change"].astype(str).str.strip() != ""
        )
        tidy["mut_type"] = tidy["protein_change"].apply(classify_protein_change)
        return tidy


# ------------------------------------------------------------------------
# 3. BUILD COMBINED PER-SAMPLE TABLE
# ------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STEP 2: Building combined alteration table")
print("=" * 70)

combined = None

if raw.get("mutations") is not None:
    mut = load_mutation_data(raw["mutations"])
    if "has_mutation" not in mut.columns:
        mut["has_mutation"] = mut["mut_type"] != "WT"
    combined = mut[["SAMPLE_ID", "has_mutation", "mut_type"]]
    print(f"Mutations loaded: {combined['has_mutation'].sum()} mutated samples "
          f"out of {len(combined)}")

if raw.get("cna") is not None:
    cna = load_gene_matrix(raw["cna"]).rename(columns={"value": "cna_value"})
    cna["cna_value"] = pd.to_numeric(cna["cna_value"], errors="coerce")
    cna_map = {-2: "Deep deletion", -1: "Shallow deletion/loss",
               0: "Diploid", 1: "Gain", 2: "Amplification"}
    cna["cna_call"] = cna["cna_value"].map(cna_map)
    print(f"CNA loaded. Call distribution:\n{cna['cna_call'].value_counts(dropna=False)}")
    combined = cna if combined is None else combined.merge(cna, on="SAMPLE_ID", how="outer")

if raw.get("structural_variants") is not None:
    sv = load_gene_matrix(raw["structural_variants"]).rename(columns={"value": "sv_call"})
    sv["has_sv"] = sv["sv_call"].notna() & (sv["sv_call"].astype(str).str.strip() != "")
    combined = sv if combined is None else combined.merge(sv, on="SAMPLE_ID", how="outer")

if combined is None:
    raise RuntimeError("No mutation or CNA file loaded -- nothing to analyze. Check FILES paths.")

def bool_col(df, col):
    """Return df[col] as a boolean Series aligned to df.index, or an
    all-False Series of the same length if the column isn't present.
    (combined.get(col, False) breaks when col is missing, since the
    fallback plain bool has no .fillna() -- this avoids that.)"""
    if col in df.columns:
        return df[col].fillna(False).astype(bool)
    return pd.Series(False, index=df.index)


def cna_loss_col(df):
    if "cna_value" in df.columns:
        return (df["cna_value"] <= -1).fillna(False)
    return pd.Series(False, index=df.index)


# NOTE on X-linked hemizygosity: STAG2 sits on chromosome X. In males, a
# single copy is the normal state, so a GISTIC "-1" call doesn't carry the
# same meaning as it would for an autosomal gene. If your clinical file
# has a SEX column, it's worth checking that before treating cna<=-1 as
# pathological loss across all samples uniformly.
combined["STAG2_altered"] = (
    bool_col(combined, "has_mutation")
    | cna_loss_col(combined)
    | bool_col(combined, "has_sv")
)

print(f"\nTotal combined altered samples: {combined['STAG2_altered'].sum()} "
      f"out of {len(combined)}")

# Cross-check against cBioPortal's own altered/unaltered flag, if provided
if raw.get("sample_matrix") is not None:
    sm_df = raw["sample_matrix"]
    try:
        sm = load_gene_matrix(sm_df).rename(columns={"value": "cbioportal_flag"})
        sm["cbioportal_flag"] = pd.to_numeric(sm["cbioportal_flag"], errors="coerce")
        check = combined.merge(sm, on="SAMPLE_ID", how="inner")
        mismatch = check[check["STAG2_altered"] != check["cbioportal_flag"].astype(bool)]
        print(f"Cross-check vs. cBioPortal's altered flag: {len(mismatch)} of "
              f"{len(check)} samples disagree (worth reviewing if > 0)")
    except Exception as e:
        print(f"Could not cross-check sample_matrix file: {e}")

if "mut_type" in combined.columns:
    print(f"\nMutation type breakdown:\n{combined['mut_type'].value_counts(dropna=False)}")

combined.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved combined table to {OUTPUT_CSV}")


# ------------------------------------------------------------------------
# 4. mRNA EXPRESSION: altered vs. unaltered
# ------------------------------------------------------------------------
if raw.get("mrna_zscore") is not None:
    print("\n" + "=" * 70)
    print("STEP 3: mRNA expression comparison")
    print("=" * 70)
    mrna = load_gene_matrix(raw["mrna_zscore"]).rename(columns={"value": "mrna_z"})
    mrna["mrna_z"] = pd.to_numeric(mrna["mrna_z"], errors="coerce")
    merged = combined.merge(mrna, on="SAMPLE_ID", how="inner")

    altered_vals = merged.loc[merged["STAG2_altered"], "mrna_z"].dropna()
    unaltered_vals = merged.loc[~merged["STAG2_altered"], "mrna_z"].dropna()

    if len(altered_vals) > 0 and len(unaltered_vals) > 0:
        u_stat, p_val = stats.mannwhitneyu(altered_vals, unaltered_vals, alternative="two-sided")
        print(f"Altered n={len(altered_vals)}, median z={altered_vals.median():.2f}")
        print(f"Unaltered n={len(unaltered_vals)}, median z={unaltered_vals.median():.2f}")
        print(f"Mann-Whitney U p-value: {p_val:.4g}")

        fig, ax = plt.subplots(figsize=(5, 5))
        ax.boxplot([unaltered_vals, altered_vals], labels=["Unaltered", "Altered"])
        ax.set_ylabel(f"{GENE} mRNA z-score")
        ax.set_title(f"{GENE} expression by alteration status\n(p={p_val:.3g})")
        plt.tight_layout()
        plt.savefig("stag2_expression_boxplot.png", dpi=150)
        print("Saved plot to stag2_expression_boxplot.png")
    else:
        print("Not enough data in one of the groups to compare.")


# ------------------------------------------------------------------------
# 5. SURVIVAL ANALYSIS: altered vs. unaltered
# ------------------------------------------------------------------------
if raw.get("clinical") is not None:
    print("\n" + "=" * 70)
    print("STEP 4: Survival analysis")
    print("=" * 70)
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        print("lifelines not installed -- run: pip install lifelines")
        raise SystemExit

    clin = raw["clinical"]
    sample_cols = [c for c in clin.columns if "SAMPLE" in c.upper()]
    if not sample_cols:
        raise ValueError("Could not find a sample ID column in the clinical file.")
    clin = clin.rename(columns={sample_cols[0]: "SAMPLE_ID"})

    # cBioPortal clinical exports typically use OS_STATUS ("1:DECEASED"/"0:LIVING")
    # and OS_MONTHS. Adjust these column names if your file differs
    # (e.g. some studies use EFS_STATUS/EFS_MONTHS instead).
    if "OS_STATUS" not in clin.columns or "OS_MONTHS" not in clin.columns:
        print(f"Expected OS_STATUS / OS_MONTHS columns not found. "
              f"Columns present: {list(clin.columns)}")
    else:
        surv = combined.merge(clin[["SAMPLE_ID", "OS_STATUS", "OS_MONTHS"]],
                               on="SAMPLE_ID", how="inner")
        surv["event"] = surv["OS_STATUS"].astype(str).str.startswith("1")
        surv["time"] = pd.to_numeric(surv["OS_MONTHS"], errors="coerce")
        surv = surv.dropna(subset=["time"])

        kmf = KaplanMeierFitter()
        fig, ax = plt.subplots(figsize=(6, 5))
        for label, grp in surv.groupby("STAG2_altered"):
            kmf.fit(grp["time"], event_observed=grp["event"],
                    label=f"STAG2 altered={label} (n={len(grp)})")
            kmf.plot(ax=ax)

        altered_grp = surv[surv.STAG2_altered]
        unaltered_grp = surv[~surv.STAG2_altered]
        lr = logrank_test(
            altered_grp["time"], unaltered_grp["time"],
            event_observed_A=altered_grp["event"], event_observed_B=unaltered_grp["event"],
        )
        ax.set_title(f"Overall survival by {GENE} status\n(log-rank p={lr.p_value:.3g})")
        ax.set_xlabel("Months")
        ax.set_ylabel("Survival probability")
        plt.tight_layout()
        plt.savefig("stag2_survival_km.png", dpi=150)
        print(f"Log-rank p-value: {lr.p_value:.4g}")
        print("Saved plot to stag2_survival_km.png")

print("\nDone.")
print(f"Full run log saved to {LOG_TXT}")
sys.stdout = sys.__stdout__  # restore normal stdout
_log_file.close()

# ------------------------------------------------------------------------
# NEXT STEPS (not automated here):
# - Co-occurrence / mutual exclusivity with other genes (e.g. ASXL1, SRSF2,
#   U2AF1, RUNX1, or other cohesin members RAD21/SMC1A/SMC3) requires a
#   multi-gene query -- re-run the cBioPortal download with those genes
#   included, then merge each gene's "has_mutation" column and run a
#   Fisher's exact test per gene pair (scipy.stats.fisher_exact).
# - If you want to pull additional genes or studies without re-downloading
#   manually each time, cBioPortal's REST API (https://www.cbioportal.org/api)
#   can be queried directly from Python with `requests` -- happy to set
#   that up if this becomes a recurring workflow.
# ------------------------------------------------------------------------