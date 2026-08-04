import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# 1. LOAD AND PREPROCESS DATA
# -----------------------------------------------------------------------------
# Load data (keeping NA as literal missing values)
df = pd.read_csv("final_combined_data.csv")

# Clean strings and handle explicit missing indicators
df["mutations"] = (
    df["mutations"].replace(["NP", "", np.nan], "WT/NP").fillna("WT/NP")
)
df["variants"] = (
    df["variants"].replace(["NP", "", np.nan], "WT/NP").fillna("WT/NP")
)

# Binary mutation targets
df["Altered"] = df["Altered"].fillna(0).astype(int)
df["STAG2"] = df["STAG2"].fillna(0).astype(int)

# Copy Number Alteration (CNA): default missing to 0 (Diploid)
# df["CNA"] = df["CNA"].fillna(0).astype(int)

# AFTER (Fixes 'NP')
df["CNA"] = pd.to_numeric(df["CNA"], errors="coerce").fillna(0).astype(int)

# -----------------------------------------------------------------------------
# 2. FEATURE SELECTION & IMPUTATION FOR CONTINUOUS COLUMNS
# -----------------------------------------------------------------------------
numeric_cols = [
    "log2",
    "mRNA_RSEM",
    "mRNA_Diploid",
    "mRNA_Normal",
    "mRNA_all",
    "protein_RPPA",
    "protein_z_RPPA",
    "protein_level_CPTAC",
    "protein_z_CPTAC",
]

# Keep numeric columns present in the dataframe
numeric_cols = [c for c in numeric_cols if c in df.columns]

# Drop columns with more than 70% missingness to avoid heavy noise
missing_pct = df[numeric_cols].isna().mean()
valid_numeric_cols = missing_pct[missing_pct < 0.7].index.tolist()

print(f"Retained continuous features: {valid_numeric_cols}")

# Drop 'Altered' column
df = df.drop(columns=["STAG2"])

# KNN Imputation for remaining continuous features
imputer = KNNImputer(n_neighbors=10)
df_imputed_numeric = pd.DataFrame(
    imputer.fit_transform(df[valid_numeric_cols]), columns=valid_numeric_cols
)

df_imputed_numeric.to_csv("knn_imputation_data.csv");

# -----------------------------------------------------------------------------
# 3. CORRELATION MATRIX (Expression vs. Copy Number)
# -----------------------------------------------------------------------------
corr_features = valid_numeric_cols + ["CNA", "Altered"]
corr_matrix = df.join(df_imputed_numeric, rsuffix="_imp")[
    corr_features
].corr(method="spearman")

print("\n--- Spearman Correlation with STAG2 Alteration Status ---")
if "Altered" in corr_matrix.columns:
    print(corr_matrix["Altered"].sort_values(ascending=False))

    plt.figure(figsize=(10, 8), dpi=300)  # High DPI for crisp PNG output

# 3. Draw heatmap
sns.heatmap(
    corr_matrix,
    annot=True,  # Display correlation values inside cells
    fmt=".2f",  # Round values to 2 decimal places
    cmap="coolwarm",  # Red-blue diverging colormap
    vmin=-1,
    vmax=1,  # Force bounds between -1 and +1
    square=True,  # Make cells square
    linewidths=0.5,  # Add subtle grid lines between cells
    cbar_kws={"shrink": 0.8},
)

plt.title("Feature Correlation Matrix", fontsize=14, pad=12)
plt.tight_layout()

# 4. Save directly as PNG
plt.savefig("correlation_matrix.png", dpi=300, bbox_inches="tight")
plt.show()

# -----------------------------------------------------------------------------
# 4. MACHINE LEARNING PREDICTION PIPELINE
# -----------------------------------------------------------------------------
# Define features (X) and target (y: STAG2 Mutation / Altered status)
X = df_imputed_numeric
y = df["Altered"]

# Check target balance
print(f"\nTarget distribution (Altered = 1 vs 0):\n{y.value_counts()}")

if len(y.unique()) > 1 and y.value_counts().min() > 1:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train Random Forest
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    # Predictions
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]

    print("\n--- Model Evaluation ---")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_prob):.3f}")

    # Feature Importance
    feature_importance = pd.Series(
        clf.feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\n--- Feature Importances ---")
    print(feature_importance)
else:
    print(
        "\nNot enough altered sample variations in this subset to train a classifier."
    )