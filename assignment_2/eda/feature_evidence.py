"""Feature-selection evidence, computed on the TRAINING WINDOW ONLY (2023 applications).

Run:  python eda/feature_evidence.py
Writes eda/charts/feature_signal.png, feature_signal.csv and feature_overlap.csv.

This is evidence for a human decision. It does not transform or select anything in the pipeline;
the actual imputing / scaling / selection is fitted later inside the training pipeline on train rows only.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eda", "charts")
os.makedirs(OUT, exist_ok=True)

TRAIN_START, TRAIN_END = "2023-01-01", "2023-12-01"   # application months used for training
NEW = ["repayment_ability", "balance_to_debt", "inquiries_per_loan", "num_loan_types"]
DROP = {"loan_id", "Customer_ID", "snapshot_date", "label"}


def read(path):
    files = glob.glob(os.path.join(ROOT, path, "*.parquet", "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


features = read("datamart/gold/feature_store")
labels = read("datamart/gold/label_store")[["loan_id", "label"]]
df = features.merge(labels, on="loan_id", how="inner")          # join on loan_id only
df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
df = df[(df["snapshot_date"] >= TRAIN_START) & (df["snapshot_date"] <= TRAIN_END)]
print(f"training window: {len(df):,} applications, default rate {df['label'].mean():.1%}")

num_cols = [c for c in df.columns if c not in DROP and pd.api.types.is_numeric_dtype(df[c])]
cat_cols = [c for c in df.columns if c not in DROP and c not in num_cols]

# ---- 1. signal: how well each numeric feature alone ranks defaulters (direction-agnostic AUC, nulls ignored)
rows = []
for c in num_cols:
    s = df[[c, "label"]].dropna()
    if s[c].nunique() < 2:
        continue
    auc = roc_auc_score(s["label"], s[c])
    rows.append({"feature": c, "auc": auc, "signal": abs(auc - 0.5) * 2,
                 "direction": "higher = riskier" if auc > 0.5 else "higher = safer",
                 "null_pct": df[c].isna().mean() * 100, "engineered": c in NEW})
sig = pd.DataFrame(rows).sort_values("signal", ascending=False).reset_index(drop=True)
sig.to_csv(os.path.join(OUT, "feature_signal.csv"), index=False)

# ---- 2. categorical features: default rate by category
print("\nCategorical default rates (2023):")
for c in cat_cols:
    t = df.groupby(c, dropna=False)["label"].agg(["size", "mean"]).round(3)
    print(f"\n{c}\n{t.to_string()}")

# ---- 3. overlap: pairs of numeric features that move together
corr = df[num_cols].corr(method="spearman")
pairs = (corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack()
         .rename("spearman").reset_index().rename(columns={"level_0": "a", "level_1": "b"}))
pairs["abs"] = pairs["spearman"].abs()
pairs = pairs.sort_values("abs", ascending=False)
pairs.head(25).to_csv(os.path.join(OUT, "feature_overlap.csv"), index=False)

print("\nTop 15 features by single-feature signal (0 = none, 1 = perfect):")
print(sig.head(15)[["feature", "signal", "direction", "null_pct", "engineered"]].round(3).to_string(index=False))
print("\nNew features:")
print(sig[sig["engineered"]][["feature", "signal", "direction", "null_pct"]].round(3).to_string(index=False))
print("\nWeakest 8:")
print(sig.tail(8)[["feature", "signal"]].round(3).to_string(index=False))
print("\nMost overlapping pairs (|Spearman| >= 0.7):")
print(pairs[pairs["abs"] >= 0.7].head(15).round(2).to_string(index=False))

# ---- chart
SURFACE, INK, INK2, GRID, BLUE, GREY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#b9b8b2"
top = sig.head(15).iloc[::-1]
fig, ax = plt.subplots(figsize=(7.2, 4.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
ax.barh(top["feature"], top["signal"], color=[BLUE if e else GREY for e in top["engineered"]], height=0.62)
for y, v in enumerate(top["signal"]):
    ax.text(v + 0.004, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK2)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(GRID)
ax.tick_params(length=0, colors=INK2)
ax.xaxis.grid(True, color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
ax.set_xlabel("Single-feature signal  (2 × |AUC − 0.5|; 0 = none)", color=INK2)
fig.suptitle("Which features separate defaulters, using 2023 applications only", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=INK)
fig.text(0.01, 0.915, "Blue = engineered in gold for this assignment; grey = existing columns", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.9))
fig.savefig(os.path.join(OUT, "feature_signal.png"), dpi=200, facecolor=SURFACE)
print("\nsaved to", OUT)
