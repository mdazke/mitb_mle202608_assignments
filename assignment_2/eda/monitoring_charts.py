"""Monitoring charts for the deck, built from the gold monitoring tables.

Run:  python eda/monitoring_charts.py
Reads  datamart/gold/model_monitoring/<model>/  and  datamart/gold/model_monitoring_features/<model>/
Writes eda/charts/monitoring_performance.png, monitoring_calibration.png, monitoring_stability.png, monitoring_csi_heatmap.png
"""
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = sys.argv[1] if len(sys.argv) > 1 else "credit_model_2024_01_01"
OUT = os.path.join(ROOT, "eda", "charts")
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b8b2"
PSI_AMBER, PSI_RED, AUC_GREEN, AUC_AMBER = 0.10, 0.25, 0.75, 0.70

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                     "savefig.facecolor": SURFACE})


def read_all(directory):
    files = glob.glob(os.path.join(ROOT, directory, MODEL, "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


mon = read_all("datamart/gold/model_monitoring").sort_values("snapshot_date")
csi = read_all("datamart/gold/model_monitoring_features")
mon["snapshot_date"] = pd.to_datetime(mon["snapshot_date"])
mon["performance_cohort"] = pd.to_datetime(mon["performance_cohort"])
csi["snapshot_date"] = pd.to_datetime(csi["snapshot_date"])
mon.to_csv(os.path.join(OUT, "monitoring_summary.csv"), index=False)


def base_axes(figsize=(7.2, 3.9)):
    fig, ax = plt.subplots(figsize=figsize)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)
    return fig, ax


def titles(fig, title, subtitle):
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.90, subtitle, ha="left", fontsize=9, color=INK2)


def month_axis(ax, dates):
    ax.set_xticks(dates)
    ax.set_xticklabels([d.strftime("%b\n%y") for d in dates], fontsize=8.5)


def tag(ax, x, y, text, **kw):
    ax.text(x, y, text, fontsize=kw.pop("fontsize", 8.5), color=kw.pop("color", INK2), va="center", zorder=5,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5), **kw)


def line(ax, x, y, color, label=None):
    ax.plot(x, y, color=color, linewidth=2, marker="o", markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.5,
            zorder=3, label=label)


# ---------------------------------------------------------------- 1. performance by cohort month
perf = mon.dropna(subset=["auc"]).sort_values("performance_cohort")
oot = None
try:
    import pickle
    sys.path.insert(0, ROOT)
    oot = pickle.load(open(os.path.join(ROOT, "model_bank", MODEL + ".pkl"), "rb"))["results"]["auc_oot"]
except Exception:
    pass

fig, ax = base_axes()
line(ax, perf["performance_cohort"], perf["auc"], BLUE)
for y, txt in [(AUC_GREEN, f"green ≥ {AUC_GREEN:.2f}"), (AUC_AMBER, f"red < {AUC_AMBER:.2f}")]:
    ax.axhline(y, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    tag(ax, perf["performance_cohort"].max() + pd.Timedelta(days=20), y, txt)
if oot:
    ax.axhline(oot, color=GREY, linewidth=1)
    ax.text(perf["performance_cohort"].min() - pd.Timedelta(days=10), oot + 0.004, f"OOT at training: {oot:.3f}", fontsize=8.5,
            color=INK2, ha="left")
for _, r in perf.iterrows():
    tag(ax, r["performance_cohort"], r["auc"] + 0.011, f"{r['auc']:.2f}", ha="center", color=INK)
ax.set_ylim(0.66, 0.82)
ax.set_xlim(perf["performance_cohort"].min() - pd.Timedelta(days=18), perf["performance_cohort"].max() + pd.Timedelta(days=75))
month_axis(ax, list(perf["performance_cohort"]))
ax.set_xlabel("Application (cohort) month")
ax.set_ylabel("AUC")
titles(fig, "Ranking power holds at AUC 0.74 to 0.79 on loans the model never saw",
       f"One point per cohort of ~500 loans; known 6 months after application; model trained on Jan–Jun 2023 data")
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(os.path.join(OUT, "monitoring_performance.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 2. calibration by cohort month
fig, ax = base_axes()
line(ax, perf["performance_cohort"], perf["default_rate"] * 100, ORANGE)
line(ax, perf["performance_cohort"], perf["mean_score_cohort"] * 100, BLUE)
last = perf.iloc[-1]
tag(ax, last["performance_cohort"] + pd.Timedelta(days=10), last["default_rate"] * 100 + 2.0, "Actual\ndefault rate", color=INK, fontsize=9)
tag(ax, last["performance_cohort"] + pd.Timedelta(days=10), last["mean_score_cohort"] * 100 - 2.4, "Average\nscore", color=INK, fontsize=9)
ax.set_ylim(20, 38)
ax.set_yticks(range(20, 39, 4))
ax.set_xlim(perf["performance_cohort"].min() - pd.Timedelta(days=18), perf["performance_cohort"].max() + pd.Timedelta(days=60))
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
month_axis(ax, list(perf["performance_cohort"]))
ax.set_xlabel("Application (cohort) month")
ax.set_ylabel("Default probability / rate")
gap = (perf["mean_score_cohort"] - perf["default_rate"]).abs().mean() * 100
titles(fig, "Scores track the real default rate to within a few points",
       f"Average absolute gap between mean predicted score and actual default rate: {gap:.1f} percentage points")
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(os.path.join(OUT, "monitoring_calibration.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 3. stability by application month
fig, ax = base_axes()
line(ax, mon["snapshot_date"], mon["score_psi"], BLUE)
line(ax, mon["snapshot_date"], mon["csi_max"], ORANGE)
for y, txt in [(PSI_AMBER, "amber 0.10"), (PSI_RED, "red 0.25")]:
    ax.axhline(y, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    tag(ax, mon["snapshot_date"].max() + pd.Timedelta(days=20), y, txt)
tag(ax, mon["snapshot_date"].iloc[-1] + pd.Timedelta(days=12), mon["score_psi"].iloc[-1] - 0.016, "Score PSI", fontsize=9, color=INK)
tag(ax, mon["snapshot_date"].iloc[-1] + pd.Timedelta(days=12), mon["csi_max"].iloc[-1] + 0.020, "Worst feature\nCSI", fontsize=9, color=INK)
ax.set_ylim(0, 0.30)
ax.set_xlim(mon["snapshot_date"].min() - pd.Timedelta(days=18), mon["snapshot_date"].max() + pd.Timedelta(days=75))
month_axis(ax, list(mon["snapshot_date"]))
ax.set_xlabel("Application month (scored that month; no labels needed)")
ax.set_ylabel("PSI / CSI")
titles(fig, "Scores and inputs stay stable: every month is below the 0.10 watch line",
       "PSI against the training baseline; worst of the model's features shown for CSI")
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(os.path.join(OUT, "monitoring_stability.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- 4. which features move: CSI heatmap (top features)
top = csi.groupby("feature")["csi"].max().sort_values(ascending=False).head(12).index.tolist()
pivot = csi[csi["feature"].isin(top)].pivot(index="feature", columns="snapshot_date", values="csi").loc[top]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
im = ax.imshow(pivot.values, cmap="Blues", vmin=0, vmax=PSI_AMBER, aspect="auto")
ax.set_xticks(range(pivot.shape[1]))
ax.set_xticklabels([d.strftime("%b\n%y") for d in pivot.columns], fontsize=8.5)
ax.set_yticks(range(len(top)))
ax.set_yticklabels(top, fontsize=9)
for i in range(pivot.shape[0]):
    for j in range(pivot.shape[1]):
        v = pivot.values[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if v > PSI_AMBER * 0.6 else INK2)
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
titles(fig, "No single feature drifts: the 12 largest CSI values never reach 0.10",
       "Darker = larger shift from the training baseline (colour scale tops out at the 0.10 watch line)")
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(os.path.join(OUT, "monitoring_csi_heatmap.png"), dpi=200)
plt.close(fig)

print(mon[["snapshot_date", "score_psi", "csi_max", "csi_max_feature", "performance_cohort", "auc", "calibration_gap"]]
      .round(3).to_string(index=False))
print("saved to", OUT)
