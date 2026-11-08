"""EDA charts for the label definition and the train / monitor windows.

Run from anywhere:  python eda/eda_charts.py
Reads the silver loan table and the gold label store; writes PNGs + CSVs to eda/charts/.

Only descriptive statistics are produced here. Nothing is fitted, and no model sees these numbers.
"""
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "eda", "charts")
os.makedirs(OUT, exist_ok=True)

DPD, MOB = 30, 6

# palette (reference instance, light surface)
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, GREY = "#2a78d6", "#b9b8b2"

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
})


def read_parquet_dir(path):
    files = glob.glob(os.path.join(ROOT, path, "*.parquet", "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(length=0)


# ---------------------------------------------------------------- chart 1: default rate by months on book
loans = read_parquet_dir("datamart/silver/loan_daily")
loans["bad"] = (loans["dpd"] >= DPD).astype(int)

# fixed cohort: only loans observed through mob 10, so the curve is not distorted by loans dropping out
complete_ids = loans.loc[loans["mob"] == 10, "loan_id"].unique()
cohort = loans[loans["loan_id"].isin(complete_ids)]
by_mob = (cohort.groupby("mob")
          .agg(loans=("loan_id", "nunique"), pct_30dpd=("bad", "mean"))
          .reset_index())
by_mob.to_csv(os.path.join(OUT, "default_rate_by_mob.csv"), index=False)

fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.plot(by_mob["mob"], by_mob["pct_30dpd"] * 100, color=BLUE, linewidth=2, marker="o", markersize=6,
        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
ax.axvline(MOB, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
final = by_mob["pct_30dpd"].iloc[-1] * 100
ax.annotate(f"Month {MOB}: {final:.1f}%\nrate stops changing", xy=(MOB, final), xytext=(MOB - 0.25, final - 12.5),
            ha="right", color=INK, fontsize=10, arrowprops=dict(arrowstyle="-", color=INK2, linewidth=0.8))
ax.set_xticks(range(0, 11))
ax.set_xlim(-0.3, 10.3)
ax.set_ylim(0, 35)
ax.set_xlabel("Months on book (instalments since loan start)")
ax.set_ylabel(f"% of loans {DPD}+ days past due")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
style(ax)
fig.suptitle(f"The {DPD}dpd rate climbs for 5 months, then flattens at month {MOB}", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=INK)
fig.text(0.01, 0.90, f"{len(complete_ids):,} loans observed for the full 10 months", ha="left", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(os.path.join(OUT, "default_rate_by_mob.png"), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- chart 2: default rate by application month
labels = read_parquet_dir("datamart/gold/label_store")
features = read_parquet_dir("datamart/gold/feature_store")[["loan_id", "snapshot_date"]]
features = features.rename(columns={"snapshot_date": "application_month"})
features["application_month"] = pd.to_datetime(features["application_month"])

# join on loan_id ONLY: label_store.snapshot_date is the label date (application + 6 months)
df = features.merge(labels[["loan_id", "label"]], on="loan_id", how="left")
by_app = (df.groupby("application_month")
          .agg(applications=("loan_id", "size"), labelled=("label", "count"), default_rate=("label", "mean"))
          .reset_index())
by_app.to_csv(os.path.join(OUT, "default_rate_by_application_month.csv"), index=False)

lab = by_app[by_app["labelled"] > 0]
unl = by_app[by_app["labelled"] == 0]
overall = df["label"].mean() * 100

fig, ax = plt.subplots(figsize=(7.2, 3.9))
split = pd.Timestamp("2024-01-01")
ax.axvspan(lab["application_month"].min() - pd.Timedelta(days=15), split - pd.Timedelta(days=15),
           color=BLUE, alpha=0.07, linewidth=0)
ax.axvspan(unl["application_month"].min() - pd.Timedelta(days=15), unl["application_month"].max() + pd.Timedelta(days=15),
           facecolor="none", edgecolor="#d6d5cf", hatch="///", linewidth=0)
ax.axhline(overall, color=INK2, linewidth=1, linestyle=(0, (4, 3)), zorder=2)
ax.plot(lab["application_month"], lab["default_rate"] * 100, color=BLUE, linewidth=2, marker="o", markersize=6,
        markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3)
ymax = 42
ax.text(pd.Timestamp("2023-01-10"), ymax - 2.5, "2023: training pool", color=INK2, fontsize=9, va="top")
ax.text(pd.Timestamp("2024-01-10"), ymax - 2.5, "Jan–Jun 2024:\nmonitoring window", color=INK2, fontsize=9, va="top")
ax.text(pd.Timestamp("2024-10-01"), ymax - 2.5, "Jul 2024 on:\nno label yet", color=INK2, fontsize=9, va="top", ha="center")
ax.text(pd.Timestamp("2025-01-20"), overall - 2.2, f"avg {overall:.1f}%", color=INK2, fontsize=9, ha="right", va="top",
        bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
ax.set_ylim(0, ymax)
ax.set_xlim(pd.Timestamp("2022-12-15"), pd.Timestamp("2025-01-25"))
ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 4, 7, 10)))
ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b\n%Y"))
ax.set_xlabel("Application month")
ax.set_ylabel("% of applications that defaulted")
ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0f}%")
style(ax)
fig.suptitle("Default rate is steady around 29% across application months", x=0.01, ha="left",
             fontsize=12, fontweight="bold", color=INK)
fig.text(0.01, 0.90, f"{int(lab['labelled'].sum()):,} labelled of {len(df):,} applications; label = {DPD}+ dpd at month {MOB}",
         ha="left", fontsize=9, color=INK2)
fig.tight_layout(rect=(0, 0, 1, 0.89))
fig.savefig(os.path.join(OUT, "default_rate_by_application_month.png"), dpi=200)
plt.close(fig)

print(by_mob.round(3).to_string(index=False))
print(by_app.round(3).to_string(index=False))
print("saved to", OUT)
