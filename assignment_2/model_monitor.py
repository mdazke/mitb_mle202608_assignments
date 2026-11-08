"""Monitor a deployed model "as of" one run date and write the results to the gold layer.

Usage:  python model_monitor.py --snapshotdate 2024-08-01 --modelname credit_model_2024_01_01

What is known on run date T (this is a live simulation):
  * STABILITY  for the applications scored this month (cohort T). No labels needed, so available immediately.
      - score PSI : score distribution vs the training scores stored in the model artefact
      - CSI       : PSI of every model feature vs the training-window feature distribution (incl. missing values)
  * PERFORMANCE for the applications whose label has just matured (cohort T-6): label date = application + 6 months,
      so the label store partition of date T holds exactly cohort T-6. AUC, Gini, default rate, calibration gap.

Bin edges always come from the training baseline, never from the month being tested.

Outputs (one file per run date):
  datamart/gold/model_monitoring/<model>/<model>_monitoring_YYYY_MM_DD.parquet        one row, headline metrics + status
  datamart/gold/model_monitoring_features/<model>/<model>_feature_csi_YYYY_MM_DD.parquet   one row per feature
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.metrics import roc_auc_score

MODEL_BANK = "model_bank"
FEATURE_STORE = "datamart/gold/feature_store"
LABEL_STORE = "datamart/gold/label_store"
PREDICTIONS_DIR = "datamart/gold/model_predictions"
MONITOR_DIR = "datamart/gold/model_monitoring"
FEATURE_CSI_DIR = "datamart/gold/model_monitoring_features"

N_BINS = 10
EPS = 1e-4
LABEL_MOB = 6

# ---- thresholds (proposed; to be agreed with the bank's risk team) ----
PSI_AMBER, PSI_RED = 0.10, 0.25                  # score PSI and per-feature CSI
AUC_GREEN, AUC_AMBER = 0.75, 0.70                # cohort AUC: >= green, >= amber, else red


def read_partition(directory, name):
    """Read a Spark-style (folder of part files) or single-file parquet partition; empty frame if missing."""
    path = os.path.join(directory, name)
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in os.listdir(path) if f.endswith(".parquet")]
        return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if os.path.isfile(path):
        return pd.read_parquet(path)
    return pd.DataFrame()


def psi_from_counts(base_counts, cur_counts):
    b = np.asarray(base_counts, dtype=float)
    c = np.asarray(cur_counts, dtype=float)
    b = np.maximum(b / b.sum(), EPS)
    c = np.maximum(c / c.sum(), EPS)
    return float(np.sum((c - b) * np.log(c / b)))


def numeric_bins(baseline):
    """Decile edges from the BASELINE only; a separate bin (the last) holds missing values."""
    edges = np.unique(np.nanquantile(baseline.dropna().to_numpy(dtype=float), np.linspace(0, 1, N_BINS + 1)))
    return edges[1:-1]


def numeric_counts(values, inner_edges):
    v = pd.Series(values)
    counts = np.bincount(np.searchsorted(inner_edges, v.dropna().to_numpy(dtype=float), side="left"),
                         minlength=len(inner_edges) + 1)
    return np.append(counts, v.isna().sum())


def categorical_counts(values, categories):
    v = pd.Series(values).astype(object).where(pd.Series(values).notna(), "__missing__")
    v = v.where(v.isin(categories), "__other__")
    return np.array([(v == c).sum() for c in list(categories) + ["__other__"]])


def feature_psi(base, cur):
    """PSI of one feature, current vs baseline. Returns (psi, null_rate_baseline, null_rate_current)."""
    if pd.api.types.is_numeric_dtype(base):
        edges = numeric_bins(base)
        b, c = numeric_counts(base, edges), numeric_counts(cur, edges)
    else:
        cats = list(pd.Series(base).astype(object).where(pd.Series(base).notna(), "__missing__").unique())
        b, c = categorical_counts(base, cats), categorical_counts(cur, cats)
    return psi_from_counts(b + 1e-9, c + 1e-9), float(base.isna().mean()), float(cur.isna().mean())


def status(value, amber, red, higher_is_worse=True):
    if value is None or pd.isna(value):
        return None
    if higher_is_worse:
        return "red" if value > red else ("amber" if value > amber else "green")
    return "green" if value >= amber else ("amber" if value >= red else "red")


def partition_name(prefix, date_str, ext=".parquet"):
    return f"{prefix}_{date_str.replace('-', '_')}{ext}"


def main(snapshotdate, modelname):
    print("\n--- starting monitoring job ---\n")
    modelname = modelname[:-4] if modelname.endswith(".pkl") else modelname
    run_date = pd.Timestamp(snapshotdate)
    with open(os.path.join(MODEL_BANK, modelname + ".pkl"), "rb") as f:
        artefact = pickle.load(f)
    dates, feature_cols = artefact["data_dates"], artefact["feature_columns"]

    # ---------------- stability: cohort T (applications scored this month) ----------------
    preds = read_partition(os.path.join(PREDICTIONS_DIR, modelname),
                           partition_name(f"{modelname}_predictions", snapshotdate))
    row = {"snapshot_date": run_date.date(), "model_name": artefact["model_version"],
           "stability_cohort": run_date.date(), "n_scored": len(preds)}
    csi_rows = []
    if len(preds):
        base_scores = np.asarray(artefact["reference_scores"], dtype=float)
        score_edges = np.unique(np.quantile(base_scores, np.linspace(0, 1, N_BINS + 1)))[1:-1]
        row["score_psi"] = psi_from_counts(np.bincount(np.searchsorted(score_edges, base_scores, side="left"), minlength=len(score_edges) + 1) + 1e-9,
                                           np.bincount(np.searchsorted(score_edges, preds["model_predictions"].to_numpy(), side="left"), minlength=len(score_edges) + 1) + 1e-9)
        row["mean_score"] = float(preds["model_predictions"].mean())
        row["mean_score_baseline"] = float(base_scores.mean())

        # feature baseline: gold feature store, training-window application months
        months = pd.date_range(dates["train_test_start"], dates["train_test_end"], freq="MS")
        base_df = pd.concat([read_partition(FEATURE_STORE, partition_name("gold_feature_store", m.strftime("%Y-%m-%d")))
                             for m in months], ignore_index=True)
        cur_df = read_partition(FEATURE_STORE, partition_name("gold_feature_store", snapshotdate))
        for col in feature_cols:
            psi, null_base, null_cur = feature_psi(base_df[col], cur_df[col])
            csi_rows.append({"snapshot_date": run_date.date(), "model_name": artefact["model_version"], "feature": col,
                             "csi": psi, "null_rate_baseline": null_base, "null_rate_current": null_cur,
                             "status": status(psi, PSI_AMBER, PSI_RED)})
        csi = pd.DataFrame(csi_rows).sort_values("csi", ascending=False)
        row.update(csi_max=float(csi["csi"].iloc[0]), csi_max_feature=csi["feature"].iloc[0],
                   n_features_amber=int((csi["status"] == "amber").sum()), n_features_red=int((csi["status"] == "red").sum()))
    row["status_score_psi"] = status(row.get("score_psi"), PSI_AMBER, PSI_RED)
    row["status_csi"] = status(row.get("csi_max"), PSI_AMBER, PSI_RED)

    # ---------------- performance: cohort T-6 (labels matured today) ----------------
    cohort = run_date - relativedelta(months=LABEL_MOB)
    row["performance_cohort"] = cohort.date()
    cohort_preds = read_partition(os.path.join(PREDICTIONS_DIR, modelname),
                                  partition_name(f"{modelname}_predictions", cohort.strftime("%Y-%m-%d")))
    labels = read_partition(LABEL_STORE, partition_name("gold_label_store", snapshotdate))
    if len(cohort_preds) and len(labels):
        joined = cohort_preds.merge(labels[["loan_id", "label"]], on="loan_id", how="inner")
        row["n_labelled"] = len(joined)
        row["default_rate"] = float(joined["label"].mean())
        row["mean_score_cohort"] = float(joined["model_predictions"].mean())
        row["calibration_gap"] = row["mean_score_cohort"] - row["default_rate"]
        row["auc"] = float(roc_auc_score(joined["label"], joined["model_predictions"]))
        row["gini"] = 2 * row["auc"] - 1
    row["status_performance"] = status(row.get("auc"), AUC_GREEN, AUC_AMBER, higher_is_worse=False)

    # ---------------- save ----------------
    out_dir = os.path.join(MONITOR_DIR, modelname)
    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame([row]).to_parquet(os.path.join(out_dir, partition_name(f"{modelname}_monitoring", snapshotdate)), index=False)
    if csi_rows:
        csi_dir = os.path.join(FEATURE_CSI_DIR, modelname)
        os.makedirs(csi_dir, exist_ok=True)
        pd.DataFrame(csi_rows).to_parquet(os.path.join(csi_dir, partition_name(f"{modelname}_feature_csi", snapshotdate)), index=False)

    fmt = lambda k, p=3: ("n/a" if row.get(k) is None or pd.isna(row.get(k)) else f"{row[k]:.{p}f}")
    print(f"run {snapshotdate} | scored {row['n_scored']} | score PSI {fmt('score_psi')} ({row['status_score_psi']}) | "
          f"worst CSI {fmt('csi_max')} {row.get('csi_max_feature', '')} ({row['status_csi']}) | "
          f"cohort {cohort.date()}: AUC {fmt('auc')} ({row['status_performance']}), n={row.get('n_labelled', 0)}")
    print("\n--- completed monitoring job ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="monitor a deployed model as of a run date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="run date, YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True)
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
