"""Train the loan-default model "as of" a training date and save it to the model bank.

Usage:  python model_train.py --snapshotdate 2024-07-01

Everything is derived from --snapshotdate, so the same script can be re-run for any date (Airflow, backfill).

Leakage controls
  * A loan's label is only known 6 months after application, so only applications whose label date is BEFORE the
    training date are used. Features and labels are joined on loan_id (never on snapshot_date).
  * The split is by time: the most recent labelled months are held out as OOT; train/test is a random split
    of the older months.
  * Imputing, clipping, scaling, encoding and tuning are fitted inside one sklearn Pipeline on training rows only
    (and refitted per fold during cross-validation).
"""
import argparse
import glob
import os
import pickle
import pprint
from datetime import datetime

import numpy as np
import pandas as pd
from dateutil.relativedelta import relativedelta
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from utils.ml_preprocessing import PercentileClipper

LABEL_MOB = 6                         # label is observed 6 months after application
NEW_FEATURES = ["repayment_ability", "balance_to_debt", "inquiries_per_loan", "num_loan_types"]
ID_COLUMNS = ["loan_id", "Customer_ID", "snapshot_date"]
MIN_GAIN = 0.005                      # keep the engineered features only if CV AUC improves by more than this
RANDOM_STATE = 88


def read_parquet_dir(directory):
    files = glob.glob(os.path.join(directory, "*.parquet", "*.parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def build_config(train_date_str, oot_months=2, train_test_months=12, train_test_ratio=0.8):
    train_date = datetime.strptime(train_date_str, "%Y-%m-%d")
    # label date = application month + 6. It must be before the training date to be known, so:
    last_labelled_app = train_date - relativedelta(months=LABEL_MOB + 1)
    oot_start = last_labelled_app - relativedelta(months=oot_months - 1)
    train_test_end = oot_start - relativedelta(months=1)
    train_test_start = oot_start - relativedelta(months=train_test_months)
    return {
        "model_train_date_str": train_date_str,
        "train_test_start": train_test_start, "train_test_end": train_test_end,
        "oot_start": oot_start, "oot_end": last_labelled_app,
        "train_test_ratio": train_test_ratio,
    }


def make_pipeline(numeric_cols, categorical_cols):
    numeric = Pipeline([
        ("clip", PercentileClipper()),
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer([("num", numeric, numeric_cols), ("cat", categorical, categorical_cols)])
    return Pipeline([("preprocess", preprocess), ("model", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE))])


def split_columns(df, feature_cols):
    numeric = [c for c in feature_cols if pd.api.types.is_numeric_dtype(df[c])]
    categorical = [c for c in feature_cols if c not in numeric]
    return numeric, categorical


def main(snapshotdate):
    print("\n--- starting training job ---\n")
    cfg = build_config(snapshotdate)
    pprint.pprint(cfg)

    # --- load and join on loan_id ---
    features = read_parquet_dir("datamart/gold/feature_store")
    labels = read_parquet_dir("datamart/gold/label_store")[["loan_id", "label"]]
    data = features.merge(labels, on="loan_id", how="inner")
    data["snapshot_date"] = pd.to_datetime(data["snapshot_date"])

    # --- time-based windows (on application month) ---
    in_oot = (data["snapshot_date"] >= cfg["oot_start"]) & (data["snapshot_date"] <= cfg["oot_end"])
    in_tt = (data["snapshot_date"] >= cfg["train_test_start"]) & (data["snapshot_date"] <= cfg["train_test_end"])
    oot, train_test = data[in_oot], data[in_tt]
    train, test = train_test_split(train_test, test_size=1 - cfg["train_test_ratio"],
                                   random_state=RANDOM_STATE, shuffle=True, stratify=train_test["label"])
    for name, part in [("train", train), ("test", test), ("oot", oot)]:
        print(f"{name:5s} n={len(part):5d} default_rate={part['label'].mean():.3f} "
              f"apps {part['snapshot_date'].min().date()} to {part['snapshot_date'].max().date()}")

    # --- candidate feature sets: base vs base + engineered ---
    all_features = [c for c in features.columns if c not in ID_COLUMNS]
    base_features = [c for c in all_features if c not in NEW_FEATURES]
    candidates = {"base": base_features, "base_plus_engineered": all_features}

    # --- tune the regularisation strength on TRAIN only (5-fold CV), same grid for both sets ---
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results, fitted = {}, {}
    for name, cols in candidates.items():
        num_cols, cat_cols = split_columns(train, cols)
        pipe = make_pipeline(num_cols, cat_cols)
        grid = GridSearchCV(pipe, {"model__C": [0.01, 0.1, 1.0, 10.0]}, scoring="roc_auc", cv=cv, n_jobs=-1)
        grid.fit(train[cols], train["label"])
        fold_scores = cross_val_score(grid.best_estimator_, train[cols], train["label"], scoring="roc_auc", cv=cv)
        results[name] = {"cv_auc_mean": float(fold_scores.mean()), "cv_auc_std": float(fold_scores.std()),
                         "best_C": grid.best_params_["model__C"], "n_features": len(cols)}
        fitted[name] = grid.best_estimator_
        print(f"{name:22s} CV AUC {fold_scores.mean():.4f} (+/- {fold_scores.std():.4f})  C={grid.best_params_['model__C']}")

    # --- choose the simpler set unless the engineered features help meaningfully ---
    gain = results["base_plus_engineered"]["cv_auc_mean"] - results["base"]["cv_auc_mean"]
    chosen = "base_plus_engineered" if gain > MIN_GAIN else "base"
    print(f"\nCV AUC gain from engineered features: {gain:+.4f} (threshold {MIN_GAIN}) -> chosen: {chosen}")
    cols, best_model = candidates[chosen], fitted[chosen]

    # --- evaluate on train / test / OOT ---
    scores = {}
    for name, part in [("train", train), ("test", test), ("oot", oot)]:
        scores[name] = roc_auc_score(part["label"], best_model.predict_proba(part[cols])[:, 1])
    for name, auc in scores.items():
        print(f"{name:5s} AUC {auc:.4f}  Gini {2 * auc - 1:.3f}")

    # --- model artefact ---
    model_version = "credit_model_" + snapshotdate.replace("-", "_")
    artefact = {
        "model": best_model,                       # whole Pipeline: preprocessing + logistic regression
        "model_version": model_version,
        "feature_columns": cols,
        "feature_set": chosen,
        "data_dates": cfg,
        "data_stats": {k: {"n": len(v), "default_rate": round(float(v["label"].mean()), 4)}
                       for k, v in [("train", train), ("test", test), ("oot", oot)]},
        "feature_set_comparison": results,
        "results": {"auc_" + k: v for k, v in scores.items()} | {"gini_" + k: 2 * v - 1 for k, v in scores.items()},
        "hp_params": {"C": results[chosen]["best_C"]},
        # reference for stability monitoring later: score distribution on the training rows
        "reference_scores": best_model.predict_proba(train[cols])[:, 1].astype(np.float32),
    }
    os.makedirs("model_bank", exist_ok=True)
    path = os.path.join("model_bank", model_version + ".pkl")
    with open(path, "wb") as f:
        pickle.dump(artefact, f)
    print(f"\nModel saved to {path}")

    # --- prove the saved artefact reloads and reproduces OOT performance ---
    with open(path, "rb") as f:
        loaded = pickle.load(f)
    reload_auc = roc_auc_score(oot["label"], loaded["model"].predict_proba(oot[loaded["feature_columns"]])[:, 1])
    assert abs(reload_auc - scores["oot"]) < 1e-9, "reloaded model does not reproduce OOT AUC"
    print(f"Reloaded model OOT AUC {reload_auc:.4f} (matches)")
    print("\n--- completed training job ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="train model as of a snapshot date")
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD")
    main(parser.parse_args().snapshotdate)
