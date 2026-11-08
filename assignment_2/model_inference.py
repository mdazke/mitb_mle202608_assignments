"""Score one application month with a model from the model bank and save the predictions to the gold layer.

Usage:  python model_inference.py --snapshotdate 2024-03-01 --modelname credit_model_2024_07_01

One run = one application month, so Airflow can backfill month by month.

Output: datamart/gold/model_predictions/<modelname>/<modelname>_predictions_YYYY_MM_DD.parquet
        loan_id, Customer_ID, snapshot_date, model_name, model_predictions (probability of default),
        scored_in_training_window (True if the model was trained/tested on this month, so charts can mark it)
"""
import argparse
import os
import pickle

import pandas as pd

MODEL_BANK = "model_bank"
FEATURE_STORE = "datamart/gold/feature_store"
PREDICTIONS_DIR = "datamart/gold/model_predictions"


def main(snapshotdate, modelname):
    print("\n--- starting inference job ---\n")
    modelname = modelname[:-4] if modelname.endswith(".pkl") else modelname
    snapshot = pd.Timestamp(snapshotdate)

    # --- load the artefact from the model bank ---
    with open(os.path.join(MODEL_BANK, modelname + ".pkl"), "rb") as f:
        artefact = pickle.load(f)
    print("loaded model:", artefact["model_version"], "| feature set:", artefact["feature_set"])

    # --- load the features for this application month only ---
    partition = os.path.join(FEATURE_STORE, "gold_feature_store_" + snapshotdate.replace("-", "_") + ".parquet")
    features = pd.concat([pd.read_parquet(os.path.join(partition, f)) for f in os.listdir(partition) if f.endswith(".parquet")],
                         ignore_index=True)
    features["snapshot_date"] = pd.to_datetime(features["snapshot_date"])
    assert (features["snapshot_date"] == snapshot).all(), "partition contains a different snapshot_date"
    print("applications to score:", len(features))

    # --- predict: the saved Pipeline applies the training-fitted preprocessing itself ---
    scores = artefact["model"].predict_proba(features[artefact["feature_columns"]])[:, 1]

    # --- flag months the model has already seen (train/test/OOT windows) ---
    d = artefact["data_dates"]
    in_window = d["train_test_start"] <= snapshot <= d["oot_end"]

    out = features[["loan_id", "Customer_ID", "snapshot_date"]].copy()
    out["snapshot_date"] = out["snapshot_date"].dt.date
    out["model_name"] = artefact["model_version"]
    out["model_predictions"] = scores
    out["scored_in_training_window"] = bool(in_window)

    # --- save to the gold layer ---
    directory = os.path.join(PREDICTIONS_DIR, modelname)
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"{modelname}_predictions_{snapshotdate.replace('-', '_')}.parquet")
    out.to_parquet(path, index=False)
    print(f"saved {len(out)} predictions to {path} | mean score {scores.mean():.3f} | in training window: {in_window}")
    print("\n--- completed inference job ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="score one application month")
    parser.add_argument("--snapshotdate", type=str, required=True, help="application month, YYYY-MM-DD")
    parser.add_argument("--modelname", type=str, required=True, help="model name in model_bank (with or without .pkl)")
    args = parser.parse_args()
    main(args.snapshotdate, args.modelname)
