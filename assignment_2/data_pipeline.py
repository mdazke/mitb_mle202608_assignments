"""Run the bronze -> silver -> gold data pipeline for ONE snapshot month.

Usage:
  python data_pipeline.py --store label   --snapshotdate 2024-03-01
  python data_pipeline.py --store feature --snapshotdate 2024-03-01

  label   : loan management system -> gold/label_store   (default = 30+ days past due at month 6)
  feature : attributes + financials -> gold/feature_store (one row per loan application)

Airflow calls this once per store per month ({{ ds }}), so a backfill is just many runs of the same code.
"""
import argparse
import os

import pyspark

import utils.data_processing_bronze_feature_table as bronze_feature
import utils.data_processing_bronze_table as bronze_lms
import utils.data_processing_gold_feature_table as gold_feature
import utils.data_processing_gold_table as gold_label
import utils.data_processing_silver_feature_table as silver_feature
import utils.data_processing_silver_table as silver_lms

DPD, MOB = 30, 6   # label definition: 30+ days past due at month-on-book 6

FEATURE_SOURCES = {                       # clickstream is deliberately not used (see README / deck)
    "attributes": "data/features_attributes.csv",
    "financials": "data/features_financials.csv",
}


def make_dirs(*paths):
    for p in paths:
        os.makedirs(p, exist_ok=True)


def run_label_store(date_str, spark):
    bronze, silver, gold = "datamart/bronze/lms/", "datamart/silver/loan_daily/", "datamart/gold/label_store/"
    make_dirs(bronze, silver, gold)
    bronze_lms.process_bronze_table(date_str, bronze, spark)
    silver_lms.process_silver_table(date_str, bronze, silver, spark)
    gold_label.process_labels_gold_table(date_str, silver, gold, spark, dpd=DPD, mob=MOB)


def run_feature_store(date_str, spark):
    bronze = {name: f"datamart/bronze/{name}/" for name in FEATURE_SOURCES}
    silver_attr, silver_fin = "datamart/silver/attributes/", "datamart/silver/financials/"
    gold = "datamart/gold/feature_store/"
    make_dirs(*bronze.values(), silver_attr, silver_fin, gold)
    for name, csv_path in FEATURE_SOURCES.items():
        bronze_feature.process_bronze_feature_table(date_str, name, csv_path, bronze[name], spark)
    silver_feature.process_silver_attributes_table(date_str, bronze["attributes"], silver_attr, spark)
    silver_feature.process_silver_financials_table(date_str, bronze["financials"], silver_fin, spark)
    gold_feature.process_features_gold_table(date_str, silver_attr, silver_fin, gold, spark)


def main(store, snapshotdate):
    print(f"\n--- data pipeline: {store} store, snapshot {snapshotdate} ---\n")
    spark = (pyspark.sql.SparkSession.builder.appName("data_pipeline").master("local[*]")
             .config("spark.ui.enabled", "false").config("spark.sql.shuffle.partitions", "4").getOrCreate())
    spark.sparkContext.setLogLevel("ERROR")
    try:
        {"label": run_label_store, "feature": run_feature_store}[store](snapshotdate, spark)
    finally:
        spark.stop()
    print(f"\n--- completed {store} store for {snapshotdate} ---\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run the data pipeline for one snapshot month")
    parser.add_argument("--store", choices=["label", "feature"], required=True)
    parser.add_argument("--snapshotdate", type=str, required=True, help="YYYY-MM-DD (first of month)")
    args = parser.parse_args()
    main(args.store, args.snapshotdate)
