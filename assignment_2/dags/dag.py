"""Monthly ML pipeline: data -> (train once) -> inference -> monitoring.

One DAG run = one snapshot month ({{ ds }}). With catchup=True Airflow backfills every month from START_DATE,
one run at a time, in order. A "live" simulation:

  * every month   : build the label store and the feature store (bronze -> silver -> gold)
  * on TRAIN_DATE : train the model using only labels already known on that date, save it to the model bank
  * from SCORING_START : score that month's new applications with the model and write them to the gold layer
  * after scoring      : monitor stability (score PSI, feature CSI) of this month's applications and performance
                         (AUC, calibration) of the cohort whose 6-month label has just matured, then write to gold
"""
from datetime import datetime, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import ShortCircuitOperator

TRAIN_DATE = "2024-01-01"                      # model is trained as of this date, then deployed
SCORING_START = "2024-01-01"                   # first month scored in "production"
MODEL_NAME = "credit_model_" + TRAIN_DATE.replace("-", "_")
PROJECT = "/opt/airflow/project"               # the repo, mounted into the container

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=1),
}


def run(script_and_args):
    return f"cd {PROJECT} && python3 {script_and_args}"


def is_valid_snapshot(ds, **_):
    # a manual "Trigger DAG" uses today's date, which has no data: skip such runs instead of failing
    return "2023-01-01" <= ds <= "2025-01-01" and ds.endswith("-01")


def is_train_date(ds, **_):
    return ds == TRAIN_DATE


def is_scoring_month(ds, **_):
    return ds >= SCORING_START


with DAG(
    "ml_pipeline",
    default_args=default_args,
    description="monthly data pipeline, one-off training, monthly inference",
    schedule="0 0 1 * *",                      # 00:00 on the 1st of every month
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2025, 1, 1),
    catchup=True,                              # backfill every month since start_date
    max_active_runs=1,                         # months run in order, so the model exists before it is used
) as dag:

    start = EmptyOperator(task_id="start")
    window_gate = ShortCircuitOperator(task_id="is_valid_snapshot_month", python_callable=is_valid_snapshot)

    # ---- data pipeline: medallion architecture, one task per store ----
    label_store = BashOperator(
        task_id="label_store_bronze_silver_gold",
        bash_command=run('data_pipeline.py --store label --snapshotdate "{{ ds }}"'),
    )
    feature_store = BashOperator(
        task_id="feature_store_bronze_silver_gold",
        bash_command=run('data_pipeline.py --store feature --snapshotdate "{{ ds }}"'),
    )
    data_ready = EmptyOperator(task_id="data_ready")

    # ---- training: only on TRAIN_DATE. Other months skip it (and that skip must not block inference) ----
    train_gate = ShortCircuitOperator(
        task_id="is_train_month",
        python_callable=is_train_date,
        ignore_downstream_trigger_rules=False,
    )
    train_model = BashOperator(
        task_id="train_model",
        bash_command=run('model_train.py --snapshotdate "{{ ds }}"'),
    )

    # ---- inference: only from SCORING_START onwards ----
    scoring_gate = ShortCircuitOperator(
        task_id="is_scoring_month",
        python_callable=is_scoring_month,
        ignore_downstream_trigger_rules=False,
    )
    model_inference = BashOperator(
        task_id="model_inference",
        bash_command=run(f'model_inference.py --snapshotdate "{{{{ ds }}}}" --modelname {MODEL_NAME}'),
        trigger_rule="none_failed",            # train_model is skipped in most months; that is fine
    )
    model_monitor = BashOperator(
        task_id="model_monitor",
        bash_command=run(f'model_monitor.py --snapshotdate "{{{{ ds }}}}" --modelname {MODEL_NAME}'),
    )
    end = EmptyOperator(task_id="end", trigger_rule="none_failed")

    start >> window_gate >> [label_store, feature_store] >> data_ready
    data_ready >> train_gate >> train_model >> model_inference
    data_ready >> scoring_gate >> model_inference
    model_inference >> model_monitor >> end
