from datetime import datetime

from pyspark.sql.functions import col


def process_bronze_feature_table(snapshot_date_str, source_name, csv_file_path, bronze_feature_directory, spark):
    # prepare arguments
    snapshot_date = datetime.strptime(snapshot_date_str, "%Y-%m-%d")

    # load data - IRL ingest from back end source system
    # every column is read as raw text (no inferSchema) so bronze stays exactly as received, dirty values included
    df = spark.read.csv(csv_file_path, header=True, inferSchema=False).filter(col('snapshot_date') == snapshot_date_str)
    print(snapshot_date_str + ' ' + source_name + ' row count:', df.count())

    # save bronze table to datamart - IRL connect to database to write
    partition_name = "bronze_" + source_name + "_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_feature_directory + partition_name
    df.toPandas().to_csv(filepath, index=False)
    print('saved to:', filepath)

    return df
