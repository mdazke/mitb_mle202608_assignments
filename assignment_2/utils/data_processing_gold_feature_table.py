import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType


def process_features_gold_table(snapshot_date_str, silver_attributes_directory, silver_financials_directory, silver_clickstream_directory, gold_feature_store_directory, spark):
    # snapshot_date is the loan application date. Every source is read from the SAME snapshot partition only,
    # so nothing dated after the application (e.g. later clickstream months) can leak into the features.
    suffix = snapshot_date_str.replace('-','_') + '.parquet'

    # connect to silver tables
    filepath = silver_attributes_directory + "silver_attributes_" + suffix
    df_attributes = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_attributes.count())

    filepath = silver_financials_directory + "silver_financials_" + suffix
    df_financials = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_financials.count())

    filepath = silver_clickstream_directory + "silver_clickstream_" + suffix
    df_clickstream = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_clickstream.count())

    # combine: one row per loan application. Clickstream only exists for some customers, so left join and flag it
    df_clickstream = df_clickstream.drop("snapshot_date").withColumn("has_clickstream", F.lit(1).cast(IntegerType()))
    df = df_attributes.join(df_financials.drop("snapshot_date"), on="Customer_ID", how="inner")
    df = df.join(df_clickstream, on="Customer_ID", how="left")
    df = df.fillna(0, subset=["has_clickstream"])

    # augment data: join key to the label store, same format as loan_id in the loan management system
    df = df.withColumn("loan_id", F.concat(col("Customer_ID"), F.lit("_"), F.date_format(col("snapshot_date"), "yyyy_MM_dd")))

    # augment data: one flag per loan type
    loan_types = ["Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan", "Home Equity Loan", "Mortgage Loan",
                  "Not Specified", "Payday Loan", "Personal Loan", "Student Loan"]
    for loan_type in loan_types:
        flag_name = "loan_type_" + loan_type.lower().replace(' ', '_').replace('-', '_')
        df = df.withColumn(flag_name, F.when(col("Type_of_Loan").contains(loan_type), 1).otherwise(0).cast(IntegerType()))
    df = df.drop("Type_of_Loan")

    # augment data: split payment behaviour into spending level and payment size
    df = df.withColumn("spend_level", F.split(col("Payment_Behaviour"), "_").getItem(0))
    df = df.withColumn("payment_size", F.split(col("Payment_Behaviour"), "_").getItem(2))
    df = df.drop("Payment_Behaviour")

    # augment data: affordability ratios (null when the denominator is null or 0)
    ratio_map = {
        "debt_to_income": ("Outstanding_Debt", "Annual_Income"),
        "emi_to_salary": ("Total_EMI_per_month", "Monthly_Inhand_Salary"),
        "invested_to_salary": ("Amount_invested_monthly", "Monthly_Inhand_Salary"),
        "balance_to_salary": ("Monthly_Balance", "Monthly_Inhand_Salary"),
    }
    for ratio_name, (numerator, denominator) in ratio_map.items():
        df = df.withColumn(ratio_name, F.when(col(denominator) > 0, col(numerator) / col(denominator)).cast(FloatType()))

    # order columns: keys first, then attributes, financials, clickstream
    key_columns = ["loan_id", "Customer_ID", "snapshot_date"]
    df = df.select(key_columns + [c for c in df.columns if c not in key_columns])

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + suffix
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
