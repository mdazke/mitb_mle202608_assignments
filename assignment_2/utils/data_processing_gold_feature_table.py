import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import IntegerType, FloatType


def process_features_gold_table(snapshot_date_str, silver_attributes_directory, silver_financials_directory, gold_feature_store_directory, spark):
    # snapshot_date is the loan application date. Every source is read from the SAME snapshot partition only,
    # so nothing dated after the application can leak into the features.
    # Clickstream is deliberately NOT used: it exists only for customers with labels, so every application from
    # Jul 2024 onward would arrive without it and could not be scored consistently.
    suffix = snapshot_date_str.replace('-','_') + '.parquet'

    # connect to silver tables
    filepath = silver_attributes_directory + "silver_attributes_" + suffix
    df_attributes = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_attributes.count())

    filepath = silver_financials_directory + "silver_financials_" + suffix
    df_financials = spark.read.parquet(filepath)
    print('loaded from:', filepath, 'row count:', df_financials.count())

    # combine: one row per loan application
    df = df_attributes.join(df_financials.drop("snapshot_date"), on="Customer_ID", how="inner")

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

    # augment data: a few interpretable affordability / borrowing-behaviour features.
    # Row-wise arithmetic only: nothing is learned from the wider dataset, so it is safe to build in gold.
    # repayment_ability: salary left after monthly loan repayments (null if either input is null)
    df = df.withColumn("repayment_ability", (col("Monthly_Inhand_Salary") - col("Total_EMI_per_month")).cast(FloatType()))
    # balance_to_debt: monthly cash left over relative to the debt owed (null when debt is null or 0)
    df = df.withColumn("balance_to_debt", F.when(col("Outstanding_Debt") > 0, col("Monthly_Balance") / col("Outstanding_Debt")).cast(FloatType()))
    # inquiries_per_loan: credit-seeking per loan held; +1 keeps customers with 0 loans defined instead of null
    df = df.withColumn("inquiries_per_loan", (col("Num_Credit_Inquiries") / (col("Num_of_Loan") + 1)).cast(FloatType()))
    # num_loan_types: how many distinct (specified) loan types the customer holds
    typed_flags = [c for c in df.columns if c.startswith("loan_type_") and c != "loan_type_not_specified"]
    df = df.withColumn("num_loan_types", sum([col(c) for c in typed_flags]).cast(IntegerType()))

    # order columns: keys first, then attributes, financials
    key_columns = ["loan_id", "Customer_ID", "snapshot_date"]
    df = df.select(key_columns + [c for c in df.columns if c not in key_columns])

    # save gold table - IRL connect to database to write
    partition_name = "gold_feature_store_" + suffix
    filepath = gold_feature_store_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
