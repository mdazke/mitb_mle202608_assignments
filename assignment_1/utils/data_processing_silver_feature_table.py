from datetime import datetime

import pyspark.sql.functions as F

from pyspark.sql.functions import col
from pyspark.sql.types import StringType, IntegerType, FloatType, DateType


def process_silver_clickstream_table(snapshot_date_str, bronze_clickstream_directory, silver_clickstream_directory, spark):
    # connect to bronze table
    partition_name = "bronze_clickstream_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_clickstream_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    column_type_map = {"Customer_ID": StringType(), "snapshot_date": DateType()}
    for i in range(1, 21):
        column_type_map["fe_" + str(i)] = IntegerType()

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # save silver table - IRL connect to database to write
    partition_name = "silver_clickstream_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_clickstream_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_attributes_table(snapshot_date_str, bronze_attributes_directory, silver_attributes_directory, spark):
    # connect to bronze table
    partition_name = "bronze_attributes_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_attributes_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: drop PII, Name and SSN carry no predictive value and should not travel downstream
    df = df.drop("Name", "SSN")

    # clean data: strip stray "_" from numeric fields (e.g. "40_")
    df = df.withColumn("Age", F.regexp_replace(col("Age"), "_", ""))

    # clean data: enforce schema / data type
    column_type_map = {
        "Customer_ID": StringType(),
        "Age": IntegerType(),
        "Occupation": StringType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        df = df.withColumn(column, col(column).cast(new_type))

    # clean data: values outside the valid range are set to null (not imputed, imputation is left to the model pipeline)
    valid_range_map = {"Age": (14, 100)}
    for column, (low, high) in valid_range_map.items():
        df = df.withColumn(column, F.when(col(column).between(low, high), col(column)))

    # clean data: values outside the known categories (e.g. "_______") are set to null
    valid_category_map = {
        "Occupation": ["Lawyer", "Architect", "Engineer", "Accountant", "Scientist", "Teacher", "Media_Manager", "Mechanic",
                       "Developer", "Entrepreneur", "Journalist", "Doctor", "Musician", "Manager", "Writer"],
    }
    for column, categories in valid_category_map.items():
        df = df.withColumn(column, F.when(col(column).isin(categories), col(column)))

    # save silver table - IRL connect to database to write
    partition_name = "silver_attributes_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_attributes_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df


def process_silver_financials_table(snapshot_date_str, bronze_financials_directory, silver_financials_directory, spark):
    # connect to bronze table
    partition_name = "bronze_financials_" + snapshot_date_str.replace('-','_') + '.csv'
    filepath = bronze_financials_directory + partition_name
    df = spark.read.csv(filepath, header=True, inferSchema=False)
    print('loaded from:', filepath, 'row count:', df.count())

    # clean data: enforce schema / data type
    # Dictionary specifying columns and their desired datatypes
    column_type_map = {
        "Customer_ID": StringType(),
        "Annual_Income": FloatType(),
        "Monthly_Inhand_Salary": FloatType(),
        "Num_Bank_Accounts": IntegerType(),
        "Num_Credit_Card": IntegerType(),
        "Interest_Rate": IntegerType(),
        "Num_of_Loan": IntegerType(),
        "Type_of_Loan": StringType(),
        "Delay_from_due_date": IntegerType(),
        "Num_of_Delayed_Payment": IntegerType(),
        "Changed_Credit_Limit": FloatType(),
        "Num_Credit_Inquiries": IntegerType(),
        "Credit_Mix": StringType(),
        "Outstanding_Debt": FloatType(),
        "Credit_Utilization_Ratio": FloatType(),
        "Credit_History_Age": StringType(),
        "Payment_of_Min_Amount": StringType(),
        "Total_EMI_per_month": FloatType(),
        "Amount_invested_monthly": FloatType(),
        "Payment_Behaviour": StringType(),
        "Monthly_Balance": FloatType(),
        "snapshot_date": DateType(),
    }

    for column, new_type in column_type_map.items():
        if isinstance(new_type, (IntegerType, FloatType)):
            # strip stray "_" (e.g. "52312.68_", "__10000__"); a bare "_" becomes empty and casts to null
            df = df.withColumn(column, F.regexp_replace(col(column), "_", ""))
        if isinstance(new_type, IntegerType):
            # some counts are stored as decimals (e.g. "11.0")
            df = df.withColumn(column, col(column).cast(FloatType()))
        df = df.withColumn(column, col(column).cast(new_type))

    # clean data: values outside the valid range (or placeholder values such as 10000 / -3.3e26) are set to null
    # (not imputed, imputation is left to the model pipeline so it is fit on training data only)
    valid_range_map = {
        "Annual_Income": (0, 200000),
        "Num_Bank_Accounts": (0, 10),
        "Num_Credit_Card": (0, 11),
        "Interest_Rate": (0, 34),
        "Num_of_Loan": (0, 9),
        "Num_of_Delayed_Payment": (0, 30),
        "Num_Credit_Inquiries": (0, 17),
        "Total_EMI_per_month": (0, 2000),
        "Amount_invested_monthly": (0, 5000),
        "Monthly_Balance": (0, 100000),
    }
    for column, (low, high) in valid_range_map.items():
        df = df.withColumn(column, F.when(col(column).between(low, high), col(column)))

    # clean data: values outside the known categories (e.g. "_", "!@9#%8") are set to null
    valid_category_map = {
        "Credit_Mix": ["Good", "Standard", "Bad"],
        "Payment_of_Min_Amount": ["Yes", "No", "NM"],
        "Payment_Behaviour": ["Low_spent_Small_value_payments", "Low_spent_Medium_value_payments", "Low_spent_Large_value_payments",
                              "High_spent_Small_value_payments", "High_spent_Medium_value_payments", "High_spent_Large_value_payments"],
    }
    for column, categories in valid_category_map.items():
        df = df.withColumn(column, F.when(col(column).isin(categories), col(column)))

    # augment data: credit history age in months, e.g. "10 Years and 9 Months" -> 129
    df = df.withColumn("Credit_History_Months",
                       (F.regexp_extract(col("Credit_History_Age"), r"(\d+) Years", 1).cast(IntegerType()) * 12
                        + F.regexp_extract(col("Credit_History_Age"), r"(\d+) Months", 1).cast(IntegerType())).cast(IntegerType()))
    df = df.drop("Credit_History_Age")

    # save silver table - IRL connect to database to write
    partition_name = "silver_financials_" + snapshot_date_str.replace('-','_') + '.parquet'
    filepath = silver_financials_directory + partition_name
    df.write.mode("overwrite").parquet(filepath)
    print('saved to:', filepath)

    return df
