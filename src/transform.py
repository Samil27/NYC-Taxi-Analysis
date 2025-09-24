import pandas as pd
import os

RAW_DATA_PATH = r"C:\Users\mitha\Projects\NYC_Taxi_Pipeline\data\raw\yellow_tripdata_2024-01.parquet"
TRANSFORMED_DIR = r"C:\Users\mitha\Projects\NYC_Taxi_Pipeline\data\transformed"
os.makedirs(TRANSFORMED_DIR, exist_ok=True)

df = pd.read_parquet(RAW_DATA_PATH)
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])
df.dropna(inplace=True)

df.to_parquet(os.path.join(TRANSFORMED_DIR, "yellow_tripdata_2024-01.parquet"), index=False)
print("ETL complete!")
