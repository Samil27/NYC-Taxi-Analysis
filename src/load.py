from sqlalchemy import create_engine
import pandas as pd

TRANSFORMED_DATA_PATH = "data/transformed/yellow_tripdata_2024-01.parquet"
DB_URI = "postgresql+psycopg2://postgres:2710@localhost:5432/nyc_taxi"

# Read Parquet as Pandas DataFrame (can batch for large files)
df = pd.read_parquet(TRANSFORMED_DATA_PATH)

engine = create_engine(DB_URI)
df.to_sql("yellow_taxi_trips", engine, if_exists="replace", index=False)
print("Data loaded to PostgreSQL successfully!")
