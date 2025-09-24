import pandas as pd
from sqlalchemy import create_engine

def get_engine():
    return create_engine("postgresql+psycopg2://postgres:2710@localhost:5432/nyc_taxi")

def load_taxi_data():
    engine = get_engine()
    query = """
    SELECT 
        tpep_pickup_datetime,
        tpep_dropoff_datetime,
        passenger_count,
        trip_distance,
        fare_amount,
        tip_amount,
        total_amount,
        pulocationid,
        dolocationid
    FROM yellow_taxi_trips;
    """
    return pd.read_sql(query,engine)

def preprocess_data(df):
    #Convert date and time to datetime
    df['pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    #Calculate trip duration
    df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60

    #Exact hour and day of week
    df['hour'] = df['tpep_pickup_datetime'].dt.hour
    df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek

    #Calculate rush hour
    df['is_rush_hour'] = df['hour'].apply(lambda x: 1 if 7 <= x <= 10 or 16 <= x <= 19 else 0)

    return df