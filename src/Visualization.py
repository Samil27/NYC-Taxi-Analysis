import os
import matplotlib.pyplot as plt
import seaborn as sns
from db_utils import load_taxi_data, preprocess_data

#Load Data
df = load_taxi_data()

#Preprocessing funcation call
df = preprocess_data(df)


result_folder = "../results"
os.makedirs(result_folder, exist_ok=True)

# Trip Distance Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['trip_distance'], bins=100, kde=True)
plt.xlim(0, 30)  # cap outliers for better view
plt.title("Distribution of Trip Distances (miles)")
plt.xlabel("Distance (miles)")
plt.ylabel("Count")
plt.savefig(os.path.join(result_folder, "trip_distance_distribution.png"))
plt.show()

# Trip Duration Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['trip_duration'], bins=100, kde=True)
plt.xlim(0, 120)
plt.title("Distribution of Trip Duration (minutes)")
plt.xlabel("Duration (minutes)")
plt.ylabel("Count")
plt.savefig(os.path.join(result_folder, "trip_duration_distribution.png"))
plt.show()

# Fare vs Distance
plt.figure(figsize=(8,5))
sns.scatterplot(x='trip_distance', y='fare_amount', data=df, alpha=0.3)
plt.xlim(0, 30)
plt.ylim(0, 100)
plt.title("Fare Amount vs Trip Distance")
plt.xlabel("Trip Distance (miles)")
plt.ylabel("Fare Amount ($)")
plt.savefig(os.path.join(result_folder, "fare_vs_distance.png"))
plt.show()

#Demand by hour of day
df['hour'] = df['pickup_datetime'].dt.hour

plt.figure(figsize=(10,5))
sns.countplot(x='hour', data=df, palette="viridis")
plt.title("Number of Trips by Hour of Day")
plt.xlabel("Hour")
plt.ylabel("Number of Trips")
plt.savefig(os.path.join(result_folder, "demand_by_hour.png"))
plt.show()



#Aggregate Trips & Revenue by Day
# Extract date
df['date'] = df['tpep_pickup_datetime'].dt.date

# Aggregate
daily_stats = df.groupby('date').agg({
    'total_amount': 'sum',      # total revenue per day
    'trip_distance': 'mean',    # avg trip distance per day
    'trip_duration': 'mean',    # avg trip duration per day
    'passenger_count': 'mean'   # avg passengers per day
}).reset_index()

daily_stats.head()


#Plot Daily Revenue Trend
plt.figure(figsize=(12,5))
sns.lineplot(x='date', y='total_amount', data=daily_stats)
plt.title("Daily Revenue from NYC Taxi Trips")
plt.xlabel("Date")
plt.ylabel("Total Revenue ($)")
plt.xticks(rotation=45)
plt.savefig(os.path.join(result_folder, "daily_revenue_trend.png"))
plt.show()


#Plot Average Trip Distance & Duration Over Time
plt.figure(figsize=(12,5))
sns.lineplot(x='date', y='trip_distance', data=daily_stats, label="Avg Trip Distance (miles)")
sns.lineplot(x='date', y='trip_duration', data=daily_stats, label="Avg Trip Duration (minutes)")
plt.title("Average Trip Distance & Duration Over Time")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.xticks(rotation=45)
plt.savefig(os.path.join(result_folder, "average_distance.png"))
plt.show()


#Hourly Demand Heatmap (Optional Extra)
df['hour'] = df['tpep_pickup_datetime'].dt.hour

# Pivot table
hourly_trips = df.pivot_table(index='hour', columns='date', values='total_amount', aggfunc='count')

plt.figure(figsize=(15,6))
sns.heatmap(hourly_trips, cmap="YlGnBu")
plt.title("Hourly Taxi Trips Heatmap")
plt.xlabel("Date")
plt.ylabel("Hour of Day")
plt.savefig(os.path.join(result_folder, "demand_heatmap.png"))
plt.show()
