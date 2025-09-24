import os
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import seaborn as sns
from db_utils import load_taxi_data, preprocess_data

#Load Data
df = load_taxi_data()


#Preprocessing funcation call
df = preprocess_data(df)

result_folder = "../results"
os.makedirs(result_folder, exist_ok=True)

# Encode location IDs
df['PUlocationID'] = df['pulocationid'].astype('category').cat.codes
df['DOlocationID'] = df['dolocationid'].astype('category').cat.codes


# Features for anomaly detection
features = ['trip_distance', 'fare_amount', 'trip_duration', 'passenger_count', 'pulocationid', 'dolocationid', 'hour']

iso = IsolationForest(contamination=0.01, random_state=42)
df['anomaly'] = iso.fit_predict(df[features])

# -1 = anomaly, 1 = normal
anomalies = df[df['anomaly'] == -1]
print("Number of detected anomalies:", len(anomalies))

#Visualize anomalies

plt.figure(figsize=(10,5))
sns.scatterplot(data=df, x='trip_distance', y='fare_amount', hue='anomaly', palette={1:'blue', -1:'red'}, alpha=0.6)
plt.title("Anomaly Detection: Fare vs Trip Distance")
plt.savefig(os.path.join(result_folder, "visualize_anamolies.png"))
plt.show()