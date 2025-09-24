import os
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from db_utils import load_taxi_data, preprocess_data

#Load Data
df = load_taxi_data()

#Preprocessing funcation call
df = preprocess_data(df)

result_folder = "../results"
os.makedirs(result_folder, exist_ok=True)

# Encode location IDs
df['pulocationid'] = df['pulocationid'].astype('category').cat.codes
df['dolocationid'] = df['dolocationid'].astype('category').cat.codes

#Defining X and Y
features = ['trip_distance', 'passenger_count', 'pulocationid', 'dolocationid', 'hour', 'day_of_week', 'is_rush_hour']
X = df[features]
y = df['trip_duration']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#Model Definition
xg = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=200,
    learning_rate=0.001,
    max_depth=6,
    seed=42
)

#Model Training
xg.fit(X_train, y_train)
y_pred = xg.predict(X_test)

#Evaluation
rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)
print(f"Trip Duration Prediction RMSE: {rmse:.2f}")
print(f"Trip Duration Prediction R2: {r2:.2f}")









