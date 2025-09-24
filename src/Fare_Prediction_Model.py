import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from db_utils import load_taxi_data, preprocess_data

#Load Data
df = load_taxi_data()


#Preprocessing funcation call
df = preprocess_data(df)

result_folder = "../results"
os.makedirs(result_folder, exist_ok=True)

features = ['trip_distance', 'passenger_count', 'pulocationid', 
            'dolocationid', 'hour']
X = df[features]
y = df['fare_amount']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

#XGBOOST model define and training
xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', n_estimators=100, seed=42)
xg_reg.fit(X_train, y_train)
y_pred_xgb = xg_reg.predict(X_test)

print("XGBoost RMSE:", mean_squared_error(y_test, y_pred_xgb, squared=False))
print("XGBoost R2:", r2_score(y_test, y_pred_xgb))


#Predicted vs actual
plt.figure(figsize=(8,5))
plt.scatter(y_test, y_pred_xgb, alpha=0.3)
plt.xlabel("Actual Fare")
plt.ylabel("Predicted Fare")
plt.title("XGBoost: Actual vs Predicted Fare")
plt.savefig(os.path.join(result_folder, "actual_vs_predicted_fare.png"))
plt.show()


# Get feature importances
importances = xg_reg.feature_importances_
feature_names = X.columns

# Create a DataFrame for plotting
feat_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Plot
plt.figure(figsize=(8,5))
plt.barh(feat_df['Feature'], feat_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()  # highest importance on top
plt.title("XGBoost Feature Importance for Fare Prediction")
plt.xlabel("Importance")
plt.savefig(os.path.join(result_folder, "imp_fetaures_fare_prediction.png"))
plt.show()

