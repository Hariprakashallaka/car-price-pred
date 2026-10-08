import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score


# 1. Load dataset
df = pd.read_csv("data/cars.csv")

print(df.head())


# 2. Separate features and target
X = df[
    [
        "car_age",
        "km_driven",
        "engine_cc",
        "previous_owners"
    ]
]

y = df["price_lakhs"]


# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# 4. Create model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


# 5. Train model
model.fit(X_train, y_train)


# 6. Make predictions
predictions = model.predict(X_test)


# 7. Evaluate
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("MAE:", mae)
print("R2 Score:", r2)


# 8. Save model
joblib.dump(model, "model/used_car_model.pkl")

print("Model saved successfully!")