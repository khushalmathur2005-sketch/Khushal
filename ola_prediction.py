import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("ola_ride_dataset_1000.csv")

print("Dataset loaded successfully!")
print(df.head())
print("\nColumns:")
print(df.columns.tolist())

# Remove rows with missing values
df = df.dropna()

# Target column
target = "Fare_INR"

# Features
X = df.drop(columns=[target])
y = df[target]

# Separate categorical and numerical columns
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# Model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n----- MODEL RESULT -----")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 2))

# Predict first 5 test rides
print("\nActual vs Predicted:")
for actual, predicted in zip(y_test.head(5), y_pred[:5]):
    print("Actual:", round(actual, 2),
          "| Predicted:", round(predicted, 2))