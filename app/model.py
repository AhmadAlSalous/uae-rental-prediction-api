import pandas as pd 
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import numpy as np
from scipy.stats import randint, uniform

# Optional imports for gradient boosting models
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

# -----------------------------
# Load and preprocess data
# -----------------------------
data = pd.read_csv("app/uae_rental_data.csv")
data.columns = data.columns.str.strip()

cols_to_use = ["Beds", "Baths", "Area_in_sqft", "City", "Type", "Frequency", "Furnishing", "Location", "Rent"]
data = data[cols_to_use]

# Drop missing values
data = data.dropna()

# Encode categorical variables
categorical_cols = ["City", "Type", "Frequency", "Furnishing", "Location"]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Split features and target
X = data.drop("Rent", axis=1)
y = data["Rent"]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -----------------------------
# Define baseline models
# -----------------------------
models = {
    "RandomForest": RandomForestRegressor(
        n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=300, max_depth=5, random_state=42
    ),
    "XGBoost": XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8,
        colsample_bytree=0.8, random_state=42, n_jobs=-1
    ),
    "LightGBM": LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=-1,
        num_leaves=31, random_state=42, n_jobs=-1
    ),
    "CatBoost": CatBoostRegressor(
        iterations=500, learning_rate=0.05, depth=8, random_seed=42,
        silent=True
    )
}

# -----------------------------
# Train and evaluate all models
# -----------------------------
results = []

print("\n📊 Training baseline models and evaluating performance...\n")

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    # Metrics
    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    
    # Define a pseudo accuracy (within ±20%)
    tolerance = 0.2
    accuracy = np.mean(np.abs(preds - y_test) <= (tolerance * y_test)) * 100

    results.append({
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Accuracy(±20%)": accuracy
    })
    print(f"{name} done ✅\n")

# -----------------------------
# Hyperparameter tuning for top 2 models
# -----------------------------
print("\n⚙️ Starting hyperparameter tuning for top models (RandomForest & GradientBoosting)...\n")

# ---- RandomForest tuning ----
rf_param_dist = {
    "n_estimators": randint(200, 600),
    "max_depth": randint(10, 40),
    "min_samples_split": randint(2, 10),
    "min_samples_leaf": randint(1, 5),
    "max_features": ["sqrt", "log2", None]
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42, n_jobs=-1),
    rf_param_dist,
    n_iter=20,
    scoring="r2",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_

print(f"\n✅ Best RandomForest params: {rf_search.best_params_}")

# ---- GradientBoosting tuning ----
gb_param_dist = {
    "n_estimators": randint(200, 600),
    "max_depth": randint(3, 10),
    "learning_rate": uniform(0.01, 0.2),
    "subsample": uniform(0.7, 0.3)
}

gb_search = RandomizedSearchCV(
    GradientBoostingRegressor(random_state=42),
    gb_param_dist,
    n_iter=20,
    scoring="r2",
    cv=3,
    verbose=1,
    random_state=42,
    n_jobs=-1
)
gb_search.fit(X_train, y_train)
best_gb = gb_search.best_estimator_

print(f"\n✅ Best GradientBoosting params: {gb_search.best_params_}")

# -----------------------------
# Evaluate tuned models
# -----------------------------
tuned_models = {
    "RandomForest_Tuned": best_rf,
    "GradientBoosting_Tuned": best_gb
}

print("\n🎯 Evaluating tuned models...\n")

for name, model in tuned_models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    r2 = r2_score(y_test, preds)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    tolerance = 0.2
    accuracy = np.mean(np.abs(preds - y_test) <= (tolerance * y_test)) * 100

    results.append({
        "Model": name,
        "R2": r2,
        "MAE": mae,
        "RMSE": rmse,
        "Accuracy(±20%)": accuracy
    })
    print(f"{name} done ✅\n")

# -----------------------------
# Final comparison table
# -----------------------------
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(by="R2", ascending=False).reset_index(drop=True)

print("\n====================== Model Performance Comparison ======================\n")
print(results_df.to_string(index=False, formatters={
    "R2": "{:.4f}".format,
    "MAE": "{:,.2f}".format,
    "RMSE": "{:,.2f}".format,
    "Accuracy(±20%)": "{:.2f}%".format
}))
print("\n==========================================================================\n")

# -----------------------------
# Save the best model
# -----------------------------
best_model_name = results_df.iloc[0]["Model"]
best_model = None
if best_model_name in tuned_models:
    best_model = tuned_models[best_model_name]
else:
    best_model = models[best_model_name]

joblib.dump(best_model, f"app/best_rental_model_{best_model_name}.pkl")
joblib.dump(list(X.columns), "app/model_columns.pkl")

print(f"🏆 Best model: {best_model_name} (saved as app/best_rental_model_{best_model_name}.pkl)")
print("\nTraining and tuning complete.\n")

# -----------------------------
# Optional: Inspect processed data
# -----------------------------
print(data.head())
print(data.columns)
