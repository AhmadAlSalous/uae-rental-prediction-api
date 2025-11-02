import pandas as pd
import joblib

# Load your dataset (the same one you used for training)
data = pd.read_csv("app/uae_rental_data.csv")
data.columns = data.columns.str.strip()

# Keep only the columns you used for training
cols_to_use = ["Beds", "Baths", "Area_in_sqft", "City", "Type", "Frequency", "Furnishing", "Location", "Rent"]
data = data[cols_to_use].dropna()

# Encode categorical columns exactly like you did during training
categorical_cols = ["City","Type","Frequency","Furnishing","Location"]
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# Save the column names used for features (exclude target)
X_columns = list(data.drop("Rent", axis=1).columns)

# Save to a file
joblib.dump(X_columns, "app/columns.pkl")
print("Columns saved successfully!")
