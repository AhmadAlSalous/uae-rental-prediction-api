from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
import os

app = FastAPI(title="UAE Rental Prediction API")

# -----------------------------
# Paths
# -----------------------------
MODEL_PATH = "app/best_rental_model_GradientBoosting.pkl"
COLUMNS_PATH = "app/model_columns.pkl"
DATA_PATH = "app/uae_rental_data.csv"

# -----------------------------
# Load model and columns
# -----------------------------
model = joblib.load(MODEL_PATH)
columns = joblib.load(COLUMNS_PATH)

# -----------------------------
# Load and clean dataset
# -----------------------------
df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()

# Strip whitespace and standardize capitalization
str_cols = ["City", "Type", "Frequency", "Furnishing", "Location"]
for col in str_cols:
    df[col] = df[col].astype(str).str.strip().str.title()

# Drop rows with missing essential info
df = df.dropna(subset=str_cols)

# -----------------------------
# Input schema
# -----------------------------
class House(BaseModel):
    beds: int
    baths: float
    area: int
    city: str
    type: str
    frequency: str
    furnishing: str
    location: str

# -----------------------------
# Serve frontend
# -----------------------------
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join("app", "static", "index.html"))

# -----------------------------
# Dynamic / dependent dropdown options
# -----------------------------
@app.get("/options")
def get_options(city: str = None):
    data = df.copy()
    if city:
        city = city.strip().title()
        data = data[data["City"] == city]

    options = {
        "City": sorted(df["City"].unique()),
        "Type": sorted(data["Type"].unique()),
        "Frequency": sorted(data["Frequency"].unique()),
        "Furnishing": sorted(data["Furnishing"].unique()),
        "Location": sorted(data["Location"].unique())
    }
    return options

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
def predict_rent(house: House):
    x = pd.DataFrame(data=np.zeros((1, len(columns))), columns=columns)

    # Numeric features
    x.loc[0, "Beds"] = house.beds
    x.loc[0, "Baths"] = house.baths
    x.loc[0, "Area_in_sqft"] = house.area

    # Categorical features
    cat_map = {
        "City": house.city.strip().title(),
        "Type": house.type.strip().title(),
        "Frequency": house.frequency.strip().title(),
        "Furnishing": house.furnishing.strip().title(),
        "Location": house.location.strip().title()
    }

    for key, val in cat_map.items():
        col_name = f"{key}_{val}"
        if col_name in x.columns:
            x.loc[0, col_name] = 1

    pred = model.predict(x)[0]
    return {"predicted_rent": round(float(pred), 2)}
