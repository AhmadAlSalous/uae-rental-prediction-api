import pandas as pd 
from sklearn.linear_model import LinearRegression
import joblib

data = pd.read_csv("app/data.csv")
X = data[["bedrooms", "bathrooms", "sqft_living"]]
Y = data[["price"]]

model = LinearRegression()
model.fit(X,Y)

##Saving the model to a file 
joblib.dump(model,"app/house_price_model.pkl")
print("model trained and saved successfully!")
