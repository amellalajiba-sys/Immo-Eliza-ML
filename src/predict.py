import joblib
import pandas as pd

# Load the trained model
model_rf = joblib.load("models/random_forest_model.pkl")
model_xgb = joblib.load("models/xgboost_model.pkl")

# Create a new property
new_house = pd.DataFrame({
    "province": ["brussels"],
    "type_property": ["house"],
    "subtype_property": ["mansion"],
    "livable_surface": [180],
    "latitude": [50.85],
    "longitude": [4.35],
    "facades": [2],
    "bedrooms": [4],
    "bathrooms": [2],
    "toilets": [2],
    "terrace": [1],
    "garden": [1],
    "garage": [1],
    "swimming_pool": [0],
    "distance_from_train_stations_by_foot": [500],
    "distance_from_elementary_school_by_foot": [300],
    "distance_from_high_school_by_foot": [700],
    "state_of_property": ["to_be_renovated"],
    "heating_type": ["gas"],
    "sun_exposure": ["south"],
    "epc_score": ["B"],
    "flooding_area_type": ["no_flooding_area"]
})

# Make predictions using the trained models
prediction = model_rf.predict(new_house)
prediction_xgb = model_xgb.predict(new_house)

print(f"Predicted price (Random Forest): €{prediction[0]:,.2f}")
print(f"Predicted price (XGBoost): €{prediction_xgb[0]:,.2f}")