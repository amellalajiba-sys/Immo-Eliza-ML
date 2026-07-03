import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from preprocessing import preprocessor
import joblib


def main():

    # Load the cleaned dataset
    df = pd.read_csv("data/cleaned_scraped_properties.csv")

    # Remove features that are not useful for price prediction
    df = df.drop(columns=["property_id"])
    df = df.drop(columns=["postal_code"])
    df = df.drop(columns=["city"])

    # Separate the target variable from the input features
    y = df["price"]
    X = df.drop("price", axis=1)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # Linear Regression model
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    # Random Forest model
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(
            n_estimators=50,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # XGBoost model
    xgb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            random_state=42
        ))
    ])

    # Train a machine learning model
    def train_model(model, X_train, y_train):
        model.fit(X_train, y_train)
        return model

    model = train_model(model, X_train, y_train)
    rf_model = train_model(rf_model, X_train, y_train)
    xgb_model = train_model(xgb_model, X_train, y_train)

    # Evaluate the model using MAE, MSE and R²
    def evaluate_model(model, X_test, y_test, name):

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)

        print(f"\n{name}")
        print(f"MAE : {mae:.2f}")
        print(f"MSE : {mse:.2f}")
        print(f"R² : {r2:.3f}")

    evaluate_model(model, X_test, y_test, "Linear Regression")
    evaluate_model(rf_model, X_test, y_test, "Random Forest")
    evaluate_model(xgb_model, X_test, y_test, "XGBoost")

    # Save the trained models for future predictions
    joblib.dump(xgb_model, "models/xgboost_model.pkl")
    joblib.dump(rf_model, "models/random_forest_model.pkl")
    joblib.dump(model, "models/linear_regression_model.pkl")


if __name__ == "__main__":
    main()