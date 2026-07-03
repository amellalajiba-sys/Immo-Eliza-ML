import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

df = pd.read_csv("data/cleaned_scraped_properties.csv")
df = df.drop(columns=["property_id"])
df = df.drop(columns=["postal_code"])
df = df.drop(columns=["city"])
#Here I'm dropping the columns "property_id", "postal_code", and "city" from the dataset because they are not relevant for predicting the price of a property.
y = df["price"]
#What the model needs to learn how to predict the price of a property based on the other features in the dataset. 
#The target variable is the price of the property, which is what we want to predict. 
X = df.drop("price", axis=1)

# Builds the preprocessing pipeline by:
# - selecting numerical and categorical features,
# - standardizing numerical data,
# - one-hot encoding categorical data,
# - combining both transformations into a reusable preprocessor.

def build_preprocessor(X):

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor

# Create the preprocessing pipeline used during model training and prediction.
preprocessor = build_preprocessor(X)