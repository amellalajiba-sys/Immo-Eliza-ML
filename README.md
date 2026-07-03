# Immo-Eliza-ML

# Immo Eliza - Machine Learning Regression

## Project Description

This project was developed as part of the BeCode AI & Data Science training.

The objective is to build a machine learning model capable of predicting the selling price of real estate properties in Belgium using a cleaned dataset provided after the scraping, cleaning and analysis phases.

Three regression models were implemented and compared:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

The complete preprocessing pipeline is included inside the trained models, allowing predictions to be made directly on new properties.

---

# Repository Structure

```text
immo-eliza-ml/
│
├── data/
│   └── cleaned_scraped_properties.csv
│
├── models/
│   ├── linear_regression_model.pkl
│   ├── random_forest_model.pkl
│   └── xgboost_model.pkl
│
├── src/
│   ├── Hiba.ipynb
│   ├── predict.py
│   ├── preprocessing.py
│   └── train.py
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

# Dataset

The project uses a cleaned dataset containing information about Belgian properties.

The target variable is:

* **price**

Several features are used to predict the selling price, including:

* province
* property type
* subtype
* livable surface
* latitude
* longitude
* number of bedrooms
* number of bathrooms
* number of facades
* terrace
* garden
* garage
* swimming pool
* EPC score
* heating type
* state of the property
* distances to schools and train stations

The following columns were removed before training because they were considered unnecessary for prediction:

* property_id
* postal_code
* city

---

# Preprocessing

The preprocessing pipeline was implemented using Scikit-Learn's `Pipeline` and `ColumnTransformer`.

The preprocessing steps include:

* Standardization of numerical features using `StandardScaler`
* One-Hot Encoding of categorical variables using `OneHotEncoder`
* Automatic handling of unseen categories during prediction with `handle_unknown="ignore"`

Using a pipeline ensures that exactly the same preprocessing steps are applied during both training and prediction.

---

# Models

Three regression models were trained and evaluated:

* Linear Regression
* Random Forest Regressor
* XGBoost Regressor

Each model was trained on 80% of the dataset and evaluated on the remaining 20%.

---

# Evaluation Metrics

The models were evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* R² Score

These metrics were used to compare the different algorithms and identify the best-performing model.

---

# Saving the Models

After training, each model is saved using Joblib.

Saved models:

* `linear_regression_model.pkl`
* `random_forest_model.pkl`
* `xgboost_model.pkl`

These files are stored inside the `models/` folder and can later be loaded without retraining.

---

# Prediction

The `predict.py` script demonstrates how to:

* load a previously trained model
* create a new property
* predict its selling price

Example:

```bash
python src/predict.py
```

Example output:

```
Predicted price (Random Forest): €534,000.00
Predicted price (XGBoost): €548,000.00
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/immo-eliza-ml.git
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate it.

Windows:

```bash
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

---

# Running the Project

Train the models:

```bash
python src/train.py
```

Predict the price of a new property:

```bash
python src/predict.py
```

---

# Tools and libraries Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* XGBoost
* Joblib

---

# Possible Improvements

Several improvements could be made to increase the model's performance:

* Hyperparameter tuning using GridSearchCV or RandomizedSearchCV
* Cross-validation
* Feature selection
* Feature engineering
* Testing additional regression models

---

# Author

Project completed as part of the BeCode Artificial Intelligence & Data Science Bootcamp by Amellal Hiba.

"https://www.linkedin.com/in/amellal-hiba-7a636940a/"

