
# Customer Churn Prediction (Machine Learning)

## About the Project

This project is a simple end-to-end machine learning solution to predict whether a customer is likely to **churn (leave a service)** or not.

The idea behind this project was to understand how a real ML workflow looks — from data cleaning and preprocessing to model training, evaluation, and finally using the trained model to make predictions on new customer data.


## What Problem Does It Solve?

Customer churn is a big problem for subscription-based businesses.
If a company can identify customers who are likely to leave, it can take action early (offers, support, discounts, etc.).

This project predicts churn based on:

* Customer demographics
* Services used
* Contract and billing information
* Tenure and charges

## Dataset

I used the **Telco Customer Churn dataset**.

* Target column: `Churn`

  * `1` → Customer left
  * `0` → Customer stayed
* The dataset contains both **categorical and numerical features**
* The data is imbalanced (more non-churn customers than churn customers)

Exploratory Data Analysis was performed in a separate notebook to understand feature distributions, correlations, and class imbalance before model training.

## How the Project Works

### 1. Data Cleaning

* Removed the `customerID` column because it is just an identifier
* Converted the `Churn` column from Yes/No to 1/0
* Checked for missing values

### 2. Feature Encoding

* Categorical columns were converted to numeric form using **Label Encoding**
* All encoders were saved so the same encoding can be used during prediction


### 3. Handling Class Imbalance

* The dataset is imbalanced, so I used **SMOTE** to oversample the minority class (churn customers)
* This helped the model learn churn patterns better

### 4. Model Training

I trained and compared multiple models:

* Decision Tree
* Random Forest
* XGBoost

Because the data is imbalanced, I used **F1-score** instead of accuracy to compare models.


### 5. Model Evaluation

* Random Forest performed the best overall
* Final model achieved around **77% accuracy**
* F1-score for churn class showed balanced precision and recall


### 6. Saving the Model

* The trained model was saved using `pickle`
* Encoders were also saved to ensure consistent predictions
* This allows reuse of the model without retraining


### 7. Prediction on New Data

A separate script (`predict.py`) was created to:

* Load the saved model and encoders
* Accept new customer input
* Encode features correctly
* Predict churn and churn probability

Example output:
Prediction: No Churn
Probability: 62% stay, 38% churn


## Project Structure

Customer-churn-ml/
├── data/raw/telco_churn.csv
├── models/
│   ├── customer_churn_model.pkl
│   └── encoders.pkl
├── notebooks/
│   └── churn_eda.ipynb
├── src/
│   ├── train_model.py
│   └── predict.py
└── README.md


## How to Run the Project

1. Activate virtual environment
venv\Scripts\activate


2. Train the model
python src/train_model.py


3. Run prediction
python src/predict.py



## What I Learned

* How to handle imbalanced datasets using SMOTE
* Why F1-score is better than accuracy for churn problems
* Importance of consistent feature encoding
* Saving and loading ML models properly
* Building a real-world ML workflow, not just training a model

