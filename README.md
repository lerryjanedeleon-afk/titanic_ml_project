#Asciinema link
https://asciinema.org/a/KiZH7SMpHP8ytoBbAjzIYF8gJ

# Titanic Survival Prediction (Machine Learning)

This project uses **supervised learning** to predict Titanic passenger survival using the famous Kaggle Titanic Dataset.

## Algorithm
- Logistic Regression (Scikit-learn)

## Dataset
Download from: https://www.kaggle.com/c/titanic/data  
Place `train.csv` in the project folder.

## Files
- train.py → trains the ML model
- predict.py → tests the model using manual inputs
- app.py → Flask API for predictions
- model.pkl → saved trained model
- titanic_ml.py → quick dataset preview

## How to Run

### 1. Install dependencies
pip install -r requirements.txt

### 2. Train the model
python train.py

### 3. Make a prediction
python predict.py

### 4. Run API server
python app.py

