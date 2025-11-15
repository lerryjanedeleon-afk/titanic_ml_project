import pickle
import numpy as np
import os

# Get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the model from the same folder
model_path = os.path.join(BASE_DIR, "model.pkl")
model = pickle.load(open(model_path, "rb"))

# Example passenger input
# Pclass, Sex (0=male,1=female), Age, Fare
sample = np.array([[3, 1, 25, 7.25]])

# Make prediction
prediction = model.predict(sample)
print("Survived" if prediction[0] == 1 else "Did not survive")
