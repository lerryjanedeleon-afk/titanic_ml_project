import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

# Example passenger
# Pclass, Sex (0=male,1=female), Age, Fare
sample = np.array([[3, 1, 25, 7.25]])

prediction = model.predict(sample)

print("Survived" if prediction[0] == 1 else "Did not survive")
