from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

app = Flask(__name__)

# Get the folder where this script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model
model_path = os.path.join(BASE_DIR, "model.pkl")
model = pickle.load(open(model_path, "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Expected JSON: {"Pclass":3, "Sex":1, "Age":25, "Fare":7.25}
    values = np.array([[data['Pclass'], data['Sex'], data['Age'], data['Fare']]])
    prediction = model.predict(values)
    return jsonify({"survived": int(prediction[0])})

if __name__ == "__main__":
    try:
        app.run(debug=True, use_reloader=False)
    except KeyboardInterrupt:
        print("Server stopped by user")

