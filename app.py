from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    values = np.array([[data['Pclass'], data['Sex'], data['Age'], data['Fare']]])
    prediction = model.predict(values)
    return jsonify({"survived": int(prediction[0])})

if __name__ == "__main__":
    app.run(debug=True)
