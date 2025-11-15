import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pickle

# Load dataset
df = pd.read_csv("train.csv")

# Select useful columns
df = df[['Survived', 'Pclass', 'Sex', 'Age', 'Fare']]

# Preprocessing
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].mean())

# Train-test split
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Model
model = LogisticRegression(max_iter=300)
model.fit(x_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved as model.pkl")
