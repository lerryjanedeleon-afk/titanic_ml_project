import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import pickle

# Example: loading your CSV
df = pd.read_csv("titanic.csv")

# Selecting multiple columns correctly
columns = ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'Fare']
selected_data = df[columns]
print(selected_data.head())

# Preprocessing
df['Sex'] = df['Sex'].map({'male': 0, 'female': 1})
df['Age'] = df['Age'].fillna(df['Age'].mean())
df['Fare'] = df['Fare'].fillna(df['Fare'].mean())  # added for safety

# Features and target
X = df[['Pclass', 'Sex', 'Age', 'Fare']]
y = df['Survived']

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression(max_iter=300)
model.fit(x_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved as model.pkl")
