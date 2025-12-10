# -----------------------------
# IRIS FLOWER CLASSIFICATION
# -----------------------------

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load Dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Scale the Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Train the Model
model = SVC()
model.fit(X_train, y_train)

# 5. Test the Model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("------------------------------------------------")
print("      IRIS FLOWER CLASSIFICATION MODEL")
print("------------------------------------------------")
print("Accuracy of the model:", accuracy)

# 6. Predict on New Data
print("\nEnter flower measurements to predict (cm):")

sl = float(input("Sepal Length: "))
sw = float(input("Sepal Width: "))
pl = float(input("Petal Length: "))
pw = float(input("Petal Width: "))

sample = [[sl, sw, pl, pw]]
sample_scaled = scaler.transform(sample)
prediction = model.predict(sample_scaled)

print("\nPredicted Flower Type:", iris.target_names[prediction][0])
print("------------------------------------------------")
