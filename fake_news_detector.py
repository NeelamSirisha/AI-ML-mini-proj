# ------------------------------------------
# FAKE NEWS DETECTION USING MACHINE LEARNING
# ------------------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score

# Load a small sample dataset
data = {
    "text": [
        "NASA confirms water on Mars surface",
        "Government announces lockdown for 6 months",
        "Scientists discover cure for cancer",
        "Man claims he saw aliens in his backyard",
        "COVID-19 vaccine effective in reducing death rate",
        "Actor says earth is flat in viral interview",
        "AI helps doctors detect diseases earlier",
        "Fake news spreads that chocolate cures diabetes"
    ],
    "label": [
        "REAL", "REAL", "REAL",
        "FAKE", "REAL", "FAKE",
        "REAL", "FAKE"
    ]
}

df = pd.DataFrame(data)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    df["text"], df["label"], test_size=0.25, random_state=42
)

# Convert text to vectors
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train model
model = PassiveAggressiveClassifier()
model.fit(X_train_vec, y_train)

# Test accuracy
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)

print("------------------------------------------------")
print("           FAKE NEWS DETECTION SYSTEM")
print("------------------------------------------------")
print("Model Accuracy:", accuracy)

# Predict user input
print("\nEnter a news headline to check:")
headline = input("News: ")

headline_vec = vectorizer.transform([headline])
prediction = model.predict(headline_vec)[0]

print("\nPrediction:", prediction)
print("------------------------------------------------")
