# ------------------------------------------
# EMOTION DETECTION FROM TEXT
# ------------------------------------------

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Training data (small but effective)
texts = [
    "I am very happy today",
    "This is the best day of my life",
    "I feel sad and lonely",
    "I am so depressed",
    "I am very angry right now",
    "He makes me furious",
    "I am scared of the dark",
    "I feel fear inside me",
    "Wow this is amazing",
    "That's surprising and unexpected"
]

labels = [
    "HAPPY",
    "HAPPY",
    "SAD",
    "SAD",
    "ANGRY",
    "ANGRY",
    "FEAR",
    "FEAR",
    "SURPRISE",
    "SURPRISE"
]

# Vectorizing the text
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)

# Training the model
model = LogisticRegression()
model.fit(X, labels)

print("------------------------------------------------")
print("            EMOTION DETECTION SYSTEM")
print("------------------------------------------------")

while True:
    user_text = input("\nEnter a sentence (or type exit): ")

    if user_text.lower() == "exit":
        print("Thank you! Goodbye!")
        break

    # Transform user text
    X_user = vectorizer.transform([user_text])

    # Predict emotion
    prediction = model.predict(X_user)[0]

    print("\nPredicted Emotion:", prediction)
    print("------------------------------------------------")
