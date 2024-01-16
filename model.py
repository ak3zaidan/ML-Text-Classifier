from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
from sklearn.externals import joblib

# Training data
with open('data.txt', 'r') as file:
    lines = file.readlines()

# Parse data into the desired format
data = [tuple(line.strip().split(' ', 1)) for line in lines]

texts = [text for _, text in data]
categories = [category for category, _ in data]

# Shuffle the data
texts, categories = shuffle(texts, categories, random_state=42)

# Split the shuffled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(texts, categories, test_size=0.2, random_state=42)

# Convert text data to TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train the SVM model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = svm_model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Example: Classify a new string
new_string = "Innovations in blockchain technology are reshaping the finance sector."
new_string_tfidf = vectorizer.transform([new_string])
predicted_category = svm_model.predict(new_string_tfidf)[0]
print(f"Predicted Category: {predicted_category}")

# We dont want to read the data every time we want to access the model or even include the data into our app
# So we save the model along with the vectorizer

# To Save
# joblib.dump(svm_model, 'svm_model.joblib')
# joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')

# To load model when we need to use:
# svm_model = joblib.load('svm_model.joblib')
# vectorizer = joblib.load('tfidf_vectorizer.joblib')