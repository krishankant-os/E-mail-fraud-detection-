import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, classification_report


# Load dataset
dataset = pd.read_csv(
    r"C:\Users\krishankant\Downloads\archive (1)\emails.csv"
)

print(dataset.head())

# Input and output
X = dataset["text"]
Y = dataset["spam"]

# Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.20,
    random_state=42
)

# Create model
model = make_pipeline(
    TfidfVectorizer(),
    MultinomialNB()
)

# Train model
model.fit(X_train, Y_train)

# Test model
y_pred = model.predict(X_test)

# Evaluate
accuracy = accuracy_score(Y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:")
print(classification_report(Y_test, y_pred))


# Predict custom email
def custom_text(text):
    prediction = model.predict([text])[0]

    if prediction == 1:
        print(" spam")
    else:
        print(" not Spam")


custom_text(
 " unbelievable new homes made easy  im wanting to show you this  homeowner  you have been pre - approved for a $ 454 , 169 home loan at a 3 . 72 fixed rate .  this offer is being extended to you unconditionally and your credit is in no way a factor .  to take advantage of this limited time opportunity  all we ask is that you visit our website and complete  the 1 minute post approval form  look foward to hearing from you ,  dorcas pittman"
)
joblib.dump(
    model,
    r"C:\Users\krishankant\Downloads\my_model.pkl",
    compress=3
)

print("\nModel saved successfully as my_model.pkl")
