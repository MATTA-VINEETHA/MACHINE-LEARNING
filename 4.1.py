# Step 1: Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# Step 2: Load Dataset (Replace with your dataset)
df = pd.read_csv("spam.csv", encoding="latin-1")  # Example: Kaggle spam dataset
df = df[["v1", "v2"]]  # Select relevant columns (label, text)
df.columns = ["label", "text"]

# Step 3: Preprocess Data
X = df["text"]  # Features (text messages)
y = df["label"].map({"ham": 0, "spam": 1})  # Labels (0=ham, 1=spam)

# Step 4: Vectorize Text (Convert text to numbers)
vectorizer = TfidfVectorizer(stop_words="english")
X_vectorized = vectorizer.fit_transform(X)

# Step 5: Split Data into Train/Test Sets
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Step 6: Train Model (Naive Bayes for text classification)
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluate Model
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()