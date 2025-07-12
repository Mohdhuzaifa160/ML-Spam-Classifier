import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Load CSV
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']  # Rename columns

# Step 2: Convert labels to 0 (ham) and 1 (spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 3: Split into train and test
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2)

# Step 4: Text to numbers
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Step 5: Train Naive Bayes Model
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# Step 6: Predict
y_pred = model.predict(X_test_vec)

# Step 7: Accuracy
print("Model Accuracy:", accuracy_score(y_test, y_pred))
