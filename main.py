import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

def clean_text(text):
    text = text.lower()
    return re.sub(r'[^a-z\s]', '', text)

def main():
    print("--- Loading Dataset ---")
    df = pd.read_csv("dataset.csv")
    df['Clean_Tweet'] = df['Tweet'].apply(clean_text)
    
    # Vectorization
    vectorizer = TfidfVectorizer(stop_words='english')
    X = vectorizer.fit_transform(df['Clean_Tweet'])
    y = df['Sentiment']
    
    # 80/20 Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}\n")
    
    # Models
    models = {"Naïve Bayes": MultinomialNB(), "SVM": SVC(kernel='linear'), "Logistic Regression": LogisticRegression(max_iter=200)}
    
    print("--- Evaluation (Weighted Precision & Recall) ---")
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        p = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        r = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        print(f"{name} -> Precision: {p:.4f} | Recall: {r:.4f}")

if __name__ == "__main__":
    main()