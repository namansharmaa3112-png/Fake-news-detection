import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from preprocessing import clean_text

class EnsembleModel:
    def __init__(self, models):
        self.models = models

    def predict_proba(self, X):
        probs = [model.predict_proba(X) for model in self.models]
        return np.mean(probs, axis=0)

    def predict(self, X):
        avg_probs = self.predict_proba(X)
        return np.argmax(avg_probs, axis=1)


def train_and_evaluate():
    print("Starting Fake News Detection Training Pipeline...")
    
    # 1. Load dataset
    fake_path = os.path.join('data_set', 'Fake.csv')
    true_path = os.path.join('data_set', 'True.csv')
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print(f"Error: Dataset files not found in 'data_set/'. Please provide them.")
        return
        
    print("Loading dataset...")
    # Load all available rows for better model accuracy and confidence
    df_fake = pd.read_csv(fake_path)
    df_true = pd.read_csv(true_path)
    
    # 2. Add labels and combine
    print("Cleaning data...")
    if 'text' not in df_fake.columns or 'text' not in df_true.columns:
        print("Error: Dataset must contain a 'text' column.")
        return
        
    df_fake['label'] = 0
    df_true['label'] = 1
    
    df = pd.concat([df_fake, df_true], ignore_index=True)
    
    # Shuffle the full dataset to preserve diversity
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    df.dropna(subset=['text', 'label'], inplace=True)
    df['label'] = df['label'].astype(int)

    # 3. NLP Preprocessing
    print("Preprocessing text... This might take a while.")
    df['clean_text'] = df['text'].apply(clean_text)
    
    # Drop rows that became empty after cleaning
    df = df[df['clean_text'].str.strip() != '']
    
    X = df['clean_text']
    y = df['label']
    
    # 4. Convert text to numerical features using TF-IDF with n-grams and sublinear term frequency
    print("Applying TF-IDF with n-grams...")
    tfidf = TfidfVectorizer(max_features=12000, ngram_range=(1, 2), min_df=2, max_df=0.9, sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(X)
    
    # 5. Split dataset into training and testing sets (80/20)
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, stratify=y, random_state=42)
    
    # 6. Train multiple supervised models with probability calibration for higher confidence
    print("Training models...")
    base_models = {
        "Logistic Regression": LogisticRegression(max_iter=5000, solver='saga', C=1.0, penalty='l2', class_weight='balanced', random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=20, min_samples_split=4, min_samples_leaf=2, class_weight='balanced', random_state=42),
    }
    
    # Create calibrated versions for better probability estimates
    models = {}
    for name, model in base_models.items():
        models[name] = CalibratedClassifierCV(model, method='sigmoid', cv=3)
    
    results = {}
    best_f1 = -1
    best_model_name = ""
    best_model = None
    best_cm = None
    
    # 7. Evaluate models
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        results[name] = {
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1
        }
        
        # 8. Compare all models and select the best one based on F1 Score
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_model = model
            best_cm = cm
            
    ensemble_model = EnsembleModel(list(models.values()))
    
    print(f"\nTraining complete. Best Base Model: {best_model_name} with F1 Score: {best_f1:.4f}")
    print("Using ensemble of all calibrated models for production predictions.")
    
    # 9. Save ensemble model using pickle
    print("Saving best model and artifacts...")
    artifacts_dir = "artifacts"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    with open(os.path.join(artifacts_dir, 'best_model.pkl'), 'wb') as f:
        pickle.dump(ensemble_model, f)
        
    with open(os.path.join(artifacts_dir, 'tfidf_vectorizer.pkl'), 'wb') as f:
        pickle.dump(tfidf, f)
        
    metrics = {
        "best_model_name": "Ensemble of Logistic Regression + Random Forest",
        "results": results,
        "best_confusion_matrix": best_cm
    }
    
    with open(os.path.join(artifacts_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)
        
    print("All artifacts saved successfully in the 'artifacts/' directory.")
    print("You can now run 'streamlit run app.py' to launch the app.")

if __name__ == "__main__":
    train_and_evaluate()
