# Fake News Detection System
Machine learning web app that classifies news articles as Real or Fake, with Explainable AI highlights. MCA team project (3 members).

## Features
- Manual text input or live URL scraping (BeautifulSoup4)
- NLP preprocessing: lowercase, tokenization, stop-word removal (NLTK)
- TF-IDF vectorization (5000 features)
- 6 models compared: Naive Bayes, Logistic Regression, SVM, Random Forest, Decision Tree, KNN
- "Inconclusive" output when confidence is below 65%
- Explainability: important words highlighted + Word Cloud
- Plotly confidence gauge and recent analyses history

## Results (Random Forest, 80/20 split)
- Accuracy: 99.9%
- F1 Score: 0.999
- Confusion matrix: 2 errors out of 2000 test articles

## Tech Stack
Python, Scikit-learn, NLTK, Pandas, NumPy, Streamlit, Plotly, Matplotlib, WordCloud, BeautifulSoup4

## Dataset
Fake.csv and True.csv (first 5000 rows of each used).
Not included due to size. Download:  https://drive.google.com/drive/folders/1n4x1kprslNq-LrN3gtcU0ALRPTbRBgtz?usp=drive_link

## How to Run
pip install -r requirements.txt
python train.py
streamlit run app.py

## Project Structure
- train.py: training and model saving
- preprocessing.py: text cleaning and URL scraping
- app.py: Streamlit dashboard
- api.py / test_api.py: API and testing
