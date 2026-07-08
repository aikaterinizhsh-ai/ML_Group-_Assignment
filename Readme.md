# Sentiment Analysis & Topic Classification on Social Media Posts

## Machine Learning and Python — Group Assignment 2024/25

This project performs **Sentiment Analysis** and **Topic Classification** on social media posts across four topics: Politics, Elections, European Union, and Olympic Games.

---

## Dataset

- **1,280 real tweets** from three Kaggle sources
- **4 topics** (320 posts each): Politics, Elections, European Union, Olympic Games
- **3 sentiment categories**: Positive, Neutral, Negative
- **3 independent annotators** per post with majority voting

### Data Sources

| Topic | Source |
|-------|--------|
| Politics | Congress Tweets (Kaggle) |
| Elections | Congress Tweets (Kaggle) |
| European Union | Brexit Twitter Dataset (Kaggle) |
| Olympic Games | Tokyo 2020 Tweets (Kaggle) |

---

## Methods

- **Text Preprocessing**: lowercase, URL/mention/hashtag removal, stopwords, special characters
- **Feature Extraction**: TF-IDF (500 features) + word frequency scores (percentile-based)
- **Topic Classification**: Naive Bayes, KNN, Decision Tree, Random Forest, Logistic Regression, SVM
- **Sentiment Classification**: Binary (positive vs negative), neutral excluded
- **Evaluation**: Accuracy, Precision, Recall, F1-score, Confusion Matrix, Cohen's Kappa

---

## Results

### Topic Classification

| Model | F1 Score |
|-------|----------|
| Logistic Regression | **0.817** |
| SVM | 0.805 |
| Random Forest | 0.771 |
| Naive Bayes | 0.749 |
| Decision Tree | 0.677 |
| KNN | 0.481 |

### Sentiment Classification

| Model | F1 Score |
|-------|----------|
| SVM | **0.801** |
| Logistic Regression | 0.777 |
| Random Forest | 0.767 |
| Naive Bayes | 0.754 |
| Decision Tree | 0.678 |
| KNN | 0.543 |

---

## Files

- `SA_Complete_Fixed.ipynb` — Main notebook with all analysis
- `tweets_dataset.csv` — Dataset (1,280 tweets)

---

## How to Run

```bash
pip install pandas numpy matplotlib seaborn scikit-learn textblob
```

1. Place `SA_Complete_Fixed.ipynb` and `tweets_dataset.csv` in the same folder
2. Open the notebook in Jupyter or VS Code
3. Run All

---

## Technologies

- Python 3.x
- scikit-learn
- pandas, numpy
- matplotlib, seaborn
- TextBlob (pre-annotation)
- TF-IDF Vectorization

---

## Further Analysis

- Sentiment time series per topic (monthly)
- Topic-sentiment cross-tabulation
- Bi-gram extraction and classification
- Keywords and bi-grams per sentiment category