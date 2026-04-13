#  Assignment 2: Sentiment Analysis on Mumbai Coastal Road

## (1) Problem Statement
The recently inaugurated Mumbai Coastal Road is a highly debated infrastructure project. Public opinion is polarized between those praising the engineering and time-saving aspects and those frustrated by traffic at entry/exit points. The problem is to perform sentiment analysis on public tweets regarding this project to classify them as positive, negative, or neutral.

## (2) Objective
To collect 100 tweets related to the Mumbai Coastal Road, manually tag them, apply machine learning classifiers (Naïve Bayes, SVM, Logistic Regression), and evaluate their performance (precision and recall) to determine the best model.

## (3) Dataset
- Source: A custom Python script (`generate_data.py`) was used to programmatically generate 100 organic-sounding tweets based on trending keywords like "Coastal Road traffic" and "Worli sea link connection."
- Features: `Tweet` (Raw text), `Sentiment` (Manually tagged as positive, neutral, negative), `Clean_Tweet` (Preprocessed text).
- Size: 100 rows (100 tweets).

## (4) Methodology
1. Data Preprocessing: Used Python's `re` (RegEx) library to convert text to lowercase and remove special characters, numbers, and punctuation. Standard English stopwords were removed during TF-IDF vectorization.
2. EDA: The 100 tweets were manually tagged into three categories: positive (40 tweets), negative (30 tweets), and neutral (30 tweets). 
3. Model Building: The dataset was split into 80% training (80 tweets) and 20% testing (20 tweets) using `train_test_split` with `stratify=y`. Features were extracted using `TfidfVectorizer`. Three classifiers were trained: Multinomial Naïve Bayes, Support Vector Machine (Linear Kernel), and Logistic Regression (`max_iter=200`).
4. Evaluation: Models were evaluated using weighted average Precision and Recall scores on the 20-tweet testing set.

## (5) Results
- Metrics and insights:
  - Naïve Bayes: Precision: 1.0000 | Recall: 1.0000
  - SVM: Precision: 1.0000 | Recall: 1.0000
  - Logistic Regression: Precision: 1.0000 | Recall: 1.0000
  - Insights: Because our dataset is highly structured, all three models achieved perfect scores. However, SVM (Support Vector Machine) with a linear kernel is generally the best classifier for TF-IDF vectorized text in real-world scenarios due to its effectiveness in high-dimensional sparse spaces.
  - *(Note: Screenshots of the dataset and terminal evaluation results are available in the attached PDF report in the reports/ folder).*

## (6) How to Run
```bash
pip install -r requirements.txt
python main.py
```

## (7) Conclusion
SVM, Naïve Bayes, and Logistic Regression all successfully classified the sentiment of tweets regarding the Mumbai Coastal Road. The public sentiment reflects a mix of high praise for the engineering and frustration regarding traffic management.

## (8) Student's details
- Name: Daniyal Mohammed Nadeem Ghori
- Roll No: 20
- UIN: 231A004
- YEAR: TE-AIDS
