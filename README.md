# Sentiment Analysis of Tweets using Logistic Regression and Word Embeddings

## Overview
The goal of this project was to classify tweets into two categories: **Negative (`0`)** and **Positive (`1`)**.  
We used **Logistic Regression** as the classification model and **GloVe word embeddings** (`glove-twitter-25`) to represent text data numerically.

## Steps

### 1. Data Preprocessing
- **Tokenization**: Each tweet was tokenized into words.
- **Embedding**: Tokens were converted into 25-dimensional vectors using the `glove-twitter-25` model.
- **Aggregation**: For each tweet, we averaged the vectors of its words to create a single 25-dimensional vector.

### 2. Data Splitting
- The dataset was split into **Training (80%)** and **Testing (20%)** sets using `train_test_split`.
- We used `stratify=y` to ensure class distribution was preserved in both sets.

### 3. Model Training
- A **Logistic Regression** model was trained on the training set (`X_train`, `y_train`).
- The model was evaluated on the test set (`X_test`, `y_test`).

### 4. Evaluation Metrics
#### Test Set Results:
- **Accuracy**: 78.56%
- **Precision/Recall/F1-Score**:
  - Negative Class (`0`): Precision=0.80, Recall=0.97, F1-Score=0.87
  - Positive Class (`1`): Precision=0.65, Recall=0.20, F1-Score=0.31

#### Confusion Matrix:


#### Cross Validation:
- **5-Fold Cross Validation** was performed on the training set.
- Average accuracy: **78.56% ± 0.0007**

### 5. Observations
1. **Strengths**:
   - The model performed well on the **negative class** (Recall=0.97).
   - Overall accuracy was decent (**78.56%**).

2. **Weaknesses**:
   - The model struggled with the **positive class** (Recall=0.20).
   - This is likely due to the **class imbalance** in the dataset (only 20% positive samples).

### 6. Future Improvements
1. **Class Balancing**:
   - Use `class_weight='balanced'` to improve performance on the minority class.
   - Apply resampling techniques like **SMOTE** or **RandomOverSampler**.

2. **Advanced Models**:
   - Experiment with more complex models like **Neural Networks**, **LSTM**, or **Transformers** for better text representation.

3. **Better Embeddings**:
   - Use higher-dimensional embeddings like `word2vec-google-news-300` for richer semantic representations.

---

## Conclusion
This project demonstrated how to build a sentiment analysis model using Logistic Regression and GloVe embeddings.  
The model achieved **78.56% accuracy** but showed weaknesses in detecting the positive class due to class imbalance.  
Future work includes balancing the dataset and experimenting with advanced models for improved performance.