# Spam Email Classifier

A machine learning project that classifies emails/messages as spam or ham (non-spam) using Natural Language Processing techniques. The model achieves **97.76% accuracy** with careful validation to ensure reliability.

## Project Overview

This project implements a binary classification model to detect spam messages using the SMS Spam Collection dataset. It employs preprocessing techniques including tokenization and vectorization, and achieves high performance using Naive Bayes algorithm.

## Key Features

- **Text Preprocessing Pipeline**: Tokenization, stopword removal, and stemming
- **Feature Engineering**: CountVectorizer with binary features
- **Model Training**: Multinomial Naive Bayes with optimized regularization
- **Validation Framework**: Train/validation/test split with cross-validation
- **Performance Optimization**: Reduced dimensionality and efficient processing
- **Overfitting Prevention**: Regularization parameter tuning and gap analysis

## Results

The model achieves excellent performance metrics:

| Metric | Value |
|--------|-------|
| Accuracy | 97.76% |
| Precision (Ham) | 98% |
| Precision (Spam) | 93% |
| Recall (Ham) | 99% |
| Recall (Spam) | 90% |
| F1 Score (Ham) | 99% |
| F1 Score (Spam) | 91% |
| ROC AUC | 0.9748 |

### Confusion Matrix

```
          Predicted
         Ham    Spam
Actual Ham     956     10
      Spam     15    134
```

## Dataset

The project uses the SMS Spam Collection dataset, which contains:
- 5,574 messages total
- 4,827 ham messages (86.60%)
- 747 spam messages (13.40%)

Key characteristics:
- Spam messages tend to be longer (avg 138 chars) than ham messages (avg 71 chars)
- Class imbalance is handled through stratified sampling and evaluation metrics

## Implementation Details

### Prerequisites

```
pip install pandas numpy matplotlib seaborn scikit-learn nltk
```

### Technical Architecture

The implementation follows these key steps:

1. **Data Preprocessing**
   - Text cleaning (lowercase conversion, special character removal)
   - Tokenization and stopword removal
   - Word stemming using Porter Stemmer

2. **Feature Engineering**
   - Bag-of-words representation using CountVectorizer
   - Binary feature encoding
   - Dimensionality reduction (3,000 max features)

3. **Model Training**
   - Multinomial Naive Bayes classifier
   - Optimized regularization (alpha=0.1)
   - Pipeline approach for streamlined processing

4. **Validation Framework**
   - 60/20/20 train/validation/test split
   - Stratified sampling to maintain class distribution
   - 5-fold cross-validation
   - Learning curve analysis

5. **Performance Optimization**
   - Efficient text processing
   - Execution time: 4.23 seconds total
   - Prediction time: ~0.001 seconds per message

## Evaluation

### Overfitting Analysis
- Training accuracy: 99.10%
- Validation accuracy: 97.76%
- Test accuracy: 97.76%
- Train-validation gap: 1.35%
- Validation-test gap: 0.00%

These metrics confirm the model generalizes well without overfitting.

### Cross-Validation Results
- Mean CV accuracy: 97.70% (±0.84%)
- Small standard deviation indicates stable performance

### Regularization Testing
Several alpha values were tested (0.01, 0.1, 0.5, 1.0, 2.0), with 0.1 providing optimal results.

## Usage

The model can classify new messages as spam or ham:

```python
# Example predictions
test_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
    "Hi Mom, what time will you be home for dinner?",
    "URGENT: Your bank account has been compromised. Call this number immediately.",
    "Meeting at 3pm tomorrow in the conference room."
]

for message in test_messages:
    label, prob = predict_spam(message)
    print(f"Message: {message}")
    print(f"Prediction: {label} (Spam probability: {prob:.4f})")
```

## Future Improvements

Potential enhancements to explore:
1. Implementing advanced NLP techniques like word embeddings
2. Testing ensemble methods for improved accuracy
3. Building a simple web interface for interactive spam detection
4. Adding model explainability to show influential words
5. Expanding to other languages or specialized domains

## Skills Demonstrated

This project showcases several important data science and machine learning skills:

- **Natural Language Processing**: Text preprocessing, tokenization, stemming
- **Feature Engineering**: Text vectorization, dimensionality reduction
- **Machine Learning**: Classification modeling, hyperparameter tuning
- **Model Validation**: Cross-validation, learning curves, overfitting detection
- **Performance Optimization**: Time and space complexity improvements
- **Python Programming**: Efficient use of pandas, scikit-learn, and NLTK

## Author

Ezekiel Folarin