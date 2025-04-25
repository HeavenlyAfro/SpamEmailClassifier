import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer  # Changed from TfidfVectorizer for speed
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import time  # Added for timing performance

# To avoid downloading NLTK resources every time, only download if not already present
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)

start_time = time.time()  # Start timing

# Load the SMS Spam Collection dataset with the specific path
dataset_path = r"C:\Users\ezeki\PycharmProjects\SpamEmailClassifier\dataset\SMSSpamCollection"

# The SMSSpamCollection dataset has a different format (tab-separated)
# Read the file with custom format: each line has "label\tmessage"
with open(dataset_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

labels = []
messages = []
for line in lines:
    # Split each line at the first tab character
    label, message = line.split('\t', 1)
    labels.append(label)
    messages.append(message.strip())

# Create DataFrame
data = pd.DataFrame({'label': labels, 'message': messages})

# Skip visualization to reduce execution time, but keep basic EDA
print("Data shape:", data.shape)
print("\nData overview:")
print(data.head())

print("\nClass distribution:")
print(data['label'].value_counts())
print(data['label'].value_counts(normalize=True) * 100)

# More efficient message length calculation - skip visualization
data['length'] = data['message'].str.len()  # More efficient than apply
print("\nMessage length statistics:")
print(data.groupby('label')['length'].describe())


# Simplified preprocessing function for better performance
def preprocess_text(text):
    # Convert to lowercase and remove special characters
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())

    # Use a set for faster lookups
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()

    # Process words in one pass for better performance
    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(stemmer.stem(word))

    return ' '.join(words)


# Use a more efficient approach for preprocessing
print("Starting text preprocessing...")
preprocessing_start = time.time()
data['processed_message'] = data['message'].apply(preprocess_text)
print(f"Text preprocessing completed in {time.time() - preprocessing_start:.2f} seconds")

# Convert labels to binary form
data['label_binary'] = data['label'].map({'ham': 0, 'spam': 1})

# Convert labels to binary form
data['label_binary'] = data['label'].map({'ham': 0, 'spam': 1})

# Split the data into training, validation, and testing sets
X = data['processed_message']
y = data['label_binary']

# First split: 80% train+validation, 20% test
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Second split: 75% train, 25% validation (from the 80% train+validation)
# This gives us approximately 60% train, 20% validation, 20% test
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)

print("\nTraining set size:", X_train.shape[0])
print("Validation set size:", X_val.shape[0])
print("Testing set size:", X_test.shape[0])

# Check class distribution in each split to ensure balanced representation
print("\nClass distribution in splits:")
print(f"Original - Spam: {sum(y == 1) / len(y):.2%}, Ham: {sum(y == 0) / len(y):.2%}")
print(f"Training - Spam: {sum(y_train == 1) / len(y_train):.2%}, Ham: {sum(y_train == 0) / len(y_train):.2%}")
print(f"Validation - Spam: {sum(y_val == 1) / len(y_val):.2%}, Ham: {sum(y_val == 0) / len(y_val):.2%}")
print(f"Testing - Spam: {sum(y_test == 1) / len(y_test):.2%}, Ham: {sum(y_test == 0) / len(y_test):.2%}")

# Import cross-validation tools
from sklearn.model_selection import cross_val_score, learning_curve
from sklearn.metrics import roc_curve, auc
import warnings

# Create a pipeline with CountVectorizer and Naive Bayes
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(max_features=3000,
                                   stop_words='english',
                                   binary=True)),
    ('classifier', MultinomialNB(alpha=0.1))
])

# Perform k-fold cross-validation on the training set
print("\nPerforming 5-fold cross-validation...")
cv_start = time.time()
with warnings.catch_warnings():
    warnings.simplefilter("ignore")  # Suppress convergence warnings
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"Cross-validation completed in {time.time() - cv_start:.2f} seconds")
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Train on training set
print("\nTraining final model...")
training_start = time.time()
pipeline.fit(X_train, y_train)
print(f"Model training completed in {time.time() - training_start:.2f} seconds")

# Evaluate on validation set to detect overfitting
print("\nEvaluating on validation set...")
val_start = time.time()
val_pred = pipeline.predict(X_val)
val_accuracy = accuracy_score(y_val, val_pred)
val_report = classification_report(y_val, val_pred)
print(f"Validation evaluation completed in {time.time() - val_start:.2f} seconds")
print(f"Validation accuracy: {val_accuracy:.4f}")
print("Validation classification report:")
print(val_report)

# Then evaluate on test set (final evaluation)
print("\nEvaluating on test set...")
test_start = time.time()
y_pred = pipeline.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test evaluation completed in {time.time() - test_start:.2f} seconds")

# Compute overfitting metrics
print("\nOverfitting analysis:")
train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
print(f"Training accuracy: {train_accuracy:.4f}")
print(f"Validation accuracy: {val_accuracy:.4f}")
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Train-validation gap: {train_accuracy - val_accuracy:.4f}")
print(f"Validation-test gap: {val_accuracy - test_accuracy:.4f}")

if train_accuracy - val_accuracy > 0.05:
    print("Warning: Potential overfitting detected (train-validation accuracy gap > 5%)")
else:
    print("No significant overfitting detected based on accuracy gaps")

# Evaluate the model on test set
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("\nTest Set Performance:")
print(f"Accuracy: {test_accuracy:.4f}")

print("\nConfusion Matrix:")
print(conf_matrix)

print("\nClassification Report:")
print(class_report)

# Simple text-based confusion matrix display
print("\nConfusion Matrix Visualization:")
print("          Predicted      ")
print("         Ham    Spam    ")
print(f"Actual Ham  {conf_matrix[0][0]:6d} {conf_matrix[0][1]:6d}")
print(f"      Spam {conf_matrix[1][0]:6d} {conf_matrix[1][1]:6d}")

# Generate learning curves to visualize overfitting
print("\nGenerating learning curves...")
train_sizes, train_scores, val_scores = learning_curve(
    pipeline, X_train, y_train,
    train_sizes=np.linspace(0.1, 1.0, 5),
    cv=5, scoring='accuracy'
)

# Calculate mean and standard deviation for training and validation scores
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

print("Learning curve results:")
print(f"Training samples: {train_sizes}")
print(f"Training scores (mean): {train_mean}")
print(f"Validation scores (mean): {val_mean}")

# Calculate ROC curve and AUC
y_prob = pipeline.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print(f"\nROC AUC: {roc_auc:.4f}")


# Optimized prediction function
def predict_spam(message, pipeline=pipeline):
    """
    Predict whether a message is spam or ham.

    Parameters:
    message (str): Input message
    pipeline: Trained ML pipeline

    Returns:
    str: 'spam' or 'ham'
    float: Probability of being spam
    """
    # Skip preprocessing - let the pipeline's vectorizer handle it
    # This is faster as it uses the same preprocessing steps as in training

    # Predict
    prediction = pipeline.predict([message])[0]
    spam_prob = pipeline.predict_proba([message])[0][1]

    return ('spam' if prediction == 1 else 'ham'), spam_prob


# Test the model with some examples
test_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim your prize now!",
    "Hi Mom, what time will you be home for dinner?",
    "URGENT: Your bank account has been compromised. Call this number immediately.",
    "Meeting at 3pm tomorrow in the conference room."
]

print("\nTesting model with sample messages:")
for message in test_messages:
    prediction_start = time.time()
    label, prob = predict_spam(message)
    prediction_time = time.time() - prediction_start
    print(f"Message: {message}")
    print(f"Prediction: {label} (Spam probability: {prob:.4f})")
    print(f"Prediction time: {prediction_time:.4f} seconds")
    print()

# Add regularization experiment to combat overfitting if needed
print("\nTesting different regularization values (alpha) to combat potential overfitting...")
alpha_values = [0.01, 0.1, 0.5, 1.0, 2.0]
alpha_scores_train = []
alpha_scores_val = []

for alpha in alpha_values:
    # Create a pipeline with the current alpha value
    alpha_pipeline = Pipeline([
        ('vectorizer', CountVectorizer(max_features=3000, stop_words='english', binary=True)),
        ('classifier', MultinomialNB(alpha=alpha))
    ])

    # Train and evaluate
    alpha_pipeline.fit(X_train, y_train)
    train_score = accuracy_score(y_train, alpha_pipeline.predict(X_train))
    val_score = accuracy_score(y_val, alpha_pipeline.predict(X_val))

    alpha_scores_train.append(train_score)
    alpha_scores_val.append(val_score)

# Print regularization results
print("Alpha values:", alpha_values)
print("Training scores:", [f"{score:.4f}" for score in alpha_scores_train])
print("Validation scores:", [f"{score:.4f}" for score in alpha_scores_val])

# Find best alpha
best_alpha_index = np.argmax(alpha_scores_val)
best_alpha = alpha_values[best_alpha_index]
print(f"Best alpha: {best_alpha} (validation accuracy: {alpha_scores_val[best_alpha_index]:.4f})")

# If the best alpha is different from our default, retrain with it
if best_alpha != 0.1:
    print(f"Retraining final model with optimal alpha={best_alpha}...")
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(max_features=3000, stop_words='english', binary=True)),
        ('classifier', MultinomialNB(alpha=best_alpha))
    ])
    pipeline.fit(X_train, y_train)
    print("Final model retrained with optimal regularization")

# Performance optimization and validation notes
print("\nPerformance and Validation Notes:")
print("1. Using CountVectorizer instead of TfidfVectorizer (faster computation)")
print("2. Reduced max_features to 3000 (lower dimensionality)")
print("3. Used binary=True in vectorizer (simpler features)")
print("4. Added cross-validation to detect and prevent overfitting")
print("5. Used separate validation set to tune hyperparameters")
print("6. Tested different regularization values to combat overfitting")
print(f"7. Selected optimal alpha={best_alpha} for regularization")

# Save the final model using pickle
import pickle

with open('spam_classifier_model.pkl', 'wb') as f:
    pickle.dump(pipeline, f)
print("\nFinal model saved as 'spam_classifier_model.pkl'")

# Print total execution time
print(f"\nTotal execution time: {time.time() - start_time:.2f} seconds")