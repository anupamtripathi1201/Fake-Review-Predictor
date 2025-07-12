#  Fake Review Detector

This project is a basic machine learning model that detects fake product reviews using Natural Language Processing (NLP) techniques and Logistic Regression.

##  Overview

- Cleans and preprocesses review text (lowercasing, stopword removal, stemming)
- Converts text to numerical features using TF-IDF
- Trains a Logistic Regression model for binary classification (real or fake review)
- Achieves **86% accuracy** on test data

##  Dataset

The dataset used contains product reviews labeled as real (0) or fake (1).  
Ensure the dataset file is named correctly and placed in the correct path (as expected in the script).

##  Tech Stack

- Python 3
- Pandas
- Scikit-learn
- NLTK

##  Model Performance : 

             precision    recall  f1-score   support

           0       0.87      0.86      0.86      5113
           1       0.85      0.87      0.86      5013

    accuracy                           0.86     10126
   macro avg       0.86      0.86      0.86     10126
weighted avg       0.86      0.86      0.86     10126



##  How to Run

1. Install dependencies:

```bash
pip install pandas scikit-learn nltk


