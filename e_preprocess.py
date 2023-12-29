import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Preprocessing data from file
vect = TfidfVectorizer(stop_words='english')
def preprocess(vectorizer=vect, filename='emails.csv'):
    '''This function split a dataset into a training set and a testing set with testing set size equal to a quarter of the dataset size.
    Then it vectorizes the training and testing text.'''
    dataset1_df = pd.read_csv(filename)
    dataset1_df = dataset1_df[dataset1_df.columns[:2]]
    # print(dataset1_df.info())
    X = dataset1_df['text'].astype(str)
    y = dataset1_df['spam']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
    X_train_v = vectorizer.fit_transform(X_train)
    X_test_v = vectorizer.transform(X_test)
    X_array = X_train_v.toarray()
    # print(X_array)
    # arr_df = pd.DataFrame(X_array, columns=vect.get_feature_names_out())
    return X_train_v, X_test_v, y_train, y_test

if __name__ == '__main__':
    X_train_v, X_test_v, y_train, y_test = preprocess(vect, 'emails.csv')
    print(X_train_v, X_test_v, y_train, y_test)