import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score
from v_preprocess import preprocess_test_file, preprocess_train, preprocess_test_list
import pickle
from sklearn.model_selection import GridSearchCV

def save_model(vectorizer_file, model_file):
    # Preprocessing training data from file
    X_train_v, y_train = preprocess_train(vectorizer_file=vectorizer_file)

    # Tuning hyperparameters by using cross-validation method
    model = LogisticRegression()
    parameters = {'penalty':['l1', 'l2'],'solver':['liblinear'],'C':[1e2, 1e4, 1e6],'max_iter': [100, 200, 500]}
    gridSearch = GridSearchCV(model, param_grid=parameters, verbose=1, cv=10, n_jobs=-1)
    gridSearch.fit(X_train_v, y_train)
    print("Best parameters:", gridSearch.best_params_)
    print("Best training Score:", gridSearch.best_score_)
    pickle.dump(gridSearch, open(model_file, 'wb'))
    print('Saved model.')

def print_report(vectorizer_file, model_file):
    X_test_v, y_test = preprocess_test_file(vectorizer_file=vectorizer_file)
    model = pickle.load(open(model_file, 'rb'))
    y_predict = model.predict(X_test_v)

    # Print report
    ''' Uncomment the 2 lines below to see confusion matrix and classification report '''
    # print('Confusion matrix:\n', confusion_matrix(y_test, y_predict))
    # print('Classification report:\n', classification_report(y_test, y_predict))
    print('Testing Set Accuracy:', accuracy_score(y_test, y_predict))
    print('Precision:', precision_score(y_test, y_predict))
    print('Recall:', recall_score(y_test, y_predict))
    print('F1:', f1_score(y_test, y_predict))

def make_predictions(list_inputs: list, vectorizer_file, model_file):
    # Preprocessing testing list
    X_test_v = preprocess_test_list(test_list=list_inputs, vectorizer_file=vectorizer_file)

    # Loading model from file and make predictions
    model = pickle.load(open(model_file, 'rb'))
    y_predict = model.predict(X_test_v)
    return y_predict

if __name__ == '__main__':
    print('Evaluation of the algorithm using Count Vectorizer')
    vectorizer_file = 'loaded_models\\count_vectorizer.pkl'
    model_file = 'loaded_models\\count_logistic_regression.pkl'
    print_report(vectorizer_file=vectorizer_file, model_file= model_file)
    print('Evaluation of the algorithm using TF-IDF Vectorizer')
    vectorizer_file = 'loaded_models\\tfidf_vectorizer.pkl'
    model_file = 'loaded_models\\tfidf_logistic_regression.pkl'
    print_report(vectorizer_file=vectorizer_file, model_file= model_file)
