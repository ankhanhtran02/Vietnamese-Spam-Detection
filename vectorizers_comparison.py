import KNN, naive_bayes, logistic_regression, SVM, ANN
from v_preprocess import preprocess_test_output
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd

def TF_IDF_experiment():
    print('-------------------------------')
    print('Performance of TF-IDF vectorizer with majority vote from 5 classifiers:')
    df = pd.read_csv('datasets\\test_set.csv')
    X_test = df['text'].astype(str)
    y_test = preprocess_test_output(df['spam'])
    vectorizer_file = 'loaded_models\\tfidf_vectorizer.pkl'
    KNN_pred = KNN.make_predictions(X_test, vectorizer_file, 'loaded_models\\tfidf_KNN.pkl')
    naive_bayes_pred = naive_bayes.make_predictions(X_test, vectorizer_file, 'loaded_models\\tfidf_naive_bayes.pkl')
    logistic_regression_pred = logistic_regression.make_predictions(X_test, vectorizer_file, 'loaded_models\\tfidf_logistic_regression.pkl')
    SVM_pred = SVM.make_predictions(X_test, vectorizer_file, 'loaded_models\\tfidf_SVM.pkl')
    ANN_pred = ANN.make_predictions(X_test, vectorizer_file, 'loaded_models\\tfidf_ANN.h5')
    y_predict = []
    for i in range(len(y_test)):
        if (KNN_pred[i] + naive_bayes_pred[i] + logistic_regression_pred[i] + SVM_pred[i] + ANN_pred[i]) >= 3:
            y_predict.append(1)
        elif (KNN_pred[i] + naive_bayes_pred[i] + logistic_regression_pred[i] + SVM_pred[i] + ANN_pred[i]) < 3:
            y_predict.append(0)
    assert len(y_predict) ==  len(y_test)
    ''' Uncomment the 2 lines below to see confusion matrix and classification report '''
    # print('Confusion matrix:', confusion_matrix(y_test, y_predict))
    # print('Classification report:', classification_report(y_test, y_predict))
    print('Testing Set Accuracy:', accuracy_score(y_test, y_predict))
    print('Precision:', precision_score(y_test, y_predict))
    print('Recall:', recall_score(y_test, y_predict))
    print('F1:', f1_score(y_test, y_predict))

def BoW_experiment():
    print('-------------------------------')
    print('Performance of count vectorizer with majority vote from 5 classifiers:')
    df = pd.read_csv('datasets\\test_set.csv')
    X_test = df['text'].astype(str)
    y_test = preprocess_test_output(df['spam'])
    vectorizer_file = 'loaded_models\\count_vectorizer.pkl'
    KNN_pred = KNN.make_predictions(X_test, vectorizer_file, 'loaded_models\\count_KNN.pkl')
    naive_bayes_pred = naive_bayes.make_predictions(X_test, vectorizer_file, 'loaded_models\\count_naive_bayes.pkl')
    logistic_regression_pred = logistic_regression.make_predictions(X_test, vectorizer_file, 'loaded_models\\count_logistic_regression.pkl')
    SVM_pred = SVM.make_predictions(X_test, vectorizer_file, 'loaded_models\\count_SVM.pkl')
    ANN_pred = ANN.make_predictions(X_test, vectorizer_file, 'loaded_models\\count_ANN.h5')
    y_predict = []
    for i in range(len(y_test)):
        if (KNN_pred[i] + naive_bayes_pred[i] + logistic_regression_pred[i] + SVM_pred[i] + ANN_pred[i]) >= 3:
            y_predict.append(1)
        elif (KNN_pred[i] + naive_bayes_pred[i] + logistic_regression_pred[i] + SVM_pred[i] + ANN_pred[i]) < 3:
            y_predict.append(0)
    assert len(y_predict) ==  len(y_test)
    ''' Uncomment the 2 lines below to see confusion matrix and classification report '''
    # print('Confusion matrix:', confusion_matrix(y_test, y_predict))
    # print('Classification report:', classification_report(y_test, y_predict))
    print('Testing Set Accuracy:', accuracy_score(y_test, y_predict))
    print('Precision:', precision_score(y_test, y_predict))
    print('Recall:', recall_score(y_test, y_predict))
    print('F1:', f1_score(y_test, y_predict))

if __name__ == '__main__':
    TF_IDF_experiment()
    BoW_experiment()