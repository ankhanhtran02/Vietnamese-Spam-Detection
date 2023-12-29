from sklearn.preprocessing import MaxAbsScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import TruncatedSVD
from v_preprocess import preprocess_test_file, preprocess_train, preprocess_test_list
import pickle
import numpy as np


def save_model(vectorizer_file, model_file):
    # Preprocessing training data from file
    X_train_v, y_train = preprocess_train(vectorizer_file=vectorizer_file)

    '''
    KNN Model with SVD and using MaxAbsScaler
    '''
    scaler = MaxAbsScaler()
    X_train_v_scaled = scaler.fit_transform(X_train_v)

    # Perform SVD
    svd = TruncatedSVD(100)
    X_trainSVD = svd.fit_transform(X_train_v_scaled)

    knn = KNeighborsClassifier()

    # Tuning hyperparameters by using cross-validation method
    parameters = {'n_neighbors': [3, 5, 7, 9, 11, 13], 'metric': ['euclidean', 'manhattan', 'cosine']}
    gridSearch = GridSearchCV(knn, param_grid=parameters, cv=5)
    gridSearch.fit(X_trainSVD, y_train)
    print("Best parameters:", gridSearch.best_params_)
    print("Best training Score:", gridSearch.best_score_)

    # Save the 3 model to files
    pickle.dump([scaler, svd, gridSearch], open(model_file, 'wb'))
    print('Saved model.')

def make_predictions(list_inputs: list, vectorizer_file, model_file):
    # Preprocessing testing list
    X_test_v = preprocess_test_list(test_list=list_inputs, vectorizer_file=vectorizer_file)

    # Evaluating testing list
    scaler, svd, gridSearch = pickle.load(open(model_file, 'rb'))
    return [item[0] for item in gridSearch.predict(np.array(svd.transform(scaler.transform(X_test_v)))).reshape(-1, 1)]

def print_report(vectorizer_file, model_file):
    X_test_v, y_test = preprocess_test_file(vectorizer_file=vectorizer_file)
    scaler, svd, gridSearch = pickle.load(open(model_file, 'rb'))
    y_predict = list(gridSearch.predict(np.array(svd.transform(scaler.transform(X_test_v)))).reshape(-1, 1))
    ''' Uncomment the 2 lines below to see confusion matrix and classification report '''
    # print('Confusion matrix:\n', confusion_matrix(y_test, y_predict))
    # print('Classification report:\n', classification_report(y_test, y_predict))
    print('Testing Set Accuracy:', accuracy_score(y_test, y_predict))
    print('Precision:', precision_score(y_test, y_predict))
    print('Recall:', recall_score(y_test, y_predict))
    print('F1:', f1_score(y_test, y_predict))

if __name__ == '__main__':
    print('Evaluation of the algorithm using Count Vectorizer')
    vectorizer_file = 'loaded_models\\count_vectorizer.pkl'
    model_file = 'loaded_models\\count_KNN.pkl'
    print_report(vectorizer_file=vectorizer_file, model_file= model_file)
    print('Evaluation of the algorithm using TF-IDF Vectorizer')
    vectorizer_file = 'loaded_models\\tfidf_vectorizer.pkl'
    model_file = 'loaded_models\\tfidf_KNN.pkl'
    print_report(vectorizer_file=vectorizer_file, model_file= model_file)

