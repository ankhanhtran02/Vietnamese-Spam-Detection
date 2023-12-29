
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix
from v_preprocess import preprocess_train, preprocess_test_file, preprocess_test_list
import pickle

def save_model(vectorizer_file, model_file):
    # Preprocessing training data from file
    X_train_v, y_train = preprocess_train(vectorizer_file=vectorizer_file)
    model = LinearSVC(dual="auto", fit_intercept=True, random_state=1)
    '''
    dual = "auto": select the algorithm to solve either the dual or primal optimization problem
    fit_intercept: allow exceptions to be made
    random_state: number of randoms in the model
    '''
    model.fit(X_train_v, y_train)
    pickle.dump(model, open(model_file,'wb'))
    print('Saved model.')

def make_predictions(list_inputs: list, vectorizer_file, model_file):
    X_test_v = preprocess_test_list(test_list=list_inputs, vectorizer_file=vectorizer_file)
    model = pickle.load(open(model_file,'rb'))
    y_predict = model.predict(X_test_v)
    return y_predict

def print_report(vectorizer_file, model_file):
    X_test_v, y_test = preprocess_test_file(vectorizer_file=vectorizer_file)
    model = pickle.load(open(model_file, 'rb'))
    y_predict = model.predict(X_test_v)
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
    model_file = 'loaded_models\\count_SVM.pkl'
    print_report(vectorizer_file=vectorizer_file, model_file= model_file)
    print('Evaluation of the algorithm using TF-IDF Vectorizer')
    vectorizer_file = 'loaded_models\\tfidf_vectorizer.pkl'
    model_file = 'loaded_models\\tfidf_SVM.pkl'
    print_report(vectorizer_file=vectorizer_file, model_file= model_file)
