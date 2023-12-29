# Import the libraries
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from v_preprocess import preprocess_test_list, preprocess_test_file, preprocess_train
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

def save_model(vectorizer_file, model_file):
    # Preprocessing training data from file
    X_train_v, y_train = preprocess_train(vectorizer_file=vectorizer_file)

    # Define the Keras model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(2),
        tf.keras.layers.Activation(tf.nn.sigmoid)
    ])

    model.compile(optimizer='adam',
                loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

    model.fit(X_train_v, y_train, epochs=10)
    train_loss, train_acc = model.evaluate(X_train_v, y_train, verbose=2)
    print('\nTrain accuracy:', train_acc)
    print('\nTrain loss:', train_loss)

    # Save the model to disk
    model.save(model_file)
    print('Saved model.')

def make_predictions(list_inputs: list, vectorizer_file, model_file):
    X_test_v = preprocess_test_list(test_list = list_inputs, vectorizer_file=vectorizer_file)
    model = tf.keras.models.load_model(model_file)
    prediction_prob = model.predict(X_test_v)
    y_predict = [np.argmax(prediction_prob[i]) for i in range(prediction_prob.shape[0])]
    return y_predict

def print_report(vectorizer_file, model_file):
    X_test_v, y_test = preprocess_test_file(vectorizer_file=vectorizer_file)
    model = tf.keras.models.load_model(model_file)
    prediction_prob = model.predict(X_test_v)
    y_predict = [np.argmax(prediction_prob[i]) for i in range(prediction_prob.shape[0])]
    ''' Uncomment the 2 lines below to see confusion matrix and classification report '''
    # print('Confusion matrix:\n', confusion_matrix(y_test, y_predict))
    # print('Classification report:\n', classification_report(y_test, y_predict))
    print('Testing Set Accuracy:', accuracy_score(y_test, y_predict))
    print('Precision:', precision_score(y_test, y_predict, zero_division=0))
    print('Recall:', recall_score(y_test, y_predict,zero_division=0))
    print('F1:', f1_score(y_test, y_predict,zero_division=0))

if __name__ == '__main__':
    vectorizer_file = 'loaded_models\\count_vectorizer.pkl'
    model_file = 'loaded_models\\count_ANN.h5'
    print_report(vectorizer_file=vectorizer_file, model_file= model_file)
