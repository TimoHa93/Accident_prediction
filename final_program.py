import pandas as pd
from preprocess_data import preprocess_data, preprocess_data_to_predict
from neural_net_pred import DeepNeuralNetClassifier
import tensorflow as tf
from sklearn.model_selection import RandomizedSearchCV

#load the data
df_train = pd.read_csv('data/verkehrsunfaelle_train.csv', engine='python', index_col=0)
df_test = pd.read_csv('data/verkehrsunfaelle_test.csv', engine='python', index_col=0)

#get features and the target variable
accidents = df_train.drop('Unfallschwere', axis=1)
accidents_labels = df_train['Unfallschwere'].copy()

#preprocess the training data
model_sel, X_train, X_test, y_train, y_test = preprocess_data(faxaccidents, faxaccidents_labels)

#preprocess the test set
prediction_data = preprocess_data_to_predict(df_test, model_sel)

#initialize the DeepNeuralNetwork
dnn = DeepNeuralNetClassifier(show_progress=None, random_state=42)

#set hyper parameters for RandomizedSearchCV
parameter_distributions = {
    'n_hidden_layers': [3, 4, 5],
    'n_neurons': [40, 50, 100],
    'batch_size': [64, 128],
    'learning_rate':[0.01, 0.005],
    'activation': [tf.nn.elu, tf.nn.relu],
    'max_checks_without_progress': [20, 30],
    'batch_norm_momentum': [None, 0.9],
    'dropout_rate': [None, 0.5]
}

#find the best parameters using RandomizedSearchCV
random_search = RandomizedSearchCV(dnn, parameter_distributions, n_iter=15, scoring='accuracy', verbose=2)
random_search.fit(X_train, y_train)

#save best model
random_search.best_estimator_.save("models/accidents_grid_best_model")

#make predictions
final_predictions = random_search.best_estimator_.predict(prediction_data)
