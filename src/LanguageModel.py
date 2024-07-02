import DataAnalyzer
from enum import Enum
from ast import Import
from gc import callbacks
from pickle import FALSE
import numpy
from sklearn.model_selection import PredefinedSplit
import SignficantEscapeGenerator as SEG
import featurizer as FZ

import keras_tuner as kt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorflow.keras.models import load_model
import pandas as pd
import statistics


# Transformer
from EscapeTransformer import TransformerBlock
from EscapeTransformer import TokenAndPositionEmbedding

MAX_SEQ_LENGTH = 1280


class MODEL_NAME(Enum):
    LSTM = 1
    TRANSFORMER = 2


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Set random seed for fixed operation
tf.random.set_seed(42)


def getModelSavePath():
    return "model/escape_model"  # escape_model_0 #escape_model_1


def getLSTM_Model(input_dimension, output_dimension_of_embedding, input_len_of_seq):
    model = Sequential()
    model.add(Embedding(input_dimension, output_dimension_of_embedding,
              input_length=input_len_of_seq))
    model.add(Dropout(0.3))
    model.add(LSTM(100))
    model.add(Dropout(0.3))
    model.add(Dense(2, activation='sigmoid'))
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model


def train_model(model, x_train, y_train, x_val, y_val, model_save=False):
    NO_OF_EPOCHS = 25  # Orig: 20
    #check_point_dir = os.path.dirname(check_point_path)

    # Creating call back for saving model weight  #verbose = 1 (animate bar) 0 =>silent 2 =>Just epoch number
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True, verbose=1)

    #history = model.fit( x_train, y_train, batch_size=32, epochs=NO_OF_EPOCHS, validation_data=(x_val, y_val), callbacks = cp_callback )
    import PlotLearn_util as plt_util
    #cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=check_point_path, save_weights_only=True, verbose=1)
    stop_early = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5)
    my_callbacks = [plt_util.PlotLearning(), stop_early]
    history = model.fit(x_train, y_train, batch_size=32, epochs=NO_OF_EPOCHS,
                        validation_data=(x_val, y_val), verbose=1, callbacks=my_callbacks)

    if model_save == True:
        model.save(getModelSavePath())

    return model


'''
Returns Loss, Accuracy
'''


def evaluate_model(model, x_test, y_test):
    loss, accuracy = model.evaluate(x_test,  y_test, verbose=2)
    print("Model accuracy: {:5.2f}%".format(100 * accuracy))
    return loss, accuracy


def predict_seqeuence(model, input_sequences):
    predictions = model.predict(input_sequences)
    return predictions


'''
Value             |Best Value So Far |Hyperparameter
56                |8                 |HIDDEN_LAYER_SIZE_TRF
relu              |selu              |activation
0.35              |0.5               |DROPOUT_01
12                |36                |DENSER_LAYER_01
0.07              |0.19              |DROPOUT_02
0.0021663         |0.0025529         |lr
'''


def getTrf_model(input_dimension, embed_dim, max_len_of_input_seq):

    # DROP_OUT_RATE = 0.4  #Best dropout = 0.4 #0.5 #0.3 # #0.6   #(0.1 - original)
    DROP_OUT_RATE_1 = 0.5
    DROP_OUT_RATE_2 = 0.19
    DENSE_LAYER_1 =  36  # (20 -orig)
    FINAL_DENSE_LAYER = 2
    learning_rate = 0.0025529


    num_heads = 2  # Number of attention heads
    ff_dim = 8  # 32  # Hidden layer size in feed forward network inside transformer
    inputs = layers.Input(shape=(max_len_of_input_seq,))



    embedding_layer = TokenAndPositionEmbedding(max_len_of_input_seq, input_dimension, embed_dim)  # input_dimension = vocab size
    x_trf = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x_trf = transformer_block(x_trf)
    x_trf = layers.GlobalAveragePooling1D()(x_trf)
    x_trf = layers.Dropout(DROP_OUT_RATE_1)(x_trf)
    x_trf = layers.Dense(DENSE_LAYER_1, activation="selu")(x_trf)  # relu
    x_trf = layers.Dropout(DROP_OUT_RATE_2)(x_trf)
    outputs = layers.Dense(FINAL_DENSE_LAYER, activation="softmax")(x_trf)

    model = keras.Model(inputs=inputs, outputs=outputs)
    # original :
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # display model summary
    print(model.summary())

    return model


def build_hyperModel(hp):
    num_heads = 2  # Number of attention heads
    ff_dim = hp.Int("HIDDEN_LAYER_SIZE_TRF", min_value=8, max_value=64, step=8)  # 32  # Hidden layer size in feed forward network inside transformer
    inputs = layers.Input(shape=(MAX_SEQ_LENGTH,))

    ACTIVATION = hp.Choice("activation", ["relu", "tanh", "selu"])

    DROPOUT_01 = hp.Float('DROPOUT_01',  min_value=0.1, max_value=0.5, step=0.05)
    DENSER_LAYER_01 = hp.Int('DENSER_LAYER_01', min_value=4, max_value=64, step=8)
    DROPOUT_02 = hp.Float('DROPOUT_02', min_value=0.05, max_value=0.2, step=0.02)
    hp_learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")


    vocab_size, embed_dim = get_input_and_embed_dim()
    embedding_layer = TokenAndPositionEmbedding(MAX_SEQ_LENGTH, vocab_size, embed_dim)  # input_dimension = vocab size
    x_trf = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x_trf = transformer_block(x_trf)
    x_trf = layers.GlobalAveragePooling1D()(x_trf)


    x_trf = layers.Dropout(DROPOUT_01)(x_trf)
    x_trf = layers.Dense(DENSER_LAYER_01, activation=ACTIVATION)(x_trf)
    x_trf = layers.Dropout(DROPOUT_02)(x_trf)
    outputs = layers.Dense(2, activation="softmax")(x_trf)

    model = keras.Model(inputs=inputs, outputs=outputs)

    # original :
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate), loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    # display model summary
    print(model.summary())

    return model


def initialize_escape_tuner(x_train, y_train, x_val, y_val):
    tuner = kt.RandomSearch(build_hyperModel, objective='val_loss', max_trials=3, overwrite=True, directory="tuned", project_name="escape_tuning")  # val_accuracy #
    tuner.search(x_train, y_train, epochs=5, validation_data=(x_val, y_val))
    # Get search space summary
    tuner.search_space_summary()
    models = tuner.get_best_models(num_models=2)
    best_model = models[0]

    return best_model


def train_using_tuned_model():
    tuner = kt.RandomSearch(build_hyperModel, objective='val_loss',  overwrite=False, directory="tuned", project_name="escape_tuning")
    #trial = tuner.oracle.get_trial('tuned/escape_tuning/trial_2')
    model = tuner.hypermodel.build(tuner.get_best_hyperparameters())
    x_train, y_train = FZ.get_train_encoded_dataset()
    x_val, y_val = FZ.get_validation_encoded_dataset()
    x_test, y_test = FZ.get_test_encoded_dataset()

    train_model(model, x_train, y_train, x_val, y_val)
    # Validation Accuracy
    print("Test Dataset Accuracy: *********")
    evaluate_model(model, x_test, y_test)


def distribute_dataset(lst, train_ratio= 0.7, val_ratio=0.15):
    train_size  = int(len(lst)*train_ratio)
    val_size = train_size + int (len(lst)*val_ratio)
    train = lst[0: train_size]
    val = lst[train_size: val_size]
    test = lst[val_size:]
    return train, val, test


def train_escape_network(model_name= MODEL_NAME.TRANSFORMER):
    x_train, y_train = FZ.get_train_encoded_dataset()
    x_val, y_val = FZ.get_validation_encoded_dataset()
    x_test, y_test = FZ.get_test_encoded_dataset()
    #print("Final input shape: {} Output Shape: {}".format(X.shape, Y.shape))

    #x_train, y_train, x_val, y_val = get_train_valid_test_split(X, Y)
    # Save trainig and validation encoded dataset

    input_dimension, embed_dim = get_input_and_embed_dim()
    if model_name == MODEL_NAME.TRANSFORMER:
        model = getTrf_model(input_dimension, embed_dim, MAX_SEQ_LENGTH)
    else:
        model = getLSTM_Model(input_dimension, embed_dim, MAX_SEQ_LENGTH)
    train_model(model, x_train, y_train, x_val, y_val)


    # Validation Accuracy
    print("Validation Dataset Accuracy: *********")
    evaluate_model(model, x_test, y_test)

    model.save(getModelSavePath())


def train_escape_network_using_raw_data():
    sig_path = "/home/perm/cov/data/gen/cdhit/sig_1652837548.fas.1"
    nonSig_path = "/home/perm/cov/data/gen/cdhit/nonsig_1652840459.fas.1"
    X, Y = FZ.get_training_data(sig_path, nonSig_path)
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

    input_dimension, embed_dim = get_input_and_embed_dim()
    model = getTrf_model(input_dimension, embed_dim, MAX_SEQ_LENGTH)
    train_model(model, X_train, y_train, X_test, y_test)



    # Validation Accuracy
    print("Validation Dataset Accuracy: *********")
    evaluate_model(model, X_test, y_test)
    model.save(getModelSavePath())
    pass


def train_escape_network_using_kfold():
    sig_path = SEG.getGsaid_significantSeqs_sample_path()
    nonSig_path = SEG.getGsaid_non_significant_seq_path()
    X, Y = FZ.get_training_data(sig_path, nonSig_path)
    Y = Y.reshape(-1, 1)

    from sklearn.model_selection import StratifiedKFold
    no_of_folds = 3  # Split size for test will be 1/N so, 33% 
    kFold = StratifiedKFold(no_of_folds, shuffle=True, random_state= 42)

    kfold_acc = []
    for train_indices, test_indices in kFold.split(X, Y):
        x_train = X[train_indices]
        y_train = Y[train_indices]

        x_val = X[test_indices]
        y_val = Y[test_indices]

        input_dimension, embed_dim = get_input_and_embed_dim()
        model = getTrf_model(input_dimension, embed_dim, MAX_SEQ_LENGTH)
        train_model(model, x_train, y_train, x_val, y_val)

        print("Validation Dataset Accuracy: *********")
        loss, acc = evaluate_model(model, x_val, y_val)
        kfold_acc.append(acc)
    
    mean_acc = statistics.mean(kfold_acc)
    print("Individual accuracies of model: ", kfold_acc )
    print("***Mean Accuracy of K-Fold model is: ", mean_acc)
    


def construct_dataset():
    sig_path = SEG.getGsaid_significantSeqs_sample_path()
    nonSig_path = SEG.getGsaid_non_significant_seq_path()
    X, Y = FZ.get_training_data(sig_path, nonSig_path)

    train_ration = 0.6
    val_ration = 0.2
    x_train, x_val, x_test =  distribute_dataset(X, train_ration, val_ration)
    y_train, y_val, y_test = distribute_dataset(Y, train_ration, val_ration)

    #x_test, y_test = FZ.get_test_data()
    FZ.save_dataset((x_train, y_train), (x_val, y_val), (x_test, y_test ))
    print("Test Train and Validation constructed data saved successfully.")


def get_input_and_embed_dim():
    vocab_size = len(FZ.getVocabDictionary())  # size  #Embedding(inputDIM=vocaB_size, output_dim=Size_Of_vector_space, input_length=Length of input sequences)
    embed_dim = 20  # Embedded length of output
    return vocab_size, embed_dim


def get_train_valid_test_split(X, Y, max_len_of_input_seq):
    from sklearn.model_selection import train_test_split
    x_train, x_val, y_train, y_val = train_test_split(
        X, Y, test_size=0.3, stratify=Y, random_state=42)

    return x_train, y_train,x_val,y_val

def load_trained_model():
    return load_model(getModelSavePath())


def get_max_seq_length(X):
    max_len_of_input_seq =  max(len(row) for row in X)
    return max_len_of_input_seq


def analyze_test_predictions():
    model = load_trained_model()
    x_test, y_test = FZ.get_test_encoded_dataset()
    print(f"Shape of X_val: {x_test.shape} , y_val: {y_test.shape}")
    y_pred = predict_seqeuence(model, x_test)
    DataAnalyzer.plot_confusionMatrix(y_test, y_pred, 'confusion_matrix_test-dataset.png')
    DataAnalyzer.plot_AUC_curve(y_test, y_pred, 'auc_test-dataset.png')
    DataAnalyzer.plot_Precision_recall_curve(y_test, y_pred, 'precision_recall_test-dataset.png')
    return y_test, y_pred


def analyze_validation_predictions():
    model = load_trained_model()
    x_test, y_test = FZ.get_validation_encoded_dataset()
    print(f"Shape of X_val: {x_test.shape} , y_val: {y_test.shape}")
    y_pred = predict_seqeuence(model, x_test)
    DataAnalyzer.plot_AUC_curve(y_test, y_pred, 'auc_validation-dataset_may_12.png')
    DataAnalyzer.plot_confusionMatrix(y_test, y_pred, 'confusion_matrix_validation-dataset_may_12.png')


def evaluate_baum_dataset(csv_file_name):
    print("Evaluation Baum Predictions")
    model = load_trained_model()
    X, Y = FZ.get_baum_test_dataset(csv_file_name)
    evaluate_model(model, X, Y)


def evaluate_greany_gisaid_dataset():
    print("Evaluation Greany Predictions")
    model = load_trained_model()
    X, Y = FZ.get_greany_gisaid_dataset()
    evaluate_model(model, X, Y)

    pass
def evaluate_greany_science_dataset(GREANY_SCIENCE_CSV_FILE_NAME):
   
    print("Evaluation Greany Science Predictions for file: ", GREANY_SCIENCE_CSV_FILE_NAME)
    model = load_trained_model()
    X, Y = FZ.get_greany_science_dataset(GREANY_SCIENCE_CSV_FILE_NAME)
    evaluate_model(model, X, Y)

    pass



def evaluate_test_dataset():
    print("Evaluating test dataset")
    model = load_trained_model()
    X, Y = FZ.get_test_encoded_dataset()

    evaluate_model(model, X, Y)


def evaluate_datasets():
    evaluate_test_dataset()
    evaluate_baum_dataset("baum.csv")
    evaluate_greany_gisaid_dataset()
def test():
    print("Langauge model test function called")

if __name__ == '__main__':

    # construct_dataset()
    #train_escape_network(model_name= MODEL_NAME.TRANSFORMER)
    #train_escape_network(model_name= MODEL_NAME.LSTM)

    #analyze_test_predictions()
    #analyze_greany_predictions()
    # analyze_validation_predictions()
    evaluate_test_dataset()
    #evaluate_baum_dataset("baum.csv")
    #evaluate_greany_gisaid_dataset()
    #evaluate_greany_science_dataset("greany_science_0.3.csv")
    #evaluate_greany_science_dataset("greany_science_0.5.csv")
    # evaluate_datasets()
    #train_escape_network_using_kfold()
    
    pass
