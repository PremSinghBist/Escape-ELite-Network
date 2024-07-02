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
import numpy as np


# Transformer
from EscapeTransformer import TransformerBlock
from EscapeTransformer import TokenAndPositionEmbedding

from tensorflow.keras.models import load_model

import FE_Lang_Embed_Trainer  as eTrainer

MODEL_NAME = "FE_TRNSF_MODEL"
INPUT_DIM= 45056 #No of fetures 
VOCAB_SIZE = 25
EMBED_DIM = 20

# baseline model
def get_base_model():
    # DROP_OUT_RATE = 0.4  #Best dropout = 0.4 #0.5 #0.3 # #0.6   #(0.1 - original)
    DROP_OUT_RATE_1 = 0.5
    DROP_OUT_RATE_2 = 0.19
    DENSE_LAYER_1 =  36  # (20 -orig)
    FINAL_DENSE_LAYER = 2
    learning_rate = 0.0025529

    num_heads = 2  # Number of attention heads
    ff_dim = 8  # 32  # Hidden layer size in feed forward network inside transformer
    inputs = layers.Input(shape=(INPUT_DIM,))

    embedding_layer = TokenAndPositionEmbedding(INPUT_DIM, VOCAB_SIZE, EMBED_DIM)  # input_dimension = vocab size
    x_trf = embedding_layer(inputs)
    transformer_block = TransformerBlock(EMBED_DIM, num_heads, ff_dim)
    x_trf = transformer_block(inputs)
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

def get_training_data():
    sig_features, non_sig_features = eTrainer.get_features(eTrainer.MODEL_LSTM_OUTPUT_FEATURE_SAVE_PATH)

    sig_len =  len(sig_features)
    sig_features_output = np.ones( (sig_len, 1) )
    print("Sig feature shape: ", sig_features.shape)
    print("Sig output shape: ", sig_features_output.shape)
    sig_features = np.concatenate( (sig_features, sig_features_output) , axis=1) #Concate across last dimension
    print("Sig final shape: ", sig_features.shape)

    non_sig_len = len(non_sig_features)
    non_sig_features_output = np.zeros( (non_sig_len, 1))
    non_sig_features = np.concatenate( (non_sig_features, non_sig_features_output) , axis=1) #Concate across  last dimension
    print("Non Sig final shape: ", non_sig_features.shape)
    
    features  = np.concatenate( (sig_features, non_sig_features) , axis = 0)  #Concate both features and randomize for training 
    print(f"Combined features shape: {features.shape} ")

    #Randomize the features: 
    np.random.shuffle(features)
    print(f"After shuffle features shape: {features.shape} ")

    from sklearn.model_selection import train_test_split
    train_data, test_data = train_test_split(features, test_size=0.33, random_state=42)
    print(f'Shape of train_data : {train_data.shape} Test Shape: {test_data.shape}')

    return train_data, test_data
def evaluate_greany():
    model =load_model(f"model/{MODEL_NAME}")
    sig_features = eTrainer.get_greany_model_lstm_output_features()
    sig_len =  len(sig_features)
    sig_features_output = np.ones( (sig_len, 1) )

    print(f'Shape: Greany input feature {sig_features.shape}, output : {sig_features_output.shape}')

    #Evaluating greany fetures 
    print("Evaluating greany model fetures")
    model.evaluate(sig_features, sig_features_output)
def sample_data():
    train_data, test_data = get_training_data()
    print("Train data shape: ", train_data.shape)
    
    train_x = train_data[:, 0:INPUT_DIM]   
    print("Shape of train X: ", train_x.shape)
    train_y = train_data[:, -1]
    print("Shape of train Y: ", train_y.shape)

    print("First sample X: ", train_x[0])
    print("First sample Y: ", train_y[0])

def train_model():
    train_data, test_data = get_training_data()
    print("Train data shape: ", train_data.shape)
    
    train_x = train_data[:, 0:INPUT_DIM]   
    print("Shape of train X: ", train_x.shape)
    train_y = train_data[:, -1]
    print("Shape of train Y: ", train_y.shape)

    test_x = test_data[:, 0:INPUT_DIM]
    test_y = test_data[:, -1] #last column record


    
    model = get_base_model()
    
    model.fit(train_x, train_y, batch_size=512, epochs=40, validation_data=(test_x, test_y), verbose=1)
    model.save(f'model/{MODEL_NAME}')
    return model

if __name__ == "__main__":
    # sample_data()
    # exit()
    model = train_model()
    evaluate_greany()
