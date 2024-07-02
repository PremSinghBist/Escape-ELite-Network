import sys
import os

# os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
os.environ["CUDA_VISIBLE_DEVICES"]="4"



import tensorflow as tf
from tensorflow.keras import Input
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    concatenate, Activation, Dense, Embedding, LSTM, Reshape)
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.models as m
from tensorflow.keras.preprocessing.sequence import pad_sequences



import random
import numpy as np
from transformers import FEATURE_EXTRACTOR_MAPPING
np.random.seed(1)
import math
import pandas as pd
import matplotlib.pyplot as plt

import AdditionalSignficantEscapeSeqGenerator as ASE
import featurizer as FZ
import DataAnalyzer
FEATURE_LEARNING_MODEL_PATH = "/home/perm/sars_escape_netv2/model/feature_learning"

#m2 best model
# DISCRIMINATOR_LEARNING_MODEL_PATH = "/home/perm/sars_escape_netv2/model/disc_learning"
DISCRIMINATOR_LEARNING_MODEL_PATH = "/home/perm/sars_escape_netv2/model/sarsx_disc"
BASE_DATA_GEN_PATH = "data/gen/"
# FEATURE_EXTRACTION_SRC_PATH = "data/additional_escape_variants/gen"
# FEATURE_EXTRACTION_SAVE_PATH = "data/additional_escape_variants/gen"
# BASE_ANALYSIS_PATH = "/home/perm/sars_escape_netv2/data/results"  #"data/analysis/escape_gan"
BASE_ANALYSIS_PATH = "/home/perm/sars_escape_netv2/data/resultsx" 
# GAN_AUG_SEQS = "data/gen/escape_gan_seqs/12000.csv" #data/gen/escape_gan_seqs/fs4000_full.csv" 

TEST_DATA_PATH="/home/perm/sars_escape_netv2/data/test"
DISC_EPOCHS = 200 #15 #200
BATCH_SIZE = 128


from sklearn.model_selection import train_test_split






#LStm BASED FEATURES
# MODEL_LSTM_OUTPUT_FEATURE_ESC_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_features_esc.npz"
# MODEL_LSTM_OUTPUT_GISAID_FEATURE_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_output_gisaid_features.npz"
# MODEL_LSTM_OUTPUT_GREANY_FEATURE_SAVE_PATH = FEATURE_EXTRACTION_SRC_PATH + "/model_lstm_output_greany_features.npz"

from Bio import SeqIO


# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#     tf.config.set_visible_devices(gpus[2], 'GPU')
# else:
#     print("No GPUs available.")

'''
lengths : array containing sequence length for individual sequence [1152, 1205, ..., 1425]
seq_len : Maximum Length  of sequence (taken into consideration )
returns start index and end index in the form of generator and updates the current index  
'''
def iterate_lengths(lengths, seq_len):

    curr_idx = 0
    for length in lengths:
        if length > seq_len:
            sys.stderr.write(
                'Warning: length {} greather than expected '
                'max length {}\n'.format(length, seq_len)
            )
        yield (curr_idx, curr_idx + length)
        curr_idx += length




def cross_entropy(logprob, n_samples):
    return -logprob / n_samples

def report_performance(model_name, model, vocabulary, train_seqs, test_seqs):
    X_train, lengths_train = model.featurize_seqs(train_seqs, vocabulary)
    logprob = model.score(X_train, lengths_train)
    print('Model {}, train cross entropy: {}'.format(model_name, cross_entropy(logprob, len(lengths_train))))
    X_test, lengths_test = model.featurize_seqs(test_seqs, vocabulary)
    logprob = model.score(X_test, lengths_test)
    print('Model {}, test cross entropy: {}'.format(model_name, cross_entropy(logprob, len(lengths_test))))
def featurize_seqs(seqs):
        vocabulary = FZ.getVocabDictionary()
        start_int = len(vocabulary) + 1
        end_int = len(vocabulary) + 2
        sorted_seqs = sorted(seqs)
        X = np.concatenate([
            np.array([ start_int ] + [
                vocabulary[word] for word in seq
            ] + [ end_int ]) for seq in sorted_seqs
        ]).reshape(-1, 1)
        lens = np.array([ len(seq) + 2 for seq in sorted_seqs ])
        assert(sum(lens) == X.shape[0])
        return X, lens
def batch_train(model, seqs,  batch_size=512,  n_epochs=1):
    # Control epochs here.
    model.n_epochs_ = n_epochs
    n_batches = math.ceil(len(seqs) / float(batch_size))
    print('Traning seq batch size: {}, N batches: {}'.format(batch_size, n_batches))
    for epoch in range(n_epochs):
        random.shuffle(seqs)
        for batchi in range(n_batches):
            start = batchi * batch_size
            end = (batchi + 1) * batch_size
            seqs_batch =  seqs[start:end] 
            X, lengths = featurize_seqs(seqs_batch)
            model.fit(X, lengths)
            del seqs_batch

def train_embedding_model(seqs, epochs=1):
    seq_len = 23
    vocab_size = 27
    model = get_model_structure(seq_len, vocab_size) ##63 
    
    batch_train(model, seqs, batch_size=BATCH_SIZE, n_epochs=epochs) #Batch size: 32
   
      
def get_model_structure(seq_len=23, vocab_size=27):
        from BiLSTMLanguageModeler import BiLSTMLanguageModel
        model = BiLSTMLanguageModel(
            seq_len,
            vocab_size,
            embedding_dim=20,
            hidden_dim=256,
            n_hidden=2,
            n_epochs=1,
            batch_size=24,
            inference_batch_size=24,
            cache_dir=FEATURE_LEARNING_MODEL_PATH,
            seed=41,
            verbose=True,
        )
        model.model_.summary()
        return model                  
'''
seq_of_interest - Any single sequence under observation example : wild sequence 
'''
def predict_sequence_prob(seq_of_interest, model):
   
    X_cat, lengths = featurize_seqs(seq_of_interest)
    y_pred = model.predict(X_cat, lengths)
    print("Original y_pred shape: ",y_pred.shape)

    y_reshaped = np.reshape(y_pred, (-1, 22*28))
    print("Original y_pred shape: ",y_reshaped.shape)

    return y_reshaped




def load_fz_model():
    model = get_model_structure()
    
    #new model 
    model_path = FEATURE_LEARNING_MODEL_PATH+"/bilstm_256-08.hdf5"
    model.model_.load_weights(model_path)

    '''
    #old  model
    model_path = "/home/perm/sars_escape_netv2/model/Sen_old_model/pretrain_models/feature_learning_model/bilstm_256-04.hdf5"
    model.model_.load_weights(model_path)'''
    # sars_gan_old_generative_model = "/home/perm/cov/model/escape_gan/archives/m2/feature_learning/checkpoints/bilstm/bilstm_256-01.hdf5"
    
    print('********------Feature Learning Model Path ---------------* ', model_path)
    return model

def get_features(windowed_file):
    model =load_fz_model()
    seqs = ASE.read_combined_window_file(windowed_file)

    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape) #  (rows, 22, 512)
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    print("Reduced feature Shape after average: ", y_embed_output.shape)
    return y_embed_output
def get_single_window_feature(windowed_file):
    model =load_fz_model()
    seqs = ASE.read_window_file_using_column_name(windowed_file, 'window_seqs')

    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape) #  (rows, 22, 512)
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    print("Reduced feature Shape after average: ", y_embed_output.shape)
    return y_embed_output
    

def get_single_feature(input_window):
    model =load_fz_model()
    seqs = [input_window]

    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape) #  (rows, 22, 512)
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    print("Reduced feature Shape after average: ", y_embed_output.shape)
    return y_embed_output




def get_balanced_datasets_over_sampling(non_sig_train, non_sig_test, sig_train_orig, sig_test):
    BATCH_SIZE = 128
    SHUFFLE_BUFFER_SIZE = 40166 #366,533 + 40166
    #NOn Sig train test features
    non_sig_train_len = non_sig_train.shape[0]
    non_sig_features_train_output = np.zeros( (non_sig_train_len, 1))
    print(f" non_sig_train shape: {non_sig_train.shape} Non Sig test shape {non_sig_test.shape} ")
    non_sig_train_dataset = tf.data.Dataset.from_tensor_slices((non_sig_train, non_sig_features_train_output))

    non_sig_test_len = non_sig_test.shape[0]
    non_sig_test_features_output = np.zeros( (non_sig_test_len, 1))
    non_sig_test_dataset = tf.data.Dataset.from_tensor_slices((non_sig_test, non_sig_test_features_output))
    

    #Over sample sig train features 
    sig_train_orig = np.reshape(sig_train_orig, (-1, 512))
    sig_train_features = over_sample_array(sig_train_orig, total_samples = non_sig_train_len)
    sig_train_len =  sig_train_features.shape[0]
    print("Sig Train shape: ",sig_train_features.shape )
    sig_train_features_output = np.ones( (sig_train_len, 1) )
    sig_train_dataset = tf.data.Dataset.from_tensor_slices((sig_train_features, sig_train_features_output))

    sig_test = np.reshape(sig_test, (-1, 512))
    print("Sig Test shape: ",sig_test.shape)
    sig_test_len =  sig_test.shape[0]
    sig_test_features_output = np.ones( (sig_test_len, 1) )
    sig_test_dataset = tf.data.Dataset.from_tensor_slices((sig_test, sig_test_features_output))

    combined_train_ds = sig_train_dataset.concatenate(non_sig_train_dataset)
    combined_train_ds = combined_train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    train_dataset_cardinality = combined_train_ds.cardinality().numpy()
    print(f"Combined Train features Cardinality: {train_dataset_cardinality} ") #No of rows returns 

    combined_test_ds = sig_test_dataset.concatenate(non_sig_test_dataset)
    combined_test_ds = combined_test_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    test_dataset_cardinality = combined_test_ds.cardinality().numpy()
    print(f"Combined Train features Cardinality: {train_dataset_cardinality} ") #No of rows returns 


    return combined_train_ds, combined_test_ds

def over_sample_array(arr, total_samples=0):
    #replace True oversamples data
    over_samples = arr[np.random.choice(len(arr), size=total_samples, replace=True)]
    print("Oversamples shape is: ",over_samples.shape)
    return over_samples 

def get_dense_model():
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense 
    from tensorflow.keras.layers import BatchNormalization 
    INPUT_FEATURE_SIZE = 512
    model = Sequential()
    model.add(layers.Input(INPUT_FEATURE_SIZE, ))
    model.add(BatchNormalization())
    model.add(Dense(1, activation='sigmoid'))
    optmz = tf.keras.optimizers.Adam(
        name = "Adam",
        learning_rate = 0.00001
    )
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optmz, metrics=['accuracy', tf.keras.metrics.AUC()])
    model.summary()
    print(model.optimizer.get_config())
    
    return model

def get_dense_model_regularized():
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense 
    from tensorflow.keras.layers import BatchNormalization 
    from tensorflow.keras import regularizers
    INPUT_FEATURE_SIZE = 512
    regularizing_factor = 0.01
    model = Sequential()
    model.add(layers.Input(INPUT_FEATURE_SIZE, ))
    model.add(BatchNormalization())
     #regularizers.L2(regularizing_factor)
     #regularizers.L1(regularizing_factor)
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.L1L2(regularizing_factor)))
    optmz = tf.keras.optimizers.Adam(
        name = "Adam",
        learning_rate = 0.00001
    )
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optmz, metrics=['accuracy', tf.keras.metrics.AUC()])
    model.summary()
    print(model.optimizer.get_config())
    
    return model
def get_dense_model_hybridized():
    from tensorflow.keras import layers
    from tensorflow.keras.layers import Dense 
    from tensorflow.keras.layers import BatchNormalization 
    from tensorflow.keras import regularizers
    INPUT_FEATURE_SIZE = 512
    regularizing_factor = 0.01
    model = Sequential()
    model.add(layers.Input(INPUT_FEATURE_SIZE, ))
    model.add(BatchNormalization())
     #regularizers.L2(regularizing_factor)
     #regularizers.L1(regularizing_factor)
     #regularizers.L1L2(regularizing_factor)
    model.add(Dense(64, activation='sigmoid', kernel_regularizer=regularizers.L1(regularizing_factor)))
    model.add(Dense(1, activation='sigmoid', kernel_regularizer=regularizers.L1(regularizing_factor)))
    optmz = tf.keras.optimizers.Adam(
        name = "Adam",
        learning_rate = 0.00001
    )
    # Compile model
    model.compile(loss='binary_crossentropy', optimizer=optmz, metrics=['accuracy', tf.keras.metrics.AUC()])
    model.summary()
    print(model.optimizer.get_config())
    
    return model

def analyze_greany():
    DISCRIMINATOR_LEARNING_MODEL_PATH="/home/perm/sars_escape_netv2/data/model_results_archive/M5/sarsx_disc"
    model = m.load_model(DISCRIMINATOR_LEARNING_MODEL_PATH) 
    model.summary()
    sig_data = get_features(TEST_DATA_PATH+"/greany_sig_combined_windowed_seqs.csv")
    non_sig_data =  get_features(TEST_DATA_PATH+"/greany_non_sig_combined_windowed_seqs.csv")

    sig_len =  len(sig_data)
    sig_features_output = np.ones( (sig_len, 1) )
    non_sig_len = len(non_sig_data)
    nonsig_features_output = np.zeros( (non_sig_len, 1) )

    features = np.concatenate( (sig_data, non_sig_data), axis=0)
    targets = np.concatenate( (sig_features_output, nonsig_features_output), axis=0 )

    print("Evaluating  greany Combined ")
    model.evaluate(features, targets)
    
    #print("Evaluating  greany Sig only ")
    #model.evaluate(sig_data, sig_features_output)
    #print("Evaluating  greany Non Sig only ")
    #model.evaluate(non_sig_data, nonsig_features_output)
    '''
    #Analyze greany predictions | plot AUC_ROC / Confusion Matrix / Precision Recall
    y_preds = model.predict(features)
    save_results_to_csv(targets, y_preds, BASE_ANALYSIS_PATH+"/greany_preds.csv")
    # DataAnalyzer.plot_confusionMatrix_using_csv(BASE_ANALYSIS_PATH+"/greany_preds.csv", 
    #                                             BASE_ANALYSIS_PATH+"/greaney_confusion.png")
    
    #plot_greany_test_predictions(targets, y_preds)
    return targets, y_preds
    '''

def evaluate_baum():
    model = m.load_model(DISCRIMINATOR_LEARNING_MODEL_PATH)
    sig_data = get_features(TEST_DATA_PATH+"/baum_sig_combined_windowed_seqs.csv")
    non_sig_data = get_features(TEST_DATA_PATH+"/baum_non_sig_combined_windowed_seqs.csv")

    sig_len =  len(sig_data)
    sig_features_output = np.ones( (sig_len, 1) )
    non_sig_len = len(non_sig_data)
    nonsig_features_output = np.zeros( (non_sig_len, 1) )

    features = np.concatenate( (sig_data, non_sig_data), axis=0)
    targets = np.concatenate( (sig_features_output, nonsig_features_output), axis=0 )

    #Evaluating greany fetures 
    print("Evaluating  Baum Combined ")
    model.evaluate(features, targets)
    

    print("Evaluating  Baum Sig only ")
    model.evaluate(sig_data, sig_features_output)

    print("Evaluating  Baum Non Sig only ")
    model.evaluate(non_sig_data, nonsig_features_output)

    y_preds = model.predict(features)
    plot_baum_predictions(targets, y_preds)
    save_results_to_csv(targets, y_preds, BASE_ANALYSIS_PATH+"/baum_preds.csv")

    return targets, y_preds

def plot_integrated_auc():
    v_targets, v_pred = analyze_validation_dataset()
    b_targets, b_pred = evaluate_baum()
    g_targets, g_pred = analyze_greany()

    result_table = pd.DataFrame(columns=['classifiers', 'fpr','tpr','auc'])

    result_table = compute_auc_stats(v_targets, v_pred, result_table, 'Validation')
    result_table = compute_auc_stats(b_targets, b_pred, result_table, 'Baum' )
    result_table = compute_auc_stats(g_targets, g_pred, result_table, 'Greaney')

    result_table.set_index('dataset', inplace=True)
    plot_and_save_fig(result_table)

    

def plot_and_save_fig(result_table):
    fig = plt.figure(figsize=(8,6))
    for i in result_table.index:
        plt.plot(result_table.loc[i]['fpr'], 
                result_table.loc[i]['tpr'], 
                label="{}, AUC={:.3f}".format(i, result_table.loc[i]['auc']))
    plt.plot([0,1], [0,1], color='orange', linestyle='--')

    plt.xticks(np.arange(0.0, 1.1, step=0.1))
    plt.xlabel("False Positive Rate", fontsize=12)
    plt.yticks(np.arange(0.0, 1.1, step=0.1))
    plt.ylabel("True Positive Rate", fontsize=12)

    plt.title('ROC Curve Analysis', fontweight='bold', fontsize=13)
    plt.legend(prop={'size':10}, loc='lower right')

    fig.savefig(BASE_ANALYSIS_PATH+'/integrated_AUC.png') 
    print("Integrated AUC successfully plotted !!!")


def compute_auc_stats(target, predicted, result_table, dataset_name):
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(target,  predicted)
    auc = roc_auc_score(target, predicted)

    result_table = result_table.append({'dataset': dataset_name,
                                        'fpr':fpr, 
                                        'tpr':tpr, 
                                        'auc':auc}, ignore_index=True)

    return result_table

def featurize_sarx(sequences):
    fz_model  = load_fz_model()
    X_cat, lengths  = featurize_seqs(sequences)
    y_embed_output = fz_model.transform(X_cat, lengths)
    print("Shape of output previously: ", y_embed_output.shape) 
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    return y_embed_output

def read_non_sig_seq_from_sars():
    '''
    Retrieves nonsignificant seqeunces from previous database or sars escape
    All Duplicates and Untranslated residues X removed 
    '''
    untranslated_count = 0
    unique_windows = set()
    train_path = "/home/perm/sars_escape_netv2/data/disc/train/non_sig_combined_windows_train.csv"
    test_path = "/home/perm/sars_escape_netv2/data/disc/train/sig_combined_windows_test.csv"
    
    df1  = pd.read_csv(train_path)
    df2  = pd.read_csv(test_path)
    
    df1 = pd.concat([df1, df2], ignore_index=True)
    
    train_wild = list(df1['wild'])
    train_mutated = list(df1['mutated'])
    print(f"wild seq: {len(train_wild)} Mut: {len(train_mutated)}")
    concatenated_seqs  =train_wild + train_mutated
    print(f' After concatenation lenght: {len(concatenated_seqs)}')
    
    for seq in concatenated_seqs:
        if str(seq).__contains__('X'):
            untranslated_count+=1
            continue
        
        unique_windows.add(seq)

    print(f' Length of unique windows: {len(unique_windows)}')
    unique_windows = list(unique_windows)
    return unique_windows

def read_non_sig_seq_from_sars1():
    '''
    Retrieves nonsignificant seqeunces from previous database or sars escape 
    Untranslated residues X Not removed.
    '''
    unique_windows = set()
    train_path = "/home/perm/sars_escape_netv2/data/disc/train/non_sig_combined_windows_train.csv"
    test_path = "/home/perm/sars_escape_netv2/data/disc/train/sig_combined_windows_test.csv"
    
    df1  = pd.read_csv(train_path)
    df2  = pd.read_csv(test_path)
    
    df1 = pd.concat([df1, df2], ignore_index=True)
        
    train_wild = list(df1['wild'])
    train_mutated = list(df1['mutated'])
    print(f"wild seq: {len(train_wild)} Mut: {len(train_mutated)}")
    concatenated_seqs  = train_mutated + train_wild 
    print(f' After concatenation lenght: {len(concatenated_seqs)}')
    
    for seq in concatenated_seqs:
        unique_windows.add(seq)

    print(f' Length of unique windows: {len(unique_windows)}')
    unique_windows = list(unique_windows)
    return unique_windows
    

def execute_discriminator_network_sarsx():
    base_train_path = "/home/perm/sars_escape_netv2/data/disc/train/sarx"
    non_sig_sarsx_path = base_train_path+"/non-significantx.csv"
    sig_sarsx_path= base_train_path+"/significantx.csv"

    df1 = pd.read_csv(sig_sarsx_path)
    sig_seqs = list(df1['window_seqs'])

    df2 = pd.read_csv(non_sig_sarsx_path)
    non_sig_seqs = list(df2['window_seqs'])
    
    
    non_sig_from_sars = read_non_sig_seq_from_sars1() #read_non_sig_seq_from_sars()
    non_sig_seqs = non_sig_seqs + non_sig_from_sars #Add sequneces from both
    
    #overasample non_significatan sequences using numpy oversampling
    non_sig_seqs = np.random.choice(non_sig_seqs, size=len(sig_seqs), replace=True)
    print(f'Lenght of non sig seq after oversampling :{len(non_sig_seqs)}')
    
    print(f'Sig seq len: {len(sig_seqs)} , Non sig total: {len(non_sig_seqs)} type: {type(non_sig_seqs)} ')
    rand_sig_indexes  = np.random.randint(0, len(sig_seqs), size=len(non_sig_seqs))
    sig_samples  = sig_seqs #np.take(sig_seqs, rand_sig_indexes)#Take all sig samples
    sig_features  = featurize_sarx(sig_samples)
    non_sig_features = featurize_sarx(non_sig_seqs)
    sig_y = np.ones(len(sig_features))
    non_sigy = np.zeros(len(non_sig_features))
    
    train_xdata = np.concatenate((sig_features, non_sig_features))
    train_y_data = np.concatenate((sig_y, non_sigy))
    x_train, x_test, y_train, y_test = train_test_split(train_xdata, train_y_data, test_size=0.20, random_state=42)

    disc_model  = get_dense_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #  
    disc_model.fit(x_train, y_train,  epochs=DISC_EPOCHS, validation_data=(x_test, y_test),  callbacks= [callback], verbose=1) 
    disc_model.save(DISCRIMINATOR_LEARNING_MODEL_PATH)
    print("**-------Discriminator Network sucessfully trained and saved to path-----***",DISCRIMINATOR_LEARNING_MODEL_PATH)
    
def execute_discriminator_network_sarsx_03():
    '''
    This comprises a dataset using the filtered sequences of Sarsx 
    -Get GISAIDX dataset , filter sig and non sig sequences using EscapeNet
    -Use this dataset to improve escape network detection capability.
    '''
    non_sig_sarsx_path = "/home/perm/sars_escape_netv2/data/disc/train/sarx/non-significantx_with_score.csv"
    sig_sarsx_path= "/home/perm/sars_escape_netv2/data/disc/train/sarx/significantx_with_score.csv"
    SIG_THRESHOLD = 0.85
    NON_SIG_THRESHOLD = 0.45
    
    #sig original  
    sig_orig01_df=  pd.read_csv("/home/perm/cov/data/additional_escape_variants/gen/sig_combined_windows_train.csv")
    sig_orig02_df =  pd.read_csv("/home/perm/cov/data/additional_escape_variants/gen/sig_combined_windows_test.csv")
    sig_orig01 = sig_orig01_df['wild'].to_list() +  sig_orig01_df['mutated'].to_list()
    sig_orig02 = sig_orig02_df['wild'].to_list() +  sig_orig02_df['mutated'].to_list()
    sig_orig = sig_orig01 + sig_orig02
    
    #Sig data
    sig_df = pd.read_csv(sig_sarsx_path)
    sig_seqs = sig_df['sequence'][sig_df.score > SIG_THRESHOLD].to_list()
    
    # sig_seqs = sig_seqs + sig_orig
    sig_seqs = sig_orig
    sig_seqs = list(set(sig_seqs)) #removing sequences
    print('Final Significant Seqeunces for training after duplicate removed: ',len(sig_seqs) )
    
    sig_features = featurize_sarx(sig_seqs)
    sig_y = np.ones(len(sig_features))
    
    #Non Sig data 
    #We hypothesize that sig sequences having scores less than 0.5 are also nonsig sequences 
    nonsig_from_sig = sig_df['sequence'][sig_df.score < NON_SIG_THRESHOLD].to_list()
    #adding both non sig seqs obtained from sig ones and pure nonsig sequences
    nonsig_df = pd.read_csv(non_sig_sarsx_path)
    non_sig_seqs = nonsig_df['sequence'][nonsig_df.score < NON_SIG_THRESHOLD].to_list()
    #non_sig_seqs = np.concatenate( (non_sig_seqs, nonsig_from_sig) , axis=0) #Lets not conactenate non sif from sig

    #nonsig sequences from escapenet1
    nonsig_old = read_non_sig_seq_from_sars1()
    non_sig_seqs = non_sig_seqs + nonsig_old
    print(f'Nonsig Old: {len(nonsig_old)} Total: {len(non_sig_seqs)}')
    

    non_sig_features = featurize_sarx(non_sig_seqs)
    non_sigy = np.zeros(len(non_sig_features))
    
    print(f'Sig len: {len(sig_features)} , Non sig : {len(non_sig_features)} ')

    #training data preparation    
    train_xdata = np.concatenate((sig_features, non_sig_features))
    train_y_data = np.concatenate((sig_y, non_sigy))
    x_train, x_test, y_train, y_test = train_test_split(train_xdata, train_y_data, test_size=0.20, random_state=42, stratify=train_y_data)

    #Load and train model 
    disc_model  = get_dense_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #  
    disc_model.fit(x_train, y_train,  epochs=DISC_EPOCHS, validation_data=(x_test, y_test),  callbacks= [callback], verbose=1) 
    disc_model.save(DISCRIMINATOR_LEARNING_MODEL_PATH)
    print("**-------Discriminator Network sucessfully trained and saved to path-----***",DISCRIMINATOR_LEARNING_MODEL_PATH)

def execute_discriminator_network_sarsx_04():
    '''
    Executes the network with Filtered Sequences from original dataset  
    '''
    non_sig_sars_path = "/home/perm/cov/data/additional_escape_variants/scores/nonsig_org_scores.csv"
    sig_sars_path= "/home/perm/cov/data/additional_escape_variants/scores/sig_org_scores.csv"
    SIG_THRESHOLD = 0.70
    NON_SIG_THRESHOLD = 0.40
    
    #Sig data
    sig_df = pd.read_csv(sig_sars_path)
    sig_seqs = sig_df['sequence'][sig_df.score > SIG_THRESHOLD].to_list()
    sig_features = featurize_sarx(sig_seqs)
    sig_y = np.ones(len(sig_features))
    
    #Non Sig data 
    nonsig_df = pd.read_csv(non_sig_sars_path)
    non_sig_seqs = nonsig_df['sequence'][nonsig_df.score < NON_SIG_THRESHOLD].to_list()
    non_sig_features = featurize_sarx(non_sig_seqs)
    non_sigy = np.zeros(len(non_sig_features))
    
    print(f'Sig len: {len(sig_features)} , Non sig : {len(non_sig_features)} ')

    #training data preparation    
    train_xdata = np.concatenate((sig_features, non_sig_features))
    train_y_data = np.concatenate((sig_y, non_sigy))
    x_train, x_test, y_train, y_test = train_test_split(train_xdata, train_y_data, test_size=0.20, random_state=42, stratify=train_y_data)

    #Load and train model 
    disc_model  = get_dense_model()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #  
    disc_model.fit(x_train, y_train,  epochs=DISC_EPOCHS, validation_data=(x_test, y_test),  callbacks= [callback], verbose=1) 
    disc_model.save(DISCRIMINATOR_LEARNING_MODEL_PATH)
    print("**-------Discriminator Network sucessfully trained and saved to path-----***",DISCRIMINATOR_LEARNING_MODEL_PATH)

def execute_discriminator_network_sarsx_05():
    '''
    Implementing L2 regularization to the best data selected from model o4
    '''
    non_sig_sars_path = "/home/perm/cov/data/additional_escape_variants/scores/nonsig_org_scores.csv"
    sig_sars_path= "/home/perm/cov/data/additional_escape_variants/scores/sig_org_scores.csv"
    SIG_THRESHOLD = 0.70
    NON_SIG_THRESHOLD = 0.40
    
    #Sig data
    sig_df = pd.read_csv(sig_sars_path)
    sig_seqs = sig_df['sequence'][sig_df.score > SIG_THRESHOLD].to_list()
    sig_features = featurize_sarx(sig_seqs)
    sig_y = np.ones(len(sig_features))
    
    #Non Sig data 
    nonsig_df = pd.read_csv(non_sig_sars_path)
    non_sig_seqs = nonsig_df['sequence'][nonsig_df.score < NON_SIG_THRESHOLD].to_list()
    non_sig_features = featurize_sarx(non_sig_seqs)
    non_sigy = np.zeros(len(non_sig_features))
    
    print(f'Sig len: {len(sig_features)} , Non sig : {len(non_sig_features)} ')

    #training data preparation    
    train_xdata = np.concatenate((sig_features, non_sig_features))
    train_y_data = np.concatenate((sig_y, non_sigy))
    x_train, x_test, y_train, y_test = train_test_split(train_xdata, train_y_data, test_size=0.20, random_state=42, stratify=train_y_data)

    #Load and train model with regularization | check if it improves the performance
    disc_model  = get_dense_model_hybridized() #get_dense_model_regularized()
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=5) #  
    disc_model.fit(x_train, y_train,  epochs=DISC_EPOCHS, validation_data=(x_test, y_test),  callbacks= [callback], verbose=1) 
    disc_model.save(DISCRIMINATOR_LEARNING_MODEL_PATH)
    print("**-------Discriminator Network sucessfully trained and saved to path-----***",DISCRIMINATOR_LEARNING_MODEL_PATH)


def analyze_validation_dataset():
    model = m.load_model(DISCRIMINATOR_LEARNING_MODEL_PATH)
    model.summary()

    non_sig_data  = get_features("/home/perm/sars_escape_netv2/data/disc/train/non_sig_combined_windows_test.csv")
    sig_data =  get_features("/home/perm/sars_escape_netv2/data/disc/train/sig_combined_windows_test.csv")

    sig_len =  len(sig_data)
    sig_features_output = np.ones( (sig_len, 1) )
    non_sig_len = len(non_sig_data)
    nonsig_features_output = np.zeros( (non_sig_len, 1) )

    features = np.concatenate( (sig_data, non_sig_data), axis=0)
    targets = np.concatenate( (sig_features_output, nonsig_features_output), axis=0 )

    print("Evaluating  Validation dataset:")
    model.evaluate(features, targets)

    #Analyze  predictions | plot AUC_ROC / Confusion Matrix / Precision Recall
    y_preds = model.predict(features)

    plot_validation_predictions(targets, y_preds)
    save_results_to_csv(targets, y_preds, BASE_ANALYSIS_PATH+"/val_preds.csv")
    return targets, y_preds

def save_results_to_csv(target, predicted, save_path):
    #Flatten used for numpy becuse shape comes as 2d ie : shape (n, 1)
    df = pd.DataFrame({
        'target': target.flatten(),
        'predicted' :  predicted.flatten()
    })
    
    df.to_csv(save_path, mode= 'w', index=False)
    print(f"Prediction Results saves successfully to path:{save_path} ")

def plot_greany_test_predictions(targets, y_pred):
    model = m.load_model(DISCRIMINATOR_LEARNING_MODEL_PATH)
    DataAnalyzer.plot_AUC_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_AUC.png')
    DataAnalyzer.plot_Precision_recall_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_pr_curve.png')
    DataAnalyzer.plot_confusionMatrix(targets, y_pred, BASE_ANALYSIS_PATH+'/greany_confusion_matrix.png')

    print("Greany visual analysis plotted successfully !!!")

def plot_validation_predictions(targets, y_pred):
    model = m.load_model(DISCRIMINATOR_LEARNING_MODEL_PATH)
    DataAnalyzer.plot_AUC_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/validation_AUC.png')
    DataAnalyzer.plot_Precision_recall_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/validation_pr_curve.png')
    DataAnalyzer.plot_confusionMatrix(targets, y_pred, BASE_ANALYSIS_PATH+'/validation_confusion_matrix.png')

    print("Validation visual analysis plotted successfully !!!")

def plot_baum_predictions(targets, y_pred):
    model = m.load_model(DISCRIMINATOR_LEARNING_MODEL_PATH)
    DataAnalyzer.plot_AUC_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/baum_AUC.png')
    DataAnalyzer.plot_Precision_recall_curve(targets, y_pred, BASE_ANALYSIS_PATH+'/baum_pr_curve.png')
    DataAnalyzer.plot_confusionMatrix(targets, y_pred, BASE_ANALYSIS_PATH+'/baum_confusion_matrix.png')

    print("Validation visual analysis plotted successfully !!!")


def save_embed_learning_features():
    fl_path =  'data/gen/windowed_embed_train_seqs_35527.csv'
    df = pd.read_csv(fl_path)
    seqs = df['window_seqs'].to_list()

    model =load_fz_model()
    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape) #  (rows, 22, 512)
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    print("Reduced feature Shape after average: ", y_embed_output.shape)


    print("Shape of generated embed_learning_features: ", y_embed_output)
    # save to npy file
    np.save('data/analysis/embed_learning_feature/embed_learned_features.npy', y_embed_output)

def analyze_embed_learning_features_umap():
    features  = np.load('data/analysis/embed_learning_feature/embed_learned_features.npy')
    print("Shape of embed learned features: ", features.shape)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='SARS-COV-2 Escape Prediction')
    parser.add_argument('--predict', type=str, default='none',
                        help='Predict Greaney Sequences')
    parser.add_argument('--escape', type=str, default='' )
    parser.add_argument('--mutant', type=str )
    parser.add_argument('--plotAuc', type=str, default='')
    args = parser.parse_args()
    return args

def predict_new_sequence(sequence):
    print("Predicting if a given window is a escape sequence window")
    model = m.load_model(DISCRIMINATOR_LEARNING_MODEL_PATH)
    feature = get_single_feature(sequence)
    #Analyze  predictions | plot AUC_ROC / Confusion Matrix / Precision Recall
    y_preds = model.predict(feature)
    
    output = np.where(y_preds >0.5, 1 , 0)
    print("prediction output : ", output[0][0])
    if(output[0][0] == 1) :
        print (f"Sequence : {sequence} is escape !")
    else:
        print (f"Sequence : {sequence} is Non-escape !")
def get_single_window_based_feature(csv_path):
    model =load_fz_model()
    df = pd.read_csv(csv_path)
    column_data = df['sequence']
    seqs = column_data.to_numpy()
      
    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape) #  (rows, 22, 512)
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    print("Reduced feature Shape after average: ", y_embed_output.shape)
    return y_embed_output, seqs
    
      
def predict_new_sequences(csv_file_path, score_save_path):
    print("Predicting if a given window is a escape sequence window")
    model = m.load_model(DISCRIMINATOR_LEARNING_MODEL_PATH)
    features, seqs = get_single_window_based_feature(csv_file_path)
    y_preds = model.predict(features)
    
    print("Saving records to CSV")
    #y_preds shape : (n,1) so convert to 1D
    dataframe = pd.DataFrame({'sequence' : seqs, 'score' : y_preds.flatten() }) 
    # Save the DataFrame to a CSV file
    dataframe.to_csv(score_save_path, index=False)
    
    print("prediction output : ", y_preds.shape)
      


def is_single_res_mutant(mutant):
    try:
        original_residue = mutant[0]
        changed_to_residue= mutant[-1] 
        if len(original_residue) != 1 or len(changed_to_residue) != 1 :
            exit("The network supports only single residue substitution recognition (Eg: K512C)!! Please use other parameter --escape for escape window input. ")

        changed_position =   int(mutant[1:-1])
        if changed_position <1 or changed_position > 1273:
            exit("Please provide valid position value between (1 to 1273). ")
            return False
        return True
    except:
        exit("Please provide valid single residue mutant eg: (D512K) ! Network does not support multiple residue mutant !! ")
        return False

def feature_learning_seqs():
    seqs_train =  ASE.read_window_file('/home/perm/sars_escape_netv2/data/raw/feature_windows_v2.csv')
    # aug_seqs = ASE.read_window_file(GAN_AUG_SEQS)
    # total_seqs = seqs_train + aug_seqs
    random.shuffle(seqs_train)
    print("Total Seqs length", len(seqs_train))

    return seqs_train

def plot_greany_combined_auc_with_gan():
    aug_results = "data/analysis/escape_gan/greany_preds.csv"
    org_results = "model/pretrain_models/discriminator-model/integrated_model/analysisgreany_predictions.csv"
    save_path = "data/analysis/escape_gan"
    '''
    This method plots two Auc plots : one with original greaney auc, 
    and another after gan augumented data 
    '''
    aug_df = pd.read_csv(aug_results , header=0) #0th row contains column names
    # print(aug_df.head())
    aug_target = aug_df['target'].to_list()
    aug_pred = aug_df['predicted'].to_list()
    name_aug = "GAN-Augmented"
    
    org_df = pd.read_csv(org_results , header=0) #0th row contains column names
    org_target = org_df['target'].to_list()
    org_pred = org_df['predicted'].to_list()
    name_orig = "Original"
    DataAnalyzer.plot_auc_combined_plot(org_target, org_pred, name_orig, 
                                        aug_target, aug_pred, name_aug, save_path+'/auc_aug_vs_original_score.png')
    print("Auc saved successfully to path !!"+save_path)

def save_embedded_features(seq_path, save_features_path):
    '''
    input file (seq_path):  CSV file  
    This method extracts features using cov escape network and save features in npz format.
    '''
    embed_feautres = get_single_window_feature(seq_path)    
    np.savez(save_features_path, features=embed_feautres)
    print(f"Embed features saved successfully at: {save_features_path}")
    
    
    

if __name__ == "__main__":
    # save_feature_path  = "/home/perm/ProteinGAN/data/embed_features"
    # input_feature_path_gen_12000 = "/home/perm/cov/model/escape_gan/archives/m2/fs_12000.csv" 
    # input_feature_path_gen_8000 = "/home/perm/cov/model/escape_gan/archives/m2/fs_8000.csv"
    # input_feature_path_nat = "/home/perm/cov/model/escape_gan/archives/m2/natural.csv"
    
    # input_feature_path_full_8000 = "/home/perm/cov/model/escape_gan/archives/m2/8000.csv"
    # input_feature_path_full_12000 = "/home/perm/cov/model/escape_gan/archives/m2/12000.csv"
    
    

    # # save_embedded_features(input_feature_path_gen_12000, save_feature_path+"/gen_12000.npz")
    # save_embedded_features(input_feature_path_full_8000, save_feature_path+"/gen_full_8000.npz")
    # save_embedded_features(input_feature_path_full_12000, save_feature_path+"/gen_full_12000.npz")
    # # save_embedded_features(input_feature_path_nat, save_feature_path+"/nat.npz")
    
    # embed_seqs = feature_learning_seqs()
    # train_embedding_model(embed_seqs, epochs=8)
    # execute_discriminator_network()
    
    #---SARSX Execution ...d
    # execute_discriminator_network_sarsx()
    # execute_discriminator_network_sarsx_03()
    # execute_discriminator_network_sarsx_04()
    execute_discriminator_network_sarsx_05()
    plot_integrated_auc()
    #analyze_greany()
    # evaluate_baum()
    
    # read_non_sig_seq_from_sars() 
    # read_non_sig_seq_from_sars1()
    
   
    
    
    #DataAnalyzer.plot_confusionMatrix_using_csv(BASE_ANALYSIS_PATH+"/greany_preds.csv", 
    #                                            BASE_ANALYSIS_PATH+"/greaney_confusion.png")
    
    #save_results_to_csv(np.array([1, 0,1]), np.array([0,0,1]), BASE_ANALYSIS_PATH+"/escape_results.csv")
    # evaluate_baum()
    # analyze_validation_dataset()
    # 
    # plot_greany_combined_auc_with_gan()
    #exit()
    '''args = parse_args()
    if args.predict.strip() == 'greaney':
        analyze_greany()
    elif args.predict.strip() == 'validation':
        analyze_validation_dataset()
    elif args.predict.strip() == 'baum':
        evaluate_baum()
    elif args.escape:
        window = args.escape.strip()
        if len(window) != 20:
            exit("Please provide escape window of length 20 ")
        predict_new_sequence(window)
    elif args.mutant:
        is_single_res_mut = is_single_res_mutant(args.mutant)
        if is_single_res_mut:
            from escape import read_wildSequence
            changed_position =   int(args.mutant.strip()[1:-1]) 
            wild_seq = read_wildSequence()
            window = wild_seq[changed_position-10 : changed_position+10]
            print(f' len {len(window)}, {window}')
            predict_new_sequence(window)
    else:
        print("Please provide valid options: greaney | validation")

    pass'''



    



