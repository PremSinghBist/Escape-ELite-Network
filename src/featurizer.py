import os
import re
import numpy as np
import SignficantEscapeGenerator as SEG
import AdditionalSignficantEscapeSeqGenerator
from tensorflow import keras
import pandas as pd
import CsvGenerator as csvGen
from random import shuffle
import pandas as pd

ENCODED_DATA_SAVE_PATH = "model/encoded_dataset"
TRAIN_ENCODED_SEQ_PATH = ENCODED_DATA_SAVE_PATH + "/train_encoded_seqs.npz"
VAL_ENCODED_SEQ_PATH = ENCODED_DATA_SAVE_PATH + "/val_encoded_seqs.npz"
TEST_ENCODED_SEQ_PATH = ENCODED_DATA_SAVE_PATH + "/test_encoded_seqs.npz"
MAX_SEQ_LENGTH= 1280

def getVocabDictionary():
    AAs = [
    'A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H',
    'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W',
    'Y', 'V', 'X', 'Z', 'J', 'U', 'B',
    ]
    vocab = {
                amino_acid: index+1  for index, amino_acid in enumerate(sorted(AAs))
            }
            
    return vocab
'''
Returns integer encoded array and  sequence length for each sequence 
Sequnce lenght array will be required to extract sequence 
'''
def sequence_int_encoding(seqs):
    featurized_seqs = []
    vocab = getVocabDictionary()
    sorted_seq = sorted(seqs)
    start_boundary_index = len(vocab)+1
    end_boundary_index = len(vocab)+2 
    numpy_encoded_seq_arr_with_boundary_list = []
    #sequences_lengths = []
    for seq in sorted_seq:
        encoded_seq_arr = [vocab.get(amino_acid) for amino_acid in seq]
        #encoded_seq_arr_with_boundary = [start_boundary_index] + encoded_seq_arr + [end_boundary_index]
        #numpy_encoded_seq_arr_with_boundary = np.array(encoded_seq_arr_with_boundary)
        #numpy_encoded_seq_arr_with_boundary_list.append(numpy_encoded_seq_arr_with_boundary)
        #sequences_lengths.append(len(seq) + 2)
        featurized_seqs.append(encoded_seq_arr)
    return featurized_seqs
    #X = np.concatenate(numpy_encoded_seq_arr_with_boundary_list).reshape(-1,1)  #[ [26], [1], [2], [27], [26], [3], [27]]


def getMaxLength(seqences):
    seqLengths = [len(seq) for seq in seqences]
    return max(seqLengths)

def extractSequences(X, sequence_length_arr):
    current_index = 0
    sequences = []
    for seqLen in sequence_length_arr:
        sequences.append(X[current_index : current_index+seqLen])
        current_index = current_index + seqLen
    

    return sequences    
'''
def getSignificantSequences(process_sequences, extractSequences):
    signficantSequences = SEG.read_sequences(SEG.getSignificantFilePath())
    X_raw_pos, seq_len_arr_pos = process_sequences(signficantSequences)
    X_sig = extractSequences(X_raw_pos, seq_len_arr_pos)
    X_sig_length = len(X_sig)
    X_sig_numpy = np.array(X_sig).reshape(X_sig_length, -1)
    Y_sig_numpy = np.ones( (X_sig_length, 1) )  # (no_of_rows, 1 col)
    return X_sig_numpy,Y_sig_numpy

def getNotSignificantSequences(process_sequences, extractSequences):
    not_significantSequences = SEG.read_sequences(SEG.getNotSignificantFilePath())
    X_raw_neg, seq_len_arr_neg = process_sequences(not_significantSequences)
    X_not_sig = extractSequences(X_raw_neg, seq_len_arr_neg)
    X_not_sig_length = len(X_not_sig)
    X_not_sig_numpy = np.array(X_not_sig).reshape(X_not_sig_length, -1)
    Y_not_sig_numpy = np.zeros( (X_not_sig_length, 1) )  
    #print(X_sig_numpy.shape, X_not_sig_numpy.shape)
    return X_not_sig_numpy,Y_not_sig_numpy

def getTrainingData(process_sequences, extractSequences, getSignificantSequences, getNotSignificantSequences):
    X_sig_numpy, Y_sig_numpy = getSignificantSequences(process_sequences, extractSequences)
    X_not_sig_numpy, Y_not_sig_numpy = getNotSignificantSequences(process_sequences, extractSequences)
    X_train = np.concatenate( (X_sig_numpy, X_not_sig_numpy) )
    Y_train = np.concatenate( (Y_sig_numpy, Y_not_sig_numpy))
    return X_train,Y_train
'''
def get_identity_matrix(matrix_len):
    return [1  for i in range(matrix_len) ]

def get_zero_matrix(matrix_len):
    return [0  for i in range(matrix_len) ]
'''
This method alternatively keeps significant sequence and non significant sequence one after another 
returns: Training Sequences along with corresponding outputs 
'''
def merge_sequences(encoded_sig_seqs, sig_outputs, encoded_non_sig_seqs, non_sig_outputs):
    concatenated_arr_input = []
    concatenated_output = []
    max_length = max(len(encoded_sig_seqs), len(encoded_non_sig_seqs))
    for i in range(max_length):
        if i < len(encoded_sig_seqs):
            concatenated_arr_input.append(encoded_sig_seqs[i])
            concatenated_output.append(sig_outputs[i])
        if i < len(encoded_non_sig_seqs):
            concatenated_arr_input.append(encoded_non_sig_seqs[i])
            concatenated_output.append(non_sig_outputs[i])
    return concatenated_arr_input,concatenated_output


'''
Removes * from the sequences and returns the list
'''
def cleanse_non_significant_seqs(seqs):
    cleansed = []
    for seq in seqs:
        cleansed.append(seq.strip('*'))
    return cleansed

def generate_gsaid_significant_samples():
    non_significant_seqs = SEG.read_sequences(SEG.getGsaid_non_significant_seq_path())
    sample_length = len(non_significant_seqs)
    sample_seqs = get_significant_seqs_samples(sample_length)
    AdditionalSignficantEscapeSeqGenerator.generate_gsaid_significant_samples(sample_seqs)


'''
Retrieve significant sequence samples upto size of sample length
It removes greany and baum significant samples if present
'''
def get_significant_seqs_samples(sample_length = 17000):
    from escape import BAUM_CSV_FILE_NAME
    from escape import GREANY_GISAID_CSV_FILE_NAME
    
    significant_seqs = get_gsaid_significant_sequences()
    greany_seqs = getGreany_gisaid_significant_sequences(GREANY_GISAID_CSV_FILE_NAME)
    baum_seqs  = get_baum_sequences(BAUM_CSV_FILE_NAME)

    #Remove baum and greany sequences from training data 
    significant_seqs = subtract_sequences(significant_seqs, greany_seqs)
    significant_seqs = subtract_sequences(significant_seqs, baum_seqs)
    
    dataFrame  = pd.DataFrame(significant_seqs)
    fraction = float(sample_length / len(significant_seqs) )
    significant_seqs_samples = dataFrame.sample(frac=fraction)
    significant_seqs = significant_seqs_samples[0].to_list()

    print(f"********Significant Seq len: {len(significant_seqs)} *****")
    shuffle(significant_seqs)
    return significant_seqs


def get_training_data(sig_path, nonSig_path):
    non_significant_seqs = SEG.read_sequences(nonSig_path)
    print(f"****Non Significant Seqs length : {len(non_significant_seqs)}")

    significant_seqs = SEG.read_sequences(sig_path)

    encoded_sig_seqs = sequence_int_encoding(significant_seqs)
    sig_outputs = get_identity_matrix(len(encoded_sig_seqs))
    
    encoded_non_sig_seqs = sequence_int_encoding(non_significant_seqs)
    non_sig_outputs = get_zero_matrix(len(encoded_non_sig_seqs))

    concatenated_arr_input, concatenated_output = merge_sequences(encoded_sig_seqs, sig_outputs, encoded_non_sig_seqs, non_sig_outputs)
    #Shuffle Data frames using pandas 
    import pandas as pd
    df = pd.DataFrame(
        {
            "INPUT": concatenated_arr_input,
            "OUTPUT" : concatenated_output,
        }
    )
    df_shuffled = df.sample(frac=1).reset_index(drop=True)

    X_input = np.array(df_shuffled['INPUT'])
    Y = np.array(df_shuffled['OUTPUT']) 
    X = keras.preprocessing.sequence.pad_sequences(X_input, maxlen=MAX_SEQ_LENGTH)
    
    
    X = keras.preprocessing.sequence.pad_sequences(concatenated_arr_input, maxlen=MAX_SEQ_LENGTH)
    Y = np.array(concatenated_output)
    return X,Y
def get_greany_gisaid_dataset():
    from escape import GREANY_GISAID_CSV_FILE_NAME
    greany_sig_seqs = getGreany_gisaid_significant_sequences(GREANY_GISAID_CSV_FILE_NAME)
    encoded_greny_seqs = sequence_int_encoding(greany_sig_seqs)
    sig_outputs_greany = get_identity_matrix(len(encoded_greny_seqs))

    X_input = np.array(encoded_greny_seqs, dtype=object)
    Y = np.array(sig_outputs_greany)

    X = keras.preprocessing.sequence.pad_sequences(X_input, maxlen=MAX_SEQ_LENGTH)
    return X,Y
def get_greany_science_dataset(GREANY_SCIENCE_CSV_FILE_NAME):
    
    greany_sig_seqs = getGreany_science_significant_sequences(GREANY_SCIENCE_CSV_FILE_NAME)
    encoded_greny_seqs = sequence_int_encoding(greany_sig_seqs)
    sig_outputs_greany = get_identity_matrix(len(encoded_greny_seqs))

    X_input = np.array(encoded_greny_seqs, dtype=object)
    Y = np.array(sig_outputs_greany)

    X = keras.preprocessing.sequence.pad_sequences(X_input, maxlen=MAX_SEQ_LENGTH)
    return X,Y

def get_baum_test_dataset(csv_file_name):
    seqs = get_baum_sequences(csv_file_name)
    encoded_data = sequence_int_encoding(seqs)
    sig_outputs = get_identity_matrix(len(seqs))

    X_input = np.array(encoded_data,  dtype=object)
    Y = np.array(sig_outputs)
    X = keras.preprocessing.sequence.pad_sequences(X_input, maxlen=MAX_SEQ_LENGTH)
    return X, Y 

def get_baum_sequences(csv_file_name):
    from escape import BAUM_SEQ_PATH
    baum_data_file = BAUM_SEQ_PATH + os.path.sep +csv_file_name
    df = csvGen.read_csv(baum_data_file)
    seqs = df.iloc[:, 0].to_list() 
    print("Lenght of baum seqs:", len(seqs))

    #replace special char 
    seqs = [seq.replace("*", "") for seq in seqs ]
    return seqs

def get_greany_gisaid_sequences(csv_file_name):
    from escape import GREANY_SEQ_PATH
    greany_data_file = GREANY_SEQ_PATH + os.path.sep +csv_file_name
    df = csvGen.read_csv(greany_data_file)
    seqs = df.iloc[:, 1].to_list() 
    print("Lenght of Greany seqs:", len(seqs))

    #replace special char 
    return seqs

def get_greany_science_sequences(csv_file_name):
    from escape import GREANY_SEQ_PATH
    greany_data_file = GREANY_SEQ_PATH + os.path.sep +csv_file_name
    df = csvGen.read_csv(greany_data_file)
    seqs = df.iloc[:, 1].to_list() 
    print("Lenght of Greany seqs:", len(seqs))

    #replace special char 
    seqs = [seq.replace("*", "") for seq in seqs ]
    return seqs


def save_dataset(train_dataset, val_dataset, test_dataset):
    train_x, train_y = train_dataset
    val_x, val_y = val_dataset
    test_x, test_y = test_dataset
    np.savez(TRAIN_ENCODED_SEQ_PATH, train_x = train_x,  train_y =  train_y )
    np.savez(VAL_ENCODED_SEQ_PATH, val_x = val_x,  val_y = val_y )
    np.savez(TEST_ENCODED_SEQ_PATH, test_x = test_x,  test_y = test_y )
    print("Training and Validation Test Encoded Dataset saved successfully !")

def get_train_encoded_dataset():
    data = np.load(TRAIN_ENCODED_SEQ_PATH)
    train_x = data['train_x']
    train_y = data['train_y']
    return (train_x, train_y)

def get_validation_encoded_dataset():
    data = np.load(VAL_ENCODED_SEQ_PATH)
    val_x = data['val_x']
    val_y = data['val_y']
    return (val_x, val_y)

def get_test_encoded_dataset():
    data = np.load(TEST_ENCODED_SEQ_PATH)
    test_x = data['test_x']
    test_y = data['test_y']
    return (test_x, test_y)


def get_gsaid_significant_sequences():
    import random
    significant_seqs_gsaid= SEG.read_sequences(SEG.getGsaid_significant_seq_path())
    random.shuffle(significant_seqs_gsaid)
    return list (set(significant_seqs_gsaid))


def get_significant_escapePaper_2000_sequences():
    significant_seqs_orig = SEG.read_sequences(SEG.SINGLE_RES_SIGNIFICANT_SEQS_PATH)
    return significant_seqs_orig


'''
These are the significant sequences extracted from greany dataset 
from original paper - Natural language predicts viral escape
'''
def getGreany_gisaid_significant_sequences(csv_file_name):
    significant_seqs_greany = get_greany_gisaid_sequences(csv_file_name)
    print("Total Sequences (greany) : ", len(significant_seqs_greany))
    return significant_seqs_greany

def getGreany_science_significant_sequences(csv_file_name):
    significant_seqs_greany = get_greany_gisaid_sequences(csv_file_name)
    print("Total Sequences (greany science) : ", len(significant_seqs_greany))
    return significant_seqs_greany

def generate_sig_training_sequences(seq_name):
    final_training_sig_seq = get_gsaid_significant_sequences()
    SEG.generate_fasta_using_list(final_training_sig_seq, seq_name )

'''
This methods takes two list as input
and returns their difference A - B 
'''
def subtract_sequences(list1, list2):
    print("Intiallly: Len of List1: {} and List2 : {}".format(len(list1), len(list2) ))
    differenceList = set(list1) - set(list2)
    print("After difference, the difference list is    : ", len(differenceList))
    return list(differenceList)
    

if __name__ == '__main__':
    #get_greany_sequences('greany_gsaid.csv')
    #generate_gsaid_significant_samples()
    #print(getVocabDictionary())
    pass









  












