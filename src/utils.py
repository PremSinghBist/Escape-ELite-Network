from anndata import AnnData
from collections import Counter
import datetime
from dateutil.parser import parse as dparse
import errno
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import scanpy as sc
from scipy.sparse import csr_matrix, dok_matrix
import scipy.stats as ss
import seaborn as sns
import sys
import time
import warnings

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord


np.random.seed(1)
random.seed(1)

def tprint(string):
    string = str(string)
    sys.stdout.write(str(datetime.datetime.now()) + ' | ')
    sys.stdout.write(string + '\n')
    sys.stdout.flush()

def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise
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

'''
This method calcualte the difference between first list and second list 
returns items present in list A but not present in B
'''
def diff_operation(list_A, list_B):
    print(f"Length First set : {len(list_A)} Second: {len(list_B)}")
    dif = list (set(list_A) - set(list_B))
    print(f"Len Diff set: {len(dif)}")
    return dif

def save_csv_to_fasta(csv_file_path, save_fasta_file_path) :
    df = pd.read_csv(csv_file_path)
    seqs  =  list(df['window'])
    s_positions = list(df['start_pos'])
    end_positions = list(df['end_pos'])
    
    save_list_to_fasta(seqs, s_positions, end_positions, save_fasta_file_path)
    
def save_list_to_fasta(seqs, s_positions, end_positions, save_fasta_file_path):
    seqs_records = []
    id = 0
    for seq in seqs:
        my_desc= f"{s_positions[id]}, {end_positions[id]}"
        bio_seq = SeqIO.SeqRecord(Seq(seq), id = str(id), description=my_desc)
        seqs_records.append(bio_seq) 
        id += 1
        
    directory = os.path.dirname(save_fasta_file_path)   
    if not os.path.exists(directory):
        os.makedirs(directory) 
    
    with open(save_fasta_file_path, 'w' ) as output_handle:
        SeqIO.write(seqs_records, output_handle, "fasta")
    
    print(f"Fasta file saved successfully to {save_fasta_file_path}")
    
if __name__== "__main__":
    csv_file_path = "/home/perm/cov/data/gen/sig_windows_with_pos.csv"
    save_fasta_file_path = "/home/perm/cov/data/gen/sig_windows_with_pos.fasta"
    
    save_csv_to_fasta(csv_file_path, save_fasta_file_path)