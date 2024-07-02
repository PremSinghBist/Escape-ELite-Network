import os
from Bio.Seq import Seq
from Bio import SeqIO
from Bio.SeqRecord import SeqRecord
DIR_SEP = os.path.sep
base_path = "/home/perm/sars_escape_netv2/data/gen"
mini_dms_file = base_path+DIR_SEP+"binding_Kds.csv"
dms_file  = base_path+ DIR_SEP+"binding_Kds_Original.csv"
wild_seq_path = base_path+DIR_SEP+"cov2_spike_wt.fasta"
significant_file_original = "significant_seq_orig.fa"
not_significant_file = "duplicate_removed_not_significant_seq.fa"
mixed_seqs_file = "/gsAid/gsaid_10000_seqs.fa"

'''This is an additionaly duplicate removed file from not_significant_file where
 the non significant dataset constructed from original science paper , 
 which was previously non significant was turned into significant after mutation
 '''
SIGNIFICANT_SEQS_COMBINED = "significant_seqs_combined.fa"

SINGLE_RES_SIGNIFICANT_SEQS_PATH = "/home/perm/cov/data/additional_escape_variants/gen/SINGLE_RES_SUB.faa"




#print("Ref seq:",ref_sequence)
'''
This will compute the mutation index where the change in residue happened w.r.t to wild sequece type
Eg: mutant: Y91L 
-Get residue postion : 91  
-Actual index = 91 - 1 
-Added base number 330

'''
def compute_mutation_position(mutant):
    return int(mutant[1:-1]) - 1 + 330

'''
Header Meta Info : 
>11|6.0|345-452-508| AAAAAAAAGTTTGAAG | 1   example: [SEQ_NUMBER | LOG10KA | MUTATATION_POSITIONS | barcode | significant]
'''
def generate_fasta(sequenceDictionary):
    seq_num=0
    sequence_records = []
    for sequence in sequenceDictionary:
        sequence_meta = sequenceDictionary[sequence]
        seq_id = ''
        desc = ''
        if sequence_meta['significant'] :
            positions = [ str(index) for index in sequence_meta['mutated_position_indexes'] ]
            postions_str = '-'.join(positions)
            seq_id = str(seq_num) + "|"+str(sequence_meta['log10Ka']) + "|"+postions_str+"|" +"1"+"|"
            desc=sequence_meta['barcode']
        else:
            seq_id = str(seq_num) + "|"+"0"

        record = SeqRecord(
            Seq(str(sequence),), 
            id=seq_id,
            description=desc
            )
        sequence_records.append(record)
        seq_num = seq_num+1
    SeqIO.write(sequence_records, base_path+DIR_SEP+"dms_significant_escapes.faa", "fasta")    

def generate_fasta_using_list(sequenceList, file_name='significant_seq.fa'):
    seq_num=0
    sequence_records = []
    for sequence in sequenceList:
        seq_id = ''
        seq_id = str(seq_num) 
        record = SeqRecord(
            Seq(str(sequence),), 
            id=seq_id,
            description="|-"
            )
        sequence_records.append(record)
        seq_num = seq_num+1
    SeqIO.write(sequence_records, base_path+DIR_SEP+file_name, "fasta") 
    print("Sequences generated successfully !")
        

def parse_deep_mutational_scan_csv(mini_dms_file, ref_sequence):
    mutated_sequences_dictionary = {}
    with open(mini_dms_file) as f: 
        f.readline()
        for line in f:
        #Remove double quotes found a line , trailing white space and split based on comma separator
            fields = line.replace('"', '').rstrip().split(',')

        #if log10KA value is NA - not available skip this
            if fields[5] == 'NA':
                continue
        
        #BindingAffitnity value of escape mutation ->conver this to float as data is avaiable in string 
            log10KA = float(fields[5])
        #print(log10KA)

        #Second last file contains information about subsitution mutation ex: Y91L K199Y 
            mutated_residues_list = fields[-2].split()   #[Y91L, K199Y]
            wild_seq_residues = [residue for residue in ref_sequence] #Individual amino acids of wild sequence 
        
        #Keeps track of all mutated positions for a particular sequence
            mutated_position_indexes = []
            for mutant in mutated_residues_list:
            #print(mutant)  #A22C
                orginal_residue = mutant[0]
                mutated_to_residue = mutant[-1]
                mutation_position_index = compute_mutation_position(mutant)  

                #Substitute the mutated residue in mutated position in wild seq      
                wild_seq_residues[mutation_position_index] = mutated_to_residue
                mutated_position_indexes.append(mutation_position_index)
            #print(type(mutation_position))
            
            #print("Original: {}, mutated to {}, mutated_postion {}".format(orginal_residue, mutated_to_residue, mutation_position))
        
        #Construct mutated sequence joining back the wild_seq_residues
            mutated_sequence = ''.join(wild_seq_residues)
            mutated_sequence_meta = {}
            mutated_sequence_meta['mutated_position_indexes'] = mutated_position_indexes
        #Fitness/preference 
            mutated_sequence_meta['log10Ka'] = log10KA

            #barcode
            mutated_sequence_meta['barcode']= fields[2]

            mutated_sequence_meta['significant'] = True

            if(mutated_sequence not in mutated_sequences_dictionary):
                mutated_sequences_dictionary[mutated_sequence] = mutated_sequence_meta


            #print(mutated_sequences_dictionary)
    return mutated_sequences_dictionary
'''
Generates single residue mutated sequences using wildtype sequence 
returns list of single residue mutated sequences 
'''
def gen_mutated_seqs(ref_sequence):
    mutatedSeqs_dic = {}
    residues = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I', 'L', 'K', 'M', 
    'F', 'P', 'S', 'T', 'W', 'Y', 'V']

    for i in range(len(ref_sequence)):
        for aminoAcid in residues:
            if ref_sequence[i] == aminoAcid:
                continue
          
            mutated_seq = ref_sequence[:i] + aminoAcid + ref_sequence[i+1:]
            mutatedPosition = i + 1 #Postion is counted from 1 in mutation set so increased count by 1 
            mutatnt = ref_sequence[i] + str(mutatedPosition) +aminoAcid

            mutatedSeqs_dic[mutated_seq] = mutatnt #Eg: {ACKBEED: P3K}

    return mutatedSeqs_dic

def read_sequences(filePath):
    if not os.path.exists(filePath):
        raise Exception('Invalid File Path:',filePath)
    sequence_list = []
    with open(filePath, 'r', encoding='latin-1') as handle:    
        records = SeqIO.parse(handle, "fasta")
        for record in records:
            sequence_list.append(str(record.seq))
    return sequence_list
def read_sequences_with_limit(filePath, MAX_SEQ=10):
    if not os.path.exists(filePath):
        raise Exception('Invalid File Path:',filePath)
    sequence_list = []    
    records = SeqIO.parse(filePath, "fasta")
    for record in records:
        if len(sequence_list)>MAX_SEQ-1:
            return sequence_list
        sequence_list.append(str(record.seq))
        
    return sequence_list
def getSignificantFilePath():
    return base_path+DIR_SEP+SIGNIFICANT_SEQS_COMBINED
def getNotSignificantFilePath():
    return base_path+DIR_SEP+not_significant_file
'''
This returns the file path of sequences that contains both significant and not significant sequences
'''
def getMixedSequencesFilePath():
    return base_path+DIR_SEP+mixed_seqs_file
def getGsaid_non_significant_seq_path():
    return base_path+DIR_SEP+ "GISAID_NON_SIG.fa"
def getGsaid_significant_seq_path():
    return base_path+DIR_SEP+ "GISAID_SIG.fa"
def getGsaid_significantSeqs_sample_path():
    return base_path+DIR_SEP+ "GISAID_SIG_SAMPLES.fa"


def getGreany_significant_seq_path():
    return base_path+DIR_SEP+ "significant_seq_original_paper.fa"

def getGsaid_raw_seq_path():
    return "/home/perm/cov/data/gsAid-original/spikeprot0327.fasta"

def extract_window(segment_length=20, pos=0, input_seq = ''):
    if len(input_seq) == 0:
        return '' 
    
    seq_array = [amino_acid for amino_acid in input_seq]
    half_segment_len = int(segment_length/2)
    if pos < half_segment_len:
        split_window = ''.join(seq_array[0 : segment_length])
        
    elif pos+half_segment_len > len(seq_array):
        split_window = ''.join(seq_array[len(seq_array)-segment_length : ])
    else: 
        split_window = ''.join(seq_array[pos-half_segment_len : pos+half_segment_len])

    return split_window

def extract_window_with_pos(segment_length=20, pos=0, input_seq = ''):
    start_pos = -1
    end_pos = -1
    if len(input_seq) == 0:
        return '' 
    
    seq_array = [amino_acid for amino_acid in input_seq]
    half_segment_len = int(segment_length/2)
    if pos < half_segment_len:
        split_window = ''.join(seq_array[0 : segment_length])
        start_pos = 0
        end_pos = segment_length
        
    elif pos+half_segment_len > len(seq_array):
        split_window = ''.join(seq_array[len(seq_array)-segment_length : ])
        start_pos = len(seq_array)-segment_length
        end_pos = len(seq_array)
    else: 
        split_window = ''.join(seq_array[pos-half_segment_len : pos+half_segment_len])
        start_pos = pos-half_segment_len
        end_pos = pos+half_segment_len-1

    return split_window, start_pos, end_pos
    
def get_random_positions(max_len=1280, total_pos=100):
    max_len = 1280
    samples = [i for i in range(0, max_len) ] 
    from random import choices 
    choosen_positions = choices(samples, k = total_pos)
    return choosen_positions





if __name__ == '__main__':
    #Reading reference/base sequence (wild type sequence)
    '''ref_sequence = SeqIO.read(wild_seq_path, 'fasta').seq
    #dms_file | mini_dms_file
    mutated_sequences_dictionary = parse_deep_mutational_scan_csv(dms_file, ref_sequence)
    randomly_mutated_seq_dict = gen_mutated_seqs(ref_sequence)

    original_len_mut  = len(mutated_sequences_dictionary)
    print("Count of mutated seqs: {}, random seqs: {}".format(original_len_mut, len(randomly_mutated_seq_dict) ))
    for seq in randomly_mutated_seq_dict:
        #Randomly mutated sequences which are not available in mutated_seq_dict, we treat them as insignificant mutations.
        if seq not in mutated_sequences_dictionary:
            mutated_sequence_meta = {'significant' : False }
            mutated_sequences_dictionary[seq] = mutated_sequence_meta

    found_signficant_seq_in_randomSeq = original_len_mut + len(randomly_mutated_seq_dict) - len(mutated_sequences_dictionary) 
    print("Total Sequences {}, Significant sequences found in randomly generated sequences {} ".format(len(mutated_sequences_dictionary) ,found_signficant_seq_in_randomSeq))
    generate_fasta(mutated_sequences_dictionary) '''
    #generate_fasta_using_list(['AAC', 'ACBD', 'DPP', 'CQRT'])
    #significantPath = base_path+DIR_SEP+significant_file_original
    #read_sequences(significantPath)

    #window = extract_window(segment_length=10, pos=6, seq = '123456789abdefgh')
    #print(window)
    #print(get_random_positions(1280, 5))
    pass




        






