import os
import csv
from escape import load_baum2020
from escape import load_greaney2020
generator_base_path = "/home/perm/cov/data/gen"

def extract_baum_meta(seq_escapes, significant_seqs, not_significant):
    for seq in seq_escapes:
        seq_escape_meta_list = seq_escapes.get(seq) #meta dictionary is stored in array 
        is_significant = sum([seq_escape_meta['significant'] for seq_escape_meta in seq_escape_meta_list]) >0
        if is_significant:
            significant_seqs.append(seq)
        else: 
            not_significant.append(seq)
    
    return significant_seqs, not_significant

def extract_greany_meta(significant_seqs, not_significant):
    greany_significant_mutants = []
    greany_not_significant_mutants = []
    
    seqs_escape_greaney  = load_greaney2020()
    for seq in seqs_escape_greaney:
        seq_escape_meta_list = seqs_escape_greaney.get(seq)
        is_significant = sum([seq_escape_meta['significant'] for seq_escape_meta in seq_escape_meta_list]) >0
        if is_significant:
            for seq_escape_meta in seq_escape_meta_list:
                greany_significant_mutants.append(seq_escape_meta['mutant'])  
            significant_seqs.append(seq)
        else:
            for seq_escape_meta in seq_escape_meta_list:
                greany_not_significant_mutants.append(seq_escape_meta['mutant'])
            not_significant.append(seq)
    #remove duplicates
    significantMutantsGreany = removeDuplicatesFromList(greany_significant_mutants)
    nonSignificantMutantsGreany = removeDuplicatesFromList(greany_not_significant_mutants)

    return significant_seqs, not_significant, significantMutantsGreany, nonSignificantMutantsGreany

def removeDuplicatesFromList(seqList):
    uniqueList = []
    for seq in seqList:
        if (seq not in uniqueList):
            uniqueList.append(seq)
    return uniqueList

def generate_sequences():
    wild_seq , seq_escapes = load_baum2020()
    significant_seqs = []
    not_significant = []
    significant_seqs, not_significant_seqs = extract_baum_meta(seq_escapes, significant_seqs, not_significant)
    print("Baum Significant Sequences : {}, Not Significant Sequences: {}".format(len(significant_seqs), len(not_significant)))

    significant_seqs, not_significant_seqs, significantMutantsGreany, nonSignificantMutantsGreany = extract_greany_meta(significant_seqs, not_significant)

    unique_sig_seq = removeDuplicatesFromList(significant_seqs)
    unique_not_sig_seq = removeDuplicatesFromList(not_significant_seqs)
        
    print("Total  Significant Seq : {}, Total Not Significant: {} \n  Unique sig: {}, Unique Not sig: {}".
    format(len(significant_seqs), len(not_significant), len(unique_sig_seq), len(unique_not_sig_seq)  ))

    from SignficantEscapeGenerator import generate_fasta_using_list
    #generate_fasta_using_list(unique_sig_seq, 'significant_seq.fa')
    generate_fasta_using_list(unique_not_sig_seq, 'not_significant_seq.fa')
    print("*** File Generated successfully")

def generate_greany_significant_mutant_dataset(mutantList):
        mutant_data_path =  generator_base_path + os.path.sep +"greany_significant_mutants.csv"
        print("*Greany Significant Mutant data Path for file generation**:",mutant_data_path)
        with open(mutant_data_path, 'w', encoding='UTF-8', newline="") as f:
            writer = csv.writer(f)
            #Constructed 2D list
            for mutant in mutantList:
                writer.writerow([mutant])
def generate_greany_not_significant_mutant_dataset(mutantList):
        mutant_data_path =  generator_base_path + os.path.sep +"greany_not_significant_mutants.csv"
        print("*Greany Not Significant Mutant data Path for file generation**:",mutant_data_path)
        with open(mutant_data_path, 'w', encoding='UTF-8', newline="") as f:
            writer = csv.writer(f)
            #Constructed 2D list
            for mutant in mutantList:
                writer.writerow([mutant])

if __name__ == '__main__':
    #generate_sequences(extract_baum_meta, extract_greany_meta, removeDuplicateSequences)
    significant_seqs = []
    not_significant = []
    significant_seqs, not_significant_seqs, significantMutantsGreany, nonSignificantMutantsGreany = extract_greany_meta(significant_seqs, not_significant)
    generate_greany_significant_mutant_dataset(significantMutantsGreany)
    generate_greany_not_significant_mutant_dataset(nonSignificantMutantsGreany)
