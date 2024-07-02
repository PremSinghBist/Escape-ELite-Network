
import pandas as pd
import numpy as np
csv_standford_mutant_file="/home/perm/sars_escape_netv2/data/test/standford_with_mutant.csv"
db_path = "/home/perm/sars_escape_netv2/data/raw/stanford_mutants.csv"
def create_window_seqs(window_size=20, input_csv_path="", output_window_path=""):
    df = pd.read_csv(input_csv_path)
    windows =  set()
    windows_list = []
    no_of_segments = 0
    for index, row in df.iterrows():
        seq = row['seq']
        base_index = 0
        seq_len = len(seq)
        while base_index <= seq_len:
            seq_segment = seq[base_index: base_index+window_size]
            base_index = base_index + window_size 
            if len(seq_segment) == window_size:
                windows.add(seq_segment)
                windows_list.append(seq_segment)
                no_of_segments = no_of_segments + 1
    print(f"No_of_segments : {len(windows)}")
    print(f"Lenght of sequences including duplicate: {len(windows_list)} ")
    df1 = pd.DataFrame({'window': list(windows)})
    df1.to_csv(output_window_path, index=False)
    print("Windowed sequences sucessfully saved to path: ",output_window_path)
    
def create_featured_windows_without_zero_char(input_csv_path, output_csv_path):
    ##Clean the windows segments having 000 such as CLDSFKEELDKY000000000AX
    cleansed_seqs = []
    df3 = pd.read_csv(input_csv_path)
    for index, record in  df3.iterrows():
        #Character to ignore: is 0
        # print(f"Index : {index} , record : {record['window_seqs']}")
        ignore_window_with_digit = "0"
        if record['window_seqs'].__contains__(ignore_window_with_digit):
            continue
        cleansed_seqs.append(record['window_seqs'])
    print("Total records in cleansed sequences: ", len(cleansed_seqs))
    df4 = pd.DataFrame({'window_seqs' : list(cleansed_seqs)})    
    df4.to_csv(output_csv_path, index=False)

def construct_featured_window():
    window_size = 20
    input_csv_path = "/home/perm/sars_escape_netv2/data/raw/cleansed_standard_len.csv"
    output_window_path = "/home/perm/sars_escape_netv2/data/raw/feature_windows.csv"
    #create_window_seqs(input_csv_path, output_window_path)
    create_featured_windows_without_zero_char("/home/perm/sars_escape_netv2/data/raw/feature_windows_v1.csv",
                                            "/home/perm/sars_escape_netv2/data/raw/feature_windows_v2.csv") 
# construct_featured_window()

def construct_save_stanford_windows():
    df  = pd.read_csv(db_path)
    mutants = df['Mutation'].to_list()
    # print(len(mutants))
    # print(mutants[0:5])
    wild_seq='''MFVFLVLLPLVSSQCVNLTTRTQLPPAYTNSFTRGVYYPDKVFRSSVLHSTQDLFLPFFSNVTWFHAIHV
    SGTNGTKRFDNPVLPFNDGVYFASTEKSNIIRGWIFGTTLDSKTQSLLIVNNATNVVIKVCEFQFCNDPF
    LGVYYHKNNKSWMESEFRVYSSANNCTFEYVSQPFLMDLEGKQGNFKNLREFVFKNIDGYFKIYSKHTPI
    NLVRDLPQGFSALEPLVDLPIGINITRFQTLLALHRSYLTPGDSSSGWTAGAAAYYVGYLQPRTFLLKYN
    ENGTITDAVDCALDPLSETKCTLKSFTVEKGIYQTSNFRVQPTESIVRFPNITNLCPFGEVFNATRFASV
    YAWNRKRISNCVADYSVLYNSASFSTFKCYGVSPTKLNDLCFTNVYADSFVIRGDEVRQIAPGQTGKIAD
    YNYKLPDDFTGCVIAWNSNNLDSKVGGNYNYLYRLFRKSNLKPFERDISTEIYQAGSTPCNGVEGFNCYF
    PLQSYGFQPTNGVGYQPYRVVVLSFELLHAPATVCGPKKSTNLVKNKCVNFNFNGLTGTGVLTESNKKFL
    PFQQFGRDIADTTDAVRDPQTLEILDITPCSFGGVSVITPGTNTSNQVAVLYQDVNCTEVPVAIHADQLT
    PTWRVYSTGSNVFQTRAGCLIGAEHVNNSYECDIPIGAGICASYQTQTNSPRRARSVASQSIIAYTMSLG
    AENSVAYSNNSIAIPTNFTISVTTEILPVSMTKTSVDCTMYICGDSTECSNLLLQYGSFCTQLNRALTGI
    AVEQDKNTQEVFAQVKQIYKTPPIKDFGGFNFSQILPDPSKPSKRSFIEDLLFNKVTLADAGFIKQYGDC
    LGDIAARDLICAQKFNGLTVLPPLLTDEMIAQYTSALLAGTITSGWTFGAGAALQIPFAMQMAYRFNGIG
    VTQNVLYENQKLIANQFNSAIGKIQDSLSSTASALGKLQDVVNQNAQALNTLVKQLSSNFGAISSVLNDI
    LSRLDKVEAEVQIDRLITGRLQSLQTYVTQQLIRAAEIRASANLAATKMSECVLGQSKRVDFCGKGYHLM
    SFPQSAPHGVVFLHVTYVPAQEKNFTTAPAICHDGKAHFPREGVFVSNGTHWFVTQRNFYEPQIITTDNT
    FVSGNCDVVIGIVNNTVYDPLQPELDSFKEELDKYFKNHTSPDVDLGDISGINASVVNIQKEIDRLNEVA
    KNLNESLIDLQELGKYEQYIKWPWYIWLGFIAGLIAIVMVTIMLCCMTSCCSCLKGCCSCGSCCKFDEDD
    SEPVLKGVKLHYT'''
    wild_seq = wild_seq.replace("\n", '')
    wild_seq = wild_seq.replace(" ", '')
    print(len(wild_seq))
    wild_seq_array = [amino_acid for amino_acid in wild_seq]
    # print(len(wild_seq_array))
    # wild_seq_array[0:5]
    import copy
    windows = []
    window_size = 20
    for mutant in mutants:
        position = int(mutant[1:-1])  - 1 #For acutal zero based computation
        # print("Actual Postion: ",position +1) #
        mutated = mutant[-1]
        original = mutant[0]
        # print(f" Original : {original} Mutated: ", mutated)
        
        assert original,  wild_seq_array[position]
        # print(f' Oringal {original} and from wild_seq_array: {wild_seq_array[position]}')
        
        #Copy the wildarray with deep copy
        mutated_seq = copy.deepcopy(wild_seq_array)
        mutated_seq[position] = mutated
        
        # print("Mutated Seq: ", "".join(mutated_seq))
        # print("New Seqence mutation residue: ", mutated_seq[position])
        # print("wild sequence residue: ", wild_seq_array[position])
        
        #Exact 20 lenght window from position 
        start_pos = position - 10
        end_pos = position + 10
        if start_pos < 0:
            start_pos = 0
            end_pos = window_size
        elif end_pos >1273:
            end_pos = 1273
            start_pos = 1273 - window_size
        
        window = "".join(mutated_seq[start_pos : end_pos]) #Extract sequenc array and convert to string
        assert len(window), window_size 
        # print('Window Lenght : ', len(window))
        windows.append(window)

    print(f'Total windows:', len(windows))
    df = pd.DataFrame(data={'window_seqs': windows, 'mutants' : mutants})

    df.to_csv(csv_standford_mutant_file , index=False)
    
def evaluate_standford_escapeMutants(disc_model_path):
    import EscapeNet2Executer as Esc 
    import tensorflow.keras.models as km
    features  = Esc.get_single_window_feature(csv_standford_mutant_file)
    model = km.load_model(disc_model_path) 
    model.summary()
    evaluation_metrics = model.evaluate(features, np.ones( (features.shape[0], 1)))
    # print(evaluation_metrics)
    accuracy = evaluation_metrics[1] #0: loss 1 : Accuray | 2: Auc 
    return accuracy 

    # result = model.predict(features)
    # print("Result shape: ", result.shape)
    # boolean_result = result > 0.5
    # #Accuracy 
    # correct_classification = np.sum(boolean_result)
    # total_classification = len(boolean_result)
    # print(f'Correct classfication: {correct_classification} Total Instance: {total_classification}')
    # accuracy = correct_classification/total_classification
    # print(f"Accuracy : {accuracy}")
    
if __name__== "__main__":
    # construct_save_stanford_windows()
    old_model_disc_path = "/home/perm/sars_escape_netv2/model/Sen_old_model/pretrain_models/discriminator-model/integrated_model"
    old_model_with_gen_disc_path = "/home/perm/cov/model/escape_gan/archives/m2/disc_learning"
    print(evaluate_standford_escapeMutants(old_model_with_gen_disc_path))
    
    '''
    accuracy_m20  = evaluate_standford_escapeMutants("/home/perm/sars_escape_netv2/data/model_results_archive/m20/sarsx_disc")
    accuracy_m19  = evaluate_standford_escapeMutants("/home/perm/sars_escape_netv2/data/model_results_archive/m19/sarsx_disc")
    accuracy_m15  = evaluate_standford_escapeMutants("/home/perm/sars_escape_netv2/data/model_results_archive/m15/sarsx_disc")
    accuracy_m16  = evaluate_standford_escapeMutants("/home/perm/sars_escape_netv2/data/model_results_archive/m16/sarsx_disc")
    accuracy_m17  = evaluate_standford_escapeMutants("/home/perm/sars_escape_netv2/data/model_results_archive/m17/sarsx_disc")
    accuracy_m18  = evaluate_standford_escapeMutants("/home/perm/sars_escape_netv2/data/model_results_archive/m18/sarsx_disc")
    accuracies  = {
                    'accuracy_m20': accuracy_m20,
                    'accuracy_m19': accuracy_m19,
                    'accuracy_m15': accuracy_m15,
                    'accuracy_m16': accuracy_m16,
                    'accuracy_m17' : accuracy_m17,
                    'accuracy_m18': accuracy_m18
                   }
    print(f'Accuracies: ', accuracies)
    '''
            