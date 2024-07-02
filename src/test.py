import numpy as np
import pandas as pd
import tensorflow.keras.models as m

def get_features(windowed_file):
    model = m.load_model('/home/perm/cov/github_escape_code/pretrain_models/feature_learning_model/bilstm_256-04.hdf5')
    seqs = read_combined_window_file(windowed_file)

    X_cat, lengths = featurize_seqs(seqs)
    y_embed_output = model.transform(X_cat, lengths )
    print("Shape of output previously: ", y_embed_output.shape) #  (rows, 22, 512)
    #Reducing the feature taking the avarage from middle axis
    y_embed_output = np.average(y_embed_output, axis=1)
    print("Reduced feature Shape after average: ", y_embed_output.shape)
    return y_embed_output
	
def featurize_seqs(seqs):
	vocabulary = getVocabDictionary()
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
	
def read_combined_window_file(file_path):
    combined_features = []
    df = pd.read_csv(file_path)
    #Appending features | 
    for row in df.itertuples():
        combined_features.append(row.wild)
        combined_features.append(row.mutated)
    print("Combined Features Length: ", len(combined_features))
    #Ensure that features lenght is always even
    assert(len(combined_features) % 2 == 0)
    return combined_features
	
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
	
def analyze_greany():
    model = m.load_model("/home/perm/cov/model/integrated_model")
    model.summary()
    features = get_features("test_dataset.csv")
    y_preds = model.predict(features)
    print(y_preds)


if __name__ == "__main__":
    analyze_greany()
    
