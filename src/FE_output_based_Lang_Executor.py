import numpy as np 
import escape_validation as EV
import tensorflow as tf
from tensorflow.keras.models import load_model

import FE_Lang_Embed_Trainer  as eTrainer
MODEL_NAME= "FE_OUTPUT_MODEL"
tf.random.set_seed(42)
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]= "true"
os.environ["CUDA_VISIBLE_DEVICES"]="4" 

# baseline model
def get_base_model():
   from tensorflow.keras.models import Sequential
   from tensorflow.keras.layers import Dense   
   model = Sequential()
   model.add(Dense(512, input_shape=(616, ), activation='relu')) #input dimension - dimensions of features | input_shape is actual shape
   #model.add(Dense(128, activation='relu')) 
   model.add(Dense(64, activation='relu')) 
   #model.add(Dense(32, activation='relu')) 
   #model.add(Dense(16, activation='relu')) 
   #model.add(Dense(8, activation='relu'))
   model.add(Dense(1, activation='sigmoid'))
   # Compile model
   model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.AUC() ])
   return model

def get_training_data():
    sig_features_esc, non_sig_features_esc = eTrainer.get_features(eTrainer.MODEL_OUTPUT_FEATURE_ESC_SAVE_PATH)
    sig_features, non_sig_features = eTrainer.get_features(eTrainer.MODEL_OUTPUT_GISAID_FEATURE_SAVE_PATH)

    sig_features = np.concatenate( (sig_features, sig_features_esc) , axis=0)
    non_sig_features = np.concatenate( (non_sig_features[0: len(sig_features)], non_sig_features_esc), axis=0 )

    #/home/perm/cov/data/additional_escape_variants/gen/model_output_gisaid_features.npz

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
    model, pred_sig_features, target_sig_features_output = get_greany_target_and_predictions()

    #Evaluating greany fetures 
    print("Evaluating greany model fetures")
    model.evaluate(pred_sig_features, target_sig_features_output)

   

def get_greany_target_and_predictions():
    model =load_model(f"model/{MODEL_NAME}")
    sig_features_to_pred = eTrainer.get_greany_model_output_features()
    sig_len =  len(sig_features_to_pred)
    target_sig_features = np.ones( (sig_len, 1) )

    print(f'Shape: Greany input feature {sig_features_to_pred.shape}, output : {target_sig_features.shape}')
    return model,sig_features_to_pred,target_sig_features



def train_model():
    train_data, test_data = get_training_data()
    print("Train data shape: ", train_data.shape)
    train_x = train_data[:, 0:616]  #22*28 shape :
    print("Shape of train X: ", train_x.shape)
    train_y = train_data[:, -1]
    print("Shape of train Y: ", train_y.shape)

    test_x = test_data[:, 0:616]
    test_y = test_data[:, -1] #last column record
    
    model = get_base_model()
    
    model.fit(train_x, train_y, batch_size=32, epochs=10, validation_data=(test_x, test_y), verbose=1)
    model.save(f"model/{MODEL_NAME}")
    return model
def compute_auc(y, y_pred, thres=200):
    print("*******Computing AUC ")
    m = tf.keras.metrics.AUC(num_thresholds=thres)
    m.update_state(y, y_pred)
    print(m.result().numpy())




if __name__ == "__main__":
    #model = train_model()
    evaluate_greany() 
    #model,sig_features_to_pred,target_sig_features_output = get_greany_target_and_predictions()
    #predictions  = model.predict(sig_features_to_pred)
    #compute_auc(target_sig_features_output, predictions)
    pass




    
    

    
  
    
 

    















    

    