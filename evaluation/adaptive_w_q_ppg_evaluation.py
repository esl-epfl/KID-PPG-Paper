import numpy as np
from config import Config

from tensorflow.keras.optimizers import Adam

from sklearn.utils import shuffle


from preprocessing import preprocessing_Dalia_aligned_preproc as pp


from models import build_TEMPONet

import tensorflow as tf

import matplotlib.pyplot as plt
import os 

n_epochs = 200
batch_size = 128

overall_errors = []
activity_errors = np.empty((15, 9))
activity_errors[:] = np.nan

overall_percent_errors = []

for test_subject_id in range(1, 16):
    
    # Setup config
    cf = Config(search_type = 'NAS', root = './data/')
    
    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
    
    
    X_train = X[groups != test_subject_id]
    y_train = y[groups != test_subject_id]
    
    X_test = X[groups == test_subject_id]
    y_test = y[groups == test_subject_id]
    
    X_validate = X_test
    y_validate = y_test
    
    activity_validate = activity[groups == test_subject_id]
    
    ofmap = [
                32, 32, 63,
                62, 64, 128,
                89, 45, 38,
                50, 61, 1
            ]
    dil = [
        1, 1, 2,
        2, 1,
        2, 2
    ]
    
    n_ch = 1
    
    model = build_TEMPONet.TEMPONet_learned(1, cf.input_shape, 
                                            dil_ht = False,
                                            dil_list = dil, 
                                            ofmap = ofmap,
                                            n_ch = n_ch)
    
    model.load_weights('./saved_models/adaptive_w_q_ppg/model_weights/model_S' + str(int(test_subject_id)) + '.h5')
    
    
    X_train = X_train[:, :1, :]
    X_test = X_test[:, :1, :]
    X_validate = X_validate[:, :1, :]
    
    
    with tf.device('/cpu:0'):
        y_pred = model.predict(X_validate[..., None])
        
    plt.figure()
    plt.plot(y_pred, label = 'Prediction')
    plt.plot(y_test, label = 'Ground truth')
    plt.legend()
    
    error = np.mean(np.abs(y_pred - y_test))
    overall_errors.append(error)
    
    error_percent = np.sum(np.abs(y_pred - y_test) > 5) / y_pred.size * 100
    overall_percent_errors.append(error_percent)
    
    for cur_act in np.unique(activity_validate):
        
        error = np.mean(np.abs(y_pred[activity_validate == cur_act] \
                               - y_test[activity_validate == cur_act]))
            
        activity_errors[test_subject_id - 1, int(cur_act)] = error
      
    output_path = './results/model_predictions/adaptive_w_q_ppg/'
    isExist = os.path.exists(output_path)
    if not os.path.exists(output_path):
       os.makedirs(output_path)
       
    # Save predictions for plotting 
    np.save(output_path + 'S' \
            + str(int(test_subject_id)) + '.npy', y_pred)
    