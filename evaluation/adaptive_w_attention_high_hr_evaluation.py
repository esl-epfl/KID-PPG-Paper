import numpy as np
from config import Config

from tensorflow.keras.optimizers import Adam

from sklearn.utils import shuffle


from preprocessing import preprocessing_Dalia_aligned_preproc as pp


from models.attention_models import build_attention_model

import tensorflow as tf

import matplotlib.pyplot as plt

from tqdm import tqdm

import seaborn as sns

import os


n_epochs = 200
batch_size = 128

overall_errors = []
activity_errors = np.empty((15, 9))
activity_errors[:] = np.nan

overall_percent_errors = []

for test_subject_id in range(1, 16):
    
    n_epochs = 100
    batch_size = 256
    n_ch = 1
    patience = 150

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


    model = build_attention_model((cf.input_shape, n_ch))

    model.load_weights('./saved_models/adaptive_w_attention_high_hr/model_weights/model_S' + str(int(test_subject_id)) + '.h5')

    X_validate = X_validate[:, :1, :]
    X_validate = np.transpose(X_validate, (0, 2, 1))
    
    with tf.device('/cpu:0'):
        y_pred = model.predict(X_validate)
        

    error = np.mean(np.abs(y_pred - y_test))
    overall_errors.append(error)
    
    error_percent = np.sum(np.abs(y_pred - y_test) > 5) / y_pred.size * 100
    overall_percent_errors.append(error_percent)
    
    for cur_act in np.unique(activity_validate):
        
        error = np.mean(np.abs(y_pred[activity_validate == cur_act] \
                               - y_test[activity_validate == cur_act]))
        print(test_subject_id, cur_act, error)
            
        activity_errors[test_subject_id - 1, int(cur_act)] = error
    
    output_path = './results/model_predictions/adaptive_w_attention_high_hr/'
    isExist = os.path.exists(output_path)
    if not os.path.exists(output_path):
       os.makedirs(output_path)
       
    # Save predictions for plotting 
    np.save(output_path + 'S' \
            + str(int(test_subject_id)) + '.npy', y_pred)
        
overall_errors = np.array(overall_errors)
overall_percent_errors = np.array(overall_percent_errors)

print("Expected MAE: ", np.mean(overall_errors))
