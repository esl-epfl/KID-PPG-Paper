#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 19:26:04 2023

@author: kechris
"""

import numpy as np
from config import Config


# aliases
val_mae = 'val_mean_absolute_error'
mae = 'mean_absolute_error'

import tensorflow as tf

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut

from preprocessing import preprocessing_Dalia_aligned_preproc as pp


from models import build_TEMPONet

import pandas as pd

import time

def get_session(gpu_fraction=0.333):
    gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction,
            allow_growth=True)
    return tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(get_session())

tf.keras.utils.set_random_seed(0) 
tf.config.experimental.enable_op_determinism()

n_epochs = 500
batch_size = 128

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

# Setup config
cf = Config(search_type = 'NAS', root = './data/')

# Load data
X, y, groups, activity = pp.preprocessing(cf.dataset, cf)


group_ids = np.unique(groups)
group_ids = shuffle(group_ids)

n_groups_in_split = int(group_ids.size / 4) + 1

splits = np.array_split(group_ids, n_groups_in_split)

groups_pd = pd.Series(groups)

current_subject_counter = 0

start_time = time.time()
for split in splits:
    X, y, _, _ = pp.preprocessing(cf.dataset, cf)

    
    test_val_indexes = groups_pd.isin(split)
    train_indexes = ~test_val_indexes
    
    X_train, X_val_test = X[train_indexes], X[test_val_indexes]
    y_train, y_val_test = y[train_indexes], y[test_val_indexes]
    activity_train, activity_val_test = activity[train_indexes], activity[test_val_indexes]

    
    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups = groups[test_val_indexes])
    for validate_indexes, test_indexes in logo.split(X_val_test, y_val_test, groups[test_val_indexes]):
        
        X_validate, X_test = X_val_test[validate_indexes], X_val_test[test_indexes]
        y_validate, y_test = y_val_test[validate_indexes], y_val_test[test_indexes]
        activity_validate, activity_test = activity_val_test[validate_indexes], activity_val_test[test_indexes]
        
        groups_val = groups[test_val_indexes]
        test_subject_id = groups_val[test_indexes][0]
        
        # Build Model
        model = build_TEMPONet.TEMPONet_learned(1, cf.input_shape, 
                                                dil_ht = False,
                                                dil_list = dil, 
                                                ofmap = ofmap,
                                                n_ch = n_ch)
        
        print("===========================================")
        print("Test Subject: S" + str(int(test_subject_id)) + " (" \
              + str(current_subject_counter + 1) + " /15) ")
        val_groups = np.unique(groups_val[validate_indexes])
        for val_group in val_groups:
            print("\tValidating with S" + str(int(val_group)))
        print("===========================================")

        val_mae = 'val_mean_absolute_error'
        mae = 'mean_absolute_error'
        
        # save model weights
        checkpoint = ModelCheckpoint('./saved_models/adaptive_w_q_ppg/model_weights/model_S' + str(test_subject_id) + '.h5', 
                                     monitor = val_mae, verbose = 1, 
                                     save_best_only = True, save_weights_only = False, 
                                     mode = 'min', 
                                     save_freq = 'epoch')
        
        early_stop = EarlyStopping(monitor = val_mae, 
                                   min_delta = 0.01, 
                                   patience = 35, 
                                   mode = 'min', 
                                   verbose = 1)

        # Setup optimizer
        adam = Adam(learning_rate = cf.lr, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
        model.compile(loss='logcosh', optimizer = adam, metrics=[mae])


        X_train, y_train = shuffle(X_train, y_train)

        # ACC has already been processed during the preprocessing step so 
        # the Q-PPG only takes as an input the PPG. 
        
        X_train = X_train[:, :1, :]
        X_test = X_test[:, :1, :]
        X_validate = X_validate[:, :1, :]

        # Training
        hist = model.fit(
            x = np.transpose(X_train.reshape(X_train.shape[0], n_ch, cf.input_shape, 1), (0, 3, 2, 1)), 
            y = y_train, 
            epochs = n_epochs, 
            batch_size = batch_size,
            validation_data = (np.transpose(X_validate.reshape(X_validate.shape[0], n_ch, cf.input_shape, 1), (0, 3, 2, 1)), y_validate), 
            verbose = 1, 
            callbacks =[checkpoint, early_stop])
        
        current_subject_counter += 1
end_time = time.time()
print("Done in ", (end_time - start_time) / 3600, " hours.")
