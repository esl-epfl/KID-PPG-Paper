#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:36:00 2023

@author: kechris
"""

import numpy as np
from config import Config


import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.utils import shuffle
from sklearn.model_selection import LeaveOneGroupOut

from preprocessing import preprocessing_Dalia_aligned_preproc as pp
from preprocessing.data_generator_high_hr import DataGeneratorHighHR

from models.attention_models import build_attention_model

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
batch_size = 256
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

    X_train = X_train[:, :1, :]
    X_train = np.transpose(X_train, (0, 2, 1))


    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups = groups[test_val_indexes])
    for validate_indexes, test_indexes in logo.split(X_val_test, y_val_test, groups[test_val_indexes]):
        
        X_validate, X_test = X_val_test[validate_indexes], X_val_test[test_indexes]
        y_validate, y_test = y_val_test[validate_indexes], y_val_test[test_indexes]
        activity_validate, activity_test = activity_val_test[validate_indexes], activity_val_test[test_indexes]
        
        groups_val = groups[test_val_indexes]
        test_subject_id = groups_val[test_indexes][0]
        
        # Build Model
        model = build_attention_model((cf.input_shape, n_ch))

        
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
        checkpoint = ModelCheckpoint('./saved_models/adaptive_w_attention_high_hr/model_weights/model_S' + str(test_subject_id) + '.h5', 
                                     monitor = val_mae, verbose = 1, 
                                     save_best_only = True, save_weights_only = False, 
                                     mode = 'min', 
                                     save_freq = 'epoch')
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                                    patience = 150,
                                                    verbose = 1)


        # Setup optimizer
        adam = Adam(learning_rate = 0.0005, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
        model.compile(loss='mae', optimizer = adam, metrics=[mae])


        X_test = X_test[:, :1, :]
        X_validate = X_validate[:, :1, :]

        X_validate = np.transpose(X_validate, (0, 2, 1))
        X_test = np.transpose(X_test, (0, 2, 1))

        train_data = DataGeneratorHighHR(X_train, y_train, 
                                                   batch_size)

        # Training
        hist = model.fit(
            train_data, 
            epochs = n_epochs, 
            batch_size = batch_size,
            validation_data = (X_validate, y_validate), 
            verbose = 1, 
            callbacks =[checkpoint, early_stop])
        
        current_subject_counter += 1
end_time = time.time()
print("Done in ", (end_time - start_time) / 3600, " hours.")
