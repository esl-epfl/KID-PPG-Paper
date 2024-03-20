#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 15:29:37 2023

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


from models.temporal_attention_models import build_model_probabilistic

from preprocessing.data_generator_probabilistic_augmantation import DataGeneratorHighHRNegativeExamples

import pandas as pd

import time

from tqdm import tqdm 
import scipy
from scipy import signal

def get_session(gpu_fraction=0.333):
    gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction,
            allow_growth=True)
    return tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(get_session())

def create_temporal_pairs(X_in, y_in, groups_in, activity_in):
    
    allXs = []
    allys = []
    allgroups = []
    allactivities = []
    for group in np.unique(groups_in):
        
        curX_in = X_in[groups_in == group]
        cury_in = y_in[groups_in == group]
        cur_groups_in = groups_in[groups_in == group]
        cur_activity_in = activity_in[groups_in == group]
        
        cur_X = np.concatenate([curX_in[:-1, ...][..., None], 
                                curX_in[1:, ...][..., None]], axis = -1)
        cur_y = cury_in[1:]
        cur_groups = cur_groups_in[1:]
        cur_activity = cur_activity_in[1:]
        
        allXs.append(cur_X)
        allys.append(cur_y)
        allgroups.append(cur_groups)
        allactivities.append(cur_activity)
        
    X = np.concatenate(allXs, axis = 0)
    y = np.concatenate(allys, axis = 0)
    groups = np.concatenate(allgroups, axis = 0)
    activity = np.concatenate(allactivities, axis = 0)
        
    return X, y, groups, activity

def NLL(y, distr):
    return -distr.log_prob(y)

n_epochs = 500
batch_size = 256
n_ch = 2

# Setup config
cf = Config(search_type = 'NAS', root = './data/')

# Load data
X, y, groups, activity = pp.preprocessing(cf.dataset, cf)


group_ids = np.unique(groups)
group_ids = shuffle(group_ids)

n_groups_in_split = int(group_ids.size / 4) + 1

splits = np.array_split(group_ids, n_groups_in_split)


current_subject_counter = 0

start_time = time.time()
for split in splits:
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)
    X = X[:, 0, :]
    
    X_filtered = np.zeros(X.shape)
    y_filtered = np.random.uniform(low = 1.0, high = 300, 
                                   size = y.shape)

    slack = 2.5

    for i in tqdm(range(X.shape[0]), ascii = True):
        hr = y.flatten()[i]
        
        f_low = (hr - slack) / 60.0
        f_high = (hr + slack) / 60.0
        b = scipy.signal.firwin(81, [f_low, f_high], fs = 32.0, pass_zero = "bandpass")
        y1 = scipy.signal.filtfilt(b, 1, X[i , :])
        
        f_low = (2 * hr - slack) / 60.0
        f_high = (2 * hr + slack) / 60.0
        b = scipy.signal.firwin(81, [f_low, f_high], fs = 32.0, pass_zero = "bandpass")
        y2 = scipy.signal.filtfilt(b, 1, X[i , :])
        
        f_low = (3 * hr - slack) / 60.0
        f_high = (3 * hr + slack) / 60.0
        b = scipy.signal.firwin(81, [f_low, f_high], fs = 32.0, pass_zero = "bandpass")
        y3 = scipy.signal.filtfilt(b, 1, X[i , :])
        
        X_filtered[i, :] = X[i, :] - (y1 + y2 + y3)
    
    groups_filtered = groups.copy()
    activity_filtered = activity.copy()
    
    X, y, groups, activity = create_temporal_pairs(X, y, groups, activity)
    X_filtered, y_filtered, groups_filtered, activity_filtered \
        = create_temporal_pairs(X_filtered, y_filtered, groups_filtered, activity_filtered)

    groups_pd = pd.Series(groups)
    test_val_indexes = groups_pd.isin(split)
    train_indexes = ~test_val_indexes
    
    X_train, X_val_test = X[train_indexes], X[test_val_indexes]
    y_train, y_val_test = y[train_indexes], y[test_val_indexes]
    activity_train, activity_val_test = activity[train_indexes], activity[test_val_indexes]

    X_filtered_train, X_filtered_val_test = X_filtered[train_indexes], X_filtered[test_val_indexes]
    y_filtered_train, y_filtered_val_test = y_filtered[train_indexes], y_filtered[test_val_indexes]
    
    logo = LeaveOneGroupOut()
    logo.get_n_splits(groups = groups[test_val_indexes])
    for validate_indexes, test_indexes in logo.split(X_val_test, y_val_test, groups[test_val_indexes]):
        
        X_validate, X_test = X_val_test[validate_indexes], X_val_test[test_indexes]
        y_validate, y_test = y_val_test[validate_indexes], y_val_test[test_indexes]
        activity_validate, activity_test = activity_val_test[validate_indexes], activity_val_test[test_indexes]
        
        X_filtered_validate, X_filtered_test = X_filtered_val_test[validate_indexes], X_filtered_val_test[test_indexes]
        y_filtered_validate, y_filtered_test = y_filtered_val_test[validate_indexes], y_filtered_val_test[test_indexes]
        
        groups_val = groups[test_val_indexes]
        test_subject_id = groups_val[test_indexes][0]
        
        # Build Model
        model = build_model_probabilistic((cf.input_shape, n_ch))

        
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
        checkpoint = ModelCheckpoint('./saved_models/adaptive_w_temp_attention_prob_full_augment/model_weights/model_S' + str(test_subject_id) + '.h5', 
                                     monitor = 'val_loss', verbose = 1, 
                                     save_best_only = True, save_weights_only = False, 
                                     mode = 'min', 
                                     save_freq = 'epoch')
        
        
        early_stop = tf.keras.callbacks.EarlyStopping(monitor = 'val_loss', 
                                                    patience = 150,
                                                    verbose = 1)

        # Setup optimizer
        adam = Adam(learning_rate = 0.0005, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-08)
        model.compile(loss = NLL, optimizer = adam)


        train_data = DataGeneratorHighHRNegativeExamples(X_train, y_train,
                                                         X_filtered_train, y_filtered_train,
                                                         batch_size,
                                                         ratio_random_samples = 0.5)

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
