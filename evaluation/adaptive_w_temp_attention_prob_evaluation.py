#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 11:23:46 2023

@author: kechris
"""


import numpy as np
from config import Config

from tensorflow.keras.optimizers import Adam

from sklearn.utils import shuffle


from preprocessing import preprocessing_Dalia_aligned_preproc as pp


from models.temporal_attention_models import build_model_probabilistic

import tensorflow as tf

import matplotlib.pyplot as plt

import os 

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

std_threshold = 5

n_epochs = 200
batch_size = 128

overall_errors = []
overall_errors_std_thr = []
activity_errors = np.empty((15, 9))
activity_errors[:] = np.nan

percentages_dropped = []

nll_e = []

error_vs_std = []
for test_subject_id in range(1, 16):
    
    n_epochs = 100
    batch_size = 256
    n_ch = 2
    patience = 150

    fs = 32.0

    # Setup config
    cf = Config(search_type = 'NAS', root = './data/')

    # Load data
    X, y, groups, activity = pp.preprocessing(cf.dataset, cf)

    X = X[:, 0, :]
    # X = normalize_on_range(X)
    X, y, groups, activity = create_temporal_pairs(X, y, groups, activity)

    X_train = X[groups != test_subject_id]
    y_train = y[groups != test_subject_id]

    X_test = X[groups == test_subject_id]
    y_test = y[groups == test_subject_id].flatten()

    X_validate = X_test
    y_validate = y_test
    activity_validate = activity[groups == test_subject_id]

    model = build_model_probabilistic((cf.input_shape, n_ch))
    model.load_weights('./saved_models/adaptive_w_temp_attention_prob/model_weights/model_S' + str(int(test_subject_id)) + '.h5')


    submodel = tf.keras.models.Model(inputs = model.inputs, outputs = model.layers[-2].output)

    with tf.device('/cpu:0'):
        y_pred = submodel.predict(X_validate[..., None])
        
    y_pred_m = y_pred[:, 0]
    y_pred_std = (1 + tf.math.softplus(y_pred[:,1:2])).numpy().flatten()
    
    
    with tf.device('/cpu:0'):
        loss = NLL(y_test, model(X_validate[..., None])).numpy()
        loss = np.diagonal(loss)
        
    nll_e.append(loss.mean())
        
    x = np.arange(y_pred_m.size)

    
    error = np.mean(np.abs(y_pred_m - y_test))
    overall_errors.append(error)
    
    error_std_thr = np.mean(np.abs(y_pred_m[y_pred_std < std_threshold] - y_test[y_pred_std < std_threshold]))
    overall_errors_std_thr.append(error_std_thr)
    
    percentage_dropped = np.argwhere(y_pred_std < std_threshold).size / y_test.size    
    percentages_dropped.append(percentage_dropped)
    
    
    for cur_act in np.unique(activity_validate):
        
        error = np.mean(np.abs(y_pred_m[activity_validate == cur_act] \
                               - y_test[activity_validate == cur_act]))
            
        activity_errors[test_subject_id - 1, int(cur_act)] = error
        
    e = []
    for thr in np.arange(1, 10, 0.5):
        e.append(np.mean(np.abs(y_pred_m[y_pred_std < thr] \
                                - y_test[y_pred_std < thr])))
    
    e = np.array(e)
    
    error_vs_std.append(e)
    
    output_path = './results/model_predictions/adaptive_w_temp_attention_prob/'
    isExist = os.path.exists(output_path)
    if not os.path.exists(output_path):
       os.makedirs(output_path)
       
    # Save predictions for plotting 
    np.save(output_path + 'S' \
            + str(int(test_subject_id)) + '.npy', y_pred)
        
    np.save(output_path + 'S' \
            + str(int(test_subject_id)) + '_loss.npy', loss)
    
    
        
overall_errors = np.array(overall_errors)
nll_e = np.array(nll_e)
overall_errors_std_thr = np.array(overall_errors_std_thr)

percentages_dropped = np.array(percentages_dropped)

error_vs_std = np.stack(error_vs_std, axis = 0)

error_vs_std_m = np.mean(error_vs_std, axis = 0)
error_vs_std_std = np.std(error_vs_std, axis = 0)

x = np.arange(error_vs_std_m.size)

plt.figure()
plt.plot(x, error_vs_std_m, '-o')
plt.fill_between(x, error_vs_std_m - error_vs_std_std,  
                 error_vs_std_m + error_vs_std_std, alpha = 0.25)
