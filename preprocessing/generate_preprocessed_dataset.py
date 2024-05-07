from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
import tensorflow as tf
from config import Config

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.model_selection import LeaveOneGroupOut, GroupKFold
from sklearn.utils import shuffle

from scipy.io import loadmat

from preprocessing import preprocessing_Dalia_aligned as pp


import utils

import pickle

import matplotlib.pyplot as plt
import scipy 
from scipy import fft

from tqdm import tqdm 


from models.adaptive_linear_model import AdaptiveFilteringModel


def get_session(gpu_fraction=0.333):
    gpu_options = tf.compat.v1.GPUOptions(
            per_process_gpu_memory_fraction=gpu_fraction,
            allow_growth=True)
    return tf.compat.v1.Session(
            config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))
tf.compat.v1.keras.backend.set_session(get_session())

tf.keras.utils.set_random_seed(0) 
tf.config.experimental.enable_op_determinism()

def channel_wise_z_score_normalization(X):
    
    ms = np.zeros((X.shape[0], 4))
    stds = np.zeros((X.shape[0], 4))
    for i in range(X.shape[0]):
        curX = X[i, ...]
        
        for j in range(4): 
            std = np.std(curX[j, ...])
            m = np.mean(curX[j, ...])
            
            curX[j, ...] = curX[j, ...] - np.mean(curX[j, ...])
            
            if std != 0:
                curX[j, ...] = curX[j, ...] / std
            
            ms[i, j] = m
            stds[i, j] = std
        
        X[i, ...] = curX
        
    return X, ms, stds

def channel_wise_z_score_denormalization(X, ms, stds):
    
    for i in range(X.shape[0]):
        curX = X[i, ...]
        
        for j in range(X.shape[1]): 
            
            if stds[i, j] != 0:
                curX[j, ...] = curX[j, ...] * stds[i, j]
            
            curX[j, ...] = curX[j, ...] + ms[i, j]
        X[i, ...] = curX
    
    return X

def normalize_on_range(X):
    
    X_ = X.copy()
    
    X_[:, 0, :] = X_[:, 0, :] / 500
    
    if X.shape[1] > 1:
        X_[:, 1:, :] = X_[:, 1:, :] / 2
        
    return X_



n_epochs = 16000
batch_size = 256
n_ch = 1
patience = 150

# Setup config
cf = Config(search_type = 'NAS', root = './data/')

# Load data
X, y, groups, activity = pp.preprocessing(cf.dataset, cf)


activity = activity.flatten()

unique_groups = np.unique(groups)

all_data_X = []
all_data_y = []
all_data_groups = []
all_data_activity = []

for group in unique_groups:
    print("Processing S" + str(int(group)))
    cur_X = X[groups == group]
    
    cur_y = y[groups == group]
    cur_groups = groups[groups == group]
    cur_activity = activity[groups == group]
    
    indexes = np.argwhere(np.abs(np.diff(cur_activity)) > 0).flatten()
    indexes += 1
    indexes = np.insert(indexes, 0, 0)
    indexes = np.insert(indexes, indexes.size, cur_X.shape[0])
    
    filtered_Xs = []
    for i in tqdm(range(indexes.size - 1)):
        current_activity = cur_activity[indexes[i]]
    
        cur_activity_X = cur_X[indexes[i] : indexes[i + 1]]
        
        cur_activity_X, ms, stds = channel_wise_z_score_normalization(cur_activity_X)
        
        sgd = tf.keras.optimizers.legacy.SGD(learning_rate = 1e-7, 
                                                    momentum = 1e-2,)
        model = AdaptiveFilteringModel(local_optimizer = sgd,
                                       num_epochs_self_train = n_epochs)
        
        
        X_filtered = model(cur_activity_X[..., None]).numpy()
        
        X_filtered = X_filtered[:, None, :]
        X_filtered = channel_wise_z_score_denormalization(X_filtered, ms, stds)
        
    
        filtered_Xs.append(X_filtered)
    
    filtered_Xs = np.concatenate(filtered_Xs, axis = 0)

    all_data_X.append(filtered_Xs)
    all_data_y.append(cur_y)
    all_data_groups.append(cur_groups)
    all_data_activity.append(cur_activity)
    
all_data_X = np.concatenate(all_data_X, axis = 0)
all_data_y = np.concatenate(all_data_y, axis = 0)
all_data_groups = np.concatenate(all_data_groups, axis = 0)
all_data_activity = np.concatenate(all_data_activity , axis = 0)
    

data = dict()
data['X'] = all_data_X
data['y'] = all_data_y
data['groups'] = all_data_groups
data['act'] = all_data_activity

with open(cf.path_PPG_Dalia+'slimmed_dalia_aligned_prefiltered_80000.pkl', 'wb') as f:
    pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    