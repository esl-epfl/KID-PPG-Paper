import pickle
import numpy as np
from skimage.util.shape import view_as_windows
from scipy.io import loadmat
import random
import os

def preprocessing(dataset, cf):
    # Sampling frequency of both ppg and acceleration data in IEEE_Training dataset
    fs_IEEE_Training = 125
    # Sampling frequency of acceleration data in PPG_Dalia dataset
    # The sampling frequency of ppg data in PPG_Dalia dataset is fs_PPG_Dalia*2
    fs_PPG_Dalia = 32
    
    fs_activity = 4
    
    Sessioni = dict()
    S = dict()
    acc = dict()
    ppg = dict()
    activity = dict()
    
    random.seed(20)
     
    ground_truth = dict()
    
    val = dataset
    
   
    with open(cf.path_PPG_Dalia+'slimmed_dalia_aligned_prefiltered_80000.pkl', 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        
        X = data['X']
        y = data['y']
        groups = data['groups']
        act = data['act']
   
    print("dimensione train",X.shape, "dimesione test", y.shape,"dimensione gruppi",groups.shape)
    
    return X[:y.shape[0]], y, groups[:y.shape[0]], act[:y.shape[0]]
