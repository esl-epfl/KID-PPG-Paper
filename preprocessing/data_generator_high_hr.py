import math
import tensorflow as tf
from sklearn.utils import shuffle
import scipy
from scipy import signal

import numpy as np

import scipy.fft

from sklearn.covariance import EmpiricalCovariance

from tqdm import tqdm 

class DataGeneratorHighHR(tf.keras.utils.Sequence):
    
    def __init__(self, X, y, batch_size):
        self.X_in = X
        self.y_in = y
        self.batch_size = batch_size
        self.fs = 32.0

        
        self.create_speedup_signals()
        self.setup_easy_spedup_indexes()
        
        self.create_augmented_dataset()
        
    def __getitem__(self, index):
        
        return self.X_out[index * self.batch_size : (index + 1) * self.batch_size], \
            self.y_out[index * self.batch_size : (index + 1) * self.batch_size]
    
    def __len__(self):
        return int(np.floor(self.X_out.shape[0]) / self.batch_size)
    
       
    def setup_easy_spedup_indexes(self):
        f_ = np.argmax(np.abs(tf.signal.fft(self.X_spedup[:, :, 0]))[:, :128].T, axis = 0) * 7.5
        
        self.easy_spedup_indexes = np.argwhere(np.abs(f_.flatten() - self.y_spedup.flatten()) < 10)
        
    
    def create_augmented_dataset(self):
                
        tmp_X_spedup_easy = self.X_spedup[self.easy_spedup_indexes.flatten(), ...]
        tmp_y_spedup_easy = self.y_spedup[self.easy_spedup_indexes.flatten(), ...]

        
        self.X_out = np.concatenate([self.X_in, 
                                     tmp_X_spedup_easy,
                                     ], axis = 0)
        
        self.y_out = np.concatenate([self.y_in, 
                                     tmp_y_spedup_easy,
                                     ], axis = 0)
        
        self.X_out, self.y_out = shuffle(self.X_out, self.y_out)
        
        
    def create_speedup_signals(self):
        self.X_spedup = np.concatenate([self.X_in[:-4, ...], 
                                        self.X_in[4:, ...]], 
                                       axis = 1)[:, ::2, :]
        self.y_spedup = 2 * self.y_in[:-4]
        
        self.X_spedup = self.X_spedup[self.y_spedup.flatten() < 300]
        self.y_spedup = self.y_spedup[self.y_spedup.flatten() < 300]
        
    def on_epoch_end(self):
        # self.create_augmented_dataset()
        self.X_out, self.y_out = shuffle(self.X_out, self.y_out)
   