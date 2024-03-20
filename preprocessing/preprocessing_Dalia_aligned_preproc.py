#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:37:31 2023

@author: kechris
"""

#*----------------------------------------------------------------------------*
#* Copyright (C) 2021 Politecnico di Torino, Italy                            *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");            *
#* you may not use this file except in compliance with the License.           *
#* You may obtain a copy of the License at                                    *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                 *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software        *
#* distributed under the License is distributed on an "AS IS" BASIS,          *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
#* See the License for the specific language governing permissions and        *
#* limitations under the License.                                             *
#*                                                                            *
#* Author:  Matteo Risso                                                      *
#*----------------------------------------------------------------------------*

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
