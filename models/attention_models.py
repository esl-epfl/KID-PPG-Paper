#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 24 18:43:08 2023

@author: kechris
"""

import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions

def convolution_block(input_shape, n_filters, 
                      kernel_size = 5, 
                      dilation_rate = 2,
                      pool_size = 2,
                      padding = 'causal'):
        
    mInput = tf.keras.Input(shape = input_shape)
    m = mInput
    for i in range(3):
        m = tf.keras.layers.Conv1D(filters = n_filters,
                                   kernel_size = kernel_size,
                                   dilation_rate = dilation_rate,
                                    padding = padding,
                                   activation = 'relu')(m)
    
    m = tf.keras.layers.AveragePooling1D(pool_size = pool_size)(m)
    m = tf.keras.layers.Dropout(rate = 0.5)(m)
    
    model = tf.keras.models.Model(inputs = mInput, outputs = m)
    
    return model

def build_attention_model(input_shape, return_attention_scores = False,
                          name = None):    
    mInput = tf.keras.Input(shape = input_shape)
    
    conv_block1 = convolution_block(input_shape, n_filters = 32,
                                    pool_size = 4)
    conv_block2 = convolution_block((64, 32), n_filters = 48)
    conv_block3 = convolution_block((32, 48), n_filters = 64)
    
    m_ppg = conv_block1(mInput)
    m_ppg = conv_block2(m_ppg)
    m_ppg = conv_block3(m_ppg)

    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads = 4,
                                                         key_dim = 16,
                                                         )
    if return_attention_scores:
        m, attention_weights = attention_layer(query = m_ppg, value = m_ppg,
                                               return_attention_scores = return_attention_scores)
    else:
        m = attention_layer(query = m_ppg, value = m_ppg,
                            return_attention_scores = return_attention_scores)
    
    m = tf.keras.layers.LayerNormalization()(m)
        
    m = tf.keras.layers.Flatten()(m)
    m = tf.keras.layers.Dense(units = 32, activation = 'relu')(m)
    m = tf.keras.layers.Dense(units = 1)(m)
    
    if return_attention_scores:
        model = tf.keras.models.Model(inputs = mInput, 
                                      outputs = [m, attention_weights],
                                      name = name)
    else:
        model = tf.keras.models.Model(inputs = mInput, outputs = m,
                                      name = name)
    
    model.summary()
    
    return model 

def my_dist(params):
    return tfd.Normal(loc=params[:,0:1], 
                      scale = 1 + tf.math.softplus(params[:,1:2]))# both parameters are learnable

def build_attention_model_probabilistic(input_shape, return_attention_scores = False,
                          name = None):    
    mInput = tf.keras.Input(shape = input_shape)
    
    conv_block1 = convolution_block(input_shape, n_filters = 32,
                                    pool_size = 4)
    conv_block2 = convolution_block((64, 32), n_filters = 48)
    conv_block3 = convolution_block((32, 48), n_filters = 64)
    
    m_ppg = conv_block1(mInput)
    m_ppg = conv_block2(m_ppg)
    m_ppg = conv_block3(m_ppg)

    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads = 4,
                                                         key_dim = 16,
                                                         )
    if return_attention_scores:
        m, attention_weights = attention_layer(query = m_ppg, value = m_ppg,
                                               return_attention_scores = return_attention_scores)
    else:
        m = attention_layer(query = m_ppg, value = m_ppg,
                            return_attention_scores = return_attention_scores)
    
    m = tf.keras.layers.LayerNormalization()(m)
        
    m = tf.keras.layers.Flatten()(m)
    m = tf.keras.layers.Dense(units = 256, activation = 'relu')(m)
    m = tf.keras.layers.Dropout(rate = 0.125)(m)
    m = tf.keras.layers.Dense(units = 2)(m)
    
    m = tfp.layers.DistributionLambda(my_dist)(m)
    
    model = tf.keras.models.Model(inputs = mInput, outputs = m)
        
    model.summary()
    
    return model 