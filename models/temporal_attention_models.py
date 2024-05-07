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
    m = tf.keras.layers.Dropout(rate = 0.5)(m, training = False)
        
    model = tf.keras.models.Model(inputs = mInput, outputs = m)
    
    return model
    
def my_dist(params):
    return tfd.Normal(loc=params[:,0:1], 
                      scale = 1 + tf.math.softplus(params[:,1:2]))# both parameters are learnable
       

def build_model_probabilistic(input_shape, return_attention_weights = False):
    modal_input_shape = (input_shape[0], 1)
    
    mInput = tf.keras.Input(shape = input_shape)
    
    mInput_t_1 = mInput[..., :1]
    mInput_t = mInput[..., 1:]
    
    conv_block1 = convolution_block(modal_input_shape, n_filters = 32,
                                    pool_size = 4)
    conv_block2 = convolution_block((64, 32), n_filters = 48)
    conv_block3 = convolution_block((32, 48), n_filters = 64)
    
    m_ppg_t_1 = conv_block1(mInput_t_1)
    m_ppg_t_1 = conv_block2(m_ppg_t_1)
    m_ppg_t_1 = conv_block3(m_ppg_t_1)
    
    m_ppg_t = conv_block1(mInput_t)
    m_ppg_t = conv_block2(m_ppg_t)
    m_ppg_t = conv_block3(m_ppg_t)
    
    
    attention_layer = tf.keras.layers.MultiHeadAttention(num_heads = 4,
                                                         key_dim = 16,
                                                         )
    
    if return_attention_weights:
        m, attention_scores = attention_layer(query = m_ppg_t, value = m_ppg_t_1, return_attention_scores=True)
    else:
        m = attention_layer(query = m_ppg_t, value = m_ppg_t_1, return_attention_scores = False)
    
    m = m + m_ppg_t
    
    m = tf.keras.layers.LayerNormalization()(m)
    
        
    m = tf.keras.layers.Flatten()(m)
    m = tf.keras.layers.Dense(units = 256, activation = 'relu')(m)
    m = tf.keras.layers.Dropout(rate = 0.125)(m)
    m = tf.keras.layers.Dense(units = 2)(m)
    
    m = tfp.layers.DistributionLambda(my_dist)(m)
    
    if return_attention_weights:
        model = tf.keras.models.Model(inputs = mInput, outputs = [m, attention_scores])
    else:
        model = tf.keras.models.Model(inputs = mInput, outputs = m)
        
    model.summary()
    
    return model 