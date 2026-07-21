"""
=============================================================================
  EEGNET MODEL ARCHITECTURE
=============================================================================
Implements a standard clinical EEGNet model for raw EEG signals.
Designed to extract native temporal and spatial channel-wise features.
Input shape: (10, 256, 1) - 10 channels x 2s @ 128Hz.
Output shape: (2,) linear logits.
=============================================================================
"""

import sys

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None
    layers = None
    models = None

def build_eegnet(input_shape=(10, 256), dropout_rate=0.25, kernel_length=64, F1=8, D=2, F2=16):
    """
    Keras implementation of the standard EEGNet model.
    References: https://iopscience.iop.org/article/10.1088/1741-2552/aace8c
    """
    if tf is None or layers is None or models is None:
        raise ImportError("TensorFlow is required to compile this model")
        
    inputs = layers.Input(shape=input_shape)
    if len(input_shape) == 2:
        x = layers.Reshape((input_shape[0], input_shape[1], 1), name="eegnet_reshape")(inputs)
    else:
        x = inputs
    
    # ── Block 1: Temporal Conv & Spatial Depthwise Conv ──
    # 1. Temporal Conv
    x = layers.Conv2D(
        filters=F1, 
        kernel_size=(1, kernel_length), 
        padding='same', 
        use_bias=False,
        name="temporal_conv"
    )(x)
    x = layers.BatchNormalization(name="temp_bn")(x)
    
    # 2. Depthwise Conv (spatial filtering)
    x = layers.DepthwiseConv2D(
        kernel_size=(input_shape[0], 1), 
        padding='valid', 
        depth_multiplier=D, 
        use_bias=False,
        depthwise_constraint=tf.keras.constraints.MaxNorm(max_value=1.0),
        name="spatial_depthwise_conv"
    )(x)
    x = layers.BatchNormalization(name="spatial_bn")(x)
    x = layers.Activation('elu', name="spatial_elu")(x)
    x = layers.AveragePooling2D(pool_size=(1, 4), name="spatial_avgpool")(x)
    x = layers.Dropout(dropout_rate, name="spatial_dropout")(x)
    
    # ── Block 2: Separable Conv ──
    x = layers.SeparableConv2D(
        filters=F2, 
        kernel_size=(1, 16), 
        padding='same', 
        use_bias=False,
        name="separable_conv"
    )(x)
    x = layers.BatchNormalization(name="separable_bn")(x)
    x = layers.Activation('elu', name="separable_elu")(x)
    x = layers.AveragePooling2D(pool_size=(1, 8), name="separable_avgpool")(x)
    x = layers.Dropout(dropout_rate, name="separable_dropout")(x)
    
    # ── Classification Head ──
    x = layers.Flatten(name="flatten")(x)
    # Output 2 logits (Normal vs Seizure)
    outputs = layers.Dense(
        2, 
        activation='linear', 
        kernel_constraint=tf.keras.constraints.MaxNorm(max_value=0.25),
        name="logits"
    )(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="EEGNet_Seizure")
