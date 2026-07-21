"""
=============================================================================
  2D-CNN MODEL ARCHITECTURE
=============================================================================
Implements the adapted 2D-CNN architecture from the exploratory notebook.
Configured for:
 - Input shape: (10, 256, 1) representing 10 channels x 2s @ 128Hz (256 samples)
 - Output shape: (2,) raw linear logits for API data contract compatibility.
=============================================================================
"""

import sys

# Optional heavy dependencies
try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
except Exception:
    tf = None
    layers = None
    models = None

def build_adapted_2d_cnn(input_shape=(10, 256, 1)):
    """
    Constructs the adapted Keras 2D-CNN model matching the notebook's layout
    but refactored for (10, 256, 1) inputs and (2,) logits outputs.
    """
    if tf is None or layers is None or models is None:
        raise ImportError("TensorFlow is required to compile this model")
        
    inputs = layers.Input(shape=input_shape)
    
    # ── Block 1 ──
    x = layers.Conv2D(
        filters=64, 
        kernel_size=(2, 4), 
        padding='same', 
        activation='relu'
    )(inputs)
    x = layers.Conv2D(
        filters=64, 
        kernel_size=(2, 4), 
        strides=(1, 2), 
        padding='same', 
        activation='relu'
    )(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)
    
    # ── Block 2 ──
    x = layers.Conv2D(
        filters=128, 
        kernel_size=(2, 4), 
        padding='same', 
        activation='relu'
    )(x)
    x = layers.Conv2D(
        filters=128, 
        kernel_size=(2, 4), 
        strides=(1, 2), 
        padding='same', 
        activation='relu'
    )(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    
    # ── Block 3 ──
    x = layers.Conv2D(
        filters=256, 
        kernel_size=(4, 4), 
        padding='same', 
        activation='relu'
    )(x)
    x = layers.Conv2D(
        filters=256, 
        kernel_size=(4, 4), 
        strides=(1, 2), 
        padding='same', 
        activation='relu'
    )(x)
    x = layers.MaxPooling2D(pool_size=(1, 2))(x)
    
    # ── Classification Head ──
    x = layers.GlobalAveragePooling2D(name="global_pooling")(x)
    
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.25)(x)
    
    # Logits output: 2 units (Normal vs Seizure), linear activation to prevent double softmax
    outputs = layers.Dense(2, activation='linear', name="logits")(x)
    
    return models.Model(inputs=inputs, outputs=outputs, name="EEG_2D_CNN_Seizure")
