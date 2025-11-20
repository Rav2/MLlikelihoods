import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras import layers
from keras.metrics import MeanMetricWrapper

def mean_fourth_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return backend.mean(tf.math.square(tf.math.squared_difference(y_pred, y_true)), axis=-1)

def mixed_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return 1e-2*backend.mean(tf.math.square(tf.math.squared_difference(y_pred, y_true)), axis=-1) + backend.mean(tf.math.squared_difference(y_pred, y_true), axis=-1)

def mean_squared_error_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

def mean_absolute_error_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    return tf.keras.metrics.mean_absolute_error(y_true, y_pred)

def mean_absolute_percentage_error(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    # Should average over output dimension, not all dimensions
    return tf.reduce_mean(100 * tf.math.abs(tf.math.reciprocal_no_nan(y_true) * tf.subtract(y_true, y_pred)), axis=-1)

def hybrid_loss(y_true, y_pred, alpha=0.5):
    """Standard hybrid loss treating all outputs equally"""
    abs_loss = tf.abs(y_true-y_pred)
    rel_loss = tf.abs(y_true-y_pred)/(tf.abs(y_true)+1e-3)
    return tf.reduce_mean(alpha*abs_loss+(1-alpha)*rel_loss, axis=-1)

def weighted_hybrid_loss(y_true, y_pred, alpha=0.5, output_weights=None):
    """
    Hybrid loss with per-output weighting.
    
    Args:
        y_true: Ground truth values, shape [batch_size, 4]
        y_pred: Predictions, shape [batch_size, 4]
        alpha: Balance between absolute and relative loss (0-1)
        output_weights: List/array of 4 weights for [exp, obs, exp_asimov, obs_asimov]
                       Default: [1.0, 2.0, 1.0, 2.0] to emphasize Observed outputs
    
    Output indices:
        0: Delta_nLL_exp (Expected)
        1: Delta_nLL_obs (Observed) - typically noisier
        2: Delta_nLLA_exp (Expected Asimov)
        3: Delta_nLLA_obs (Observed Asimov) - typically noisier
    """
    if output_weights is None:
        # Weight Observed outputs 2x more (indices 1 and 3)
        output_weights = [1.0, 3.0, 1.0, 1.0]
    
    output_weights = tf.constant(output_weights, dtype=y_pred.dtype)
    
    abs_loss = tf.abs(y_true - y_pred)
    rel_loss = tf.abs(y_true - y_pred) / (tf.abs(y_true) + 1e-3)
    
    # Combine absolute and relative loss
    combined_loss = alpha * abs_loss + (1 - alpha) * rel_loss
    
    # Apply per-output weights
    weighted_loss = combined_loss * output_weights
    
    # Average across outputs (with weights built in)
    return tf.reduce_mean(weighted_loss, axis=-1)

def adaptive_weighted_loss(y_true, y_pred, alpha=0.5):
    """
    Adaptive weighted loss that learns to weight outputs based on their variance.
    
    Outputs with higher variance in training get higher weight to prevent
    the model from ignoring them.
    """
    abs_loss = tf.abs(y_true - y_pred)
    rel_loss = tf.abs(y_true - y_pred) / (tf.abs(y_true) + 1e-3)
    combined_loss = alpha * abs_loss + (1 - alpha) * rel_loss
    
    # Compute per-output variance of targets
    # Higher variance outputs get higher weight
    var_per_output = tf.math.reduce_variance(y_true, axis=0, keepdims=True)
    normalized_var = var_per_output / (tf.reduce_mean(var_per_output) + 1e-6)
    
    weighted_loss = combined_loss * normalized_var
    return tf.reduce_mean(weighted_loss, axis=-1)

def log_cosh_loss(y_true, y_pred):
    """Log-cosh loss - smooth approximation of MAE, less sensitive to outliers"""
    diff = y_pred - y_true
    return tf.reduce_mean(tf.math.log(tf.cosh(diff)), axis=-1)

def weighted_log_cosh_loss(y_true, y_pred, output_weights=None):
    """
    Log-cosh loss with per-output weighting.
    Less sensitive to outliers than MSE while still emphasizing harder outputs.
    """
    if output_weights is None:
        output_weights = [1.0, 3.0, 1.0, 1.0]
    
    output_weights = tf.constant(output_weights, dtype=y_pred.dtype)
    
    diff = y_pred - y_true
    loss_per_output = tf.math.log(tf.cosh(diff))
    weighted_loss = loss_per_output * output_weights
    
    return tf.reduce_mean(weighted_loss, axis=-1)

def triple_loss(y_true, y_pred):
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    absolute  = tf.abs(y_true-y_pred)
    rooted = tf.math.sqrt(absolute+1e-7) # eps for stability
    squared = tf.math.squared_difference(y_pred, y_true)
    return tf.reduce_mean(1.0*rooted + 0.1*absolute + 0.01*squared, axis=-1)

def weighted_triple_loss(y_true, y_pred, output_weights=None):
    """
    Triple loss (rooted + absolute + squared) with per-output weighting.
    
    Args:
        y_true: Ground truth values, shape [batch_size, 4]
        y_pred: Predictions, shape [batch_size, 4]
        output_weights: List/array of 4 weights for [exp, obs, exp_asimov, obs_asimov]
                       Default: [1.0, 3.0, 1.0, 1.0] to emphasize nLL_obs only
    
    Output indices:
        0: Delta_nLL_exp (Expected)
        1: Delta_nLL_obs (Observed) - weighted 3x higher
        2: Delta_nLLA_exp (Expected Asimov)
        3: Delta_nLLA_obs (Observed Asimov)
    """
    if output_weights is None:
        # Weight only Observed (index 1) with 3x higher precision
        output_weights = [1.0, 3.0, 1.0, 1.0]
    
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    output_weights = tf.constant(output_weights, dtype=y_pred.dtype)
    
    absolute = tf.abs(y_true - y_pred)
    rooted = tf.math.sqrt(absolute + 1e-7)  # eps for stability
    squared = tf.math.squared_difference(y_pred, y_true)
    
    # Combine different loss scales
    combined_loss = 1.0 * rooted + 0.1 * absolute + 0.01 * squared
    
    # Apply per-output weights
    weighted_loss = combined_loss * output_weights
    
    # Average across outputs (with weights built in)
    return tf.reduce_mean(weighted_loss, axis=-1)