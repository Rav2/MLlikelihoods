import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow import keras
from keras import backend
from keras import layers


class BaseLoss(keras.losses.Loss):
    """Base class for all custom losses with optional per-output weighting."""
    
    def __init__(self, output_weights=None, name="base_loss", **kwargs):
        super().__init__(name=name, **kwargs)
        if output_weights is None:
            output_weights = [1.0, 1.0, 1.0, 1.0]
        self.output_weights = tf.constant(output_weights, dtype=tf.float32)
    
    def call(self, y_true, y_pred):
        raise NotImplementedError


class MeanFourthError(BaseLoss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        loss_per_output = tf.math.square(tf.math.squared_difference(y_pred, y_true))
        weighted_loss = loss_per_output * self.output_weights
        return backend.mean(weighted_loss, axis=-1)


class MixedLoss(BaseLoss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        squared_diff_sq = tf.math.square(tf.math.squared_difference(y_pred, y_true))
        squared_diff = tf.math.squared_difference(y_pred, y_true)
        weighted_term1 = tf.reduce_mean(squared_diff_sq * self.output_weights, axis=-1)
        weighted_term2 = tf.reduce_mean(squared_diff * self.output_weights, axis=-1)
        return weighted_term1 + weighted_term2


class MeanSquaredErrorLoss(BaseLoss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mse_per_output = tf.math.squared_difference(y_true, y_pred)
        weighted_loss = mse_per_output * self.output_weights
        return tf.reduce_mean(weighted_loss, axis=-1)


class MeanAbsoluteErrorLoss(BaseLoss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mae_per_output = tf.math.abs(y_true - y_pred)
        weighted_loss = mae_per_output * self.output_weights
        return tf.reduce_mean(weighted_loss, axis=-1)


class MeanAbsolutePercentageError(BaseLoss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        mape_per_output = 100 * tf.math.abs(tf.math.reciprocal_no_nan(y_true) * tf.subtract(y_true, y_pred))
        weighted_loss = mape_per_output * self.output_weights
        return tf.reduce_mean(weighted_loss, axis=-1)


class HybridLoss(BaseLoss):
    def __init__(self, alpha=0.5, output_weights=None, **kwargs):
        super().__init__(output_weights=output_weights, name="hybrid_loss", **kwargs)
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        rel_loss = tf.abs(y_true - y_pred) / (tf.abs(y_true) + 1e-3)
        combined_loss = self.alpha * abs_loss + (1 - self.alpha) * rel_loss
        weighted_loss = combined_loss * self.output_weights
        return tf.reduce_mean(weighted_loss, axis=-1)


class AdaptiveWeightedLoss(BaseLoss):
    def __init__(self, alpha=0.5, output_weights=None, **kwargs):
        super().__init__(output_weights=output_weights, name="adaptive_weighted_loss", **kwargs)
        self.alpha = alpha
    
    def call(self, y_true, y_pred):
        abs_loss = tf.abs(y_true - y_pred)
        rel_loss = tf.abs(y_true - y_pred) / (tf.abs(y_true) + 1e-3)
        combined_loss = self.alpha * abs_loss + (1 - self.alpha) * rel_loss
        
        var_per_output = tf.math.reduce_variance(y_true, axis=0, keepdims=True)
        normalized_var = var_per_output / (tf.reduce_mean(var_per_output) + 1e-6)
        
        weighted_loss = combined_loss * normalized_var
        return tf.reduce_mean(weighted_loss, axis=-1)


class LogCoshLoss(BaseLoss):
    def call(self, y_true, y_pred):
        diff = y_pred - y_true
        loss_per_output = tf.math.log(tf.cosh(diff))
        weighted_loss = loss_per_output * self.output_weights
        return tf.reduce_mean(weighted_loss, axis=-1)


class TripleLoss(BaseLoss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        absolute = tf.abs(y_true - y_pred)
        rooted = tf.math.sqrt(absolute + 1e-7)
        squared = tf.math.squared_difference(y_pred, y_true)
        
        combined_loss = 1.0 * rooted + 0.1 * absolute + 0.01 * squared
        weighted_loss = combined_loss * self.output_weights
        
        return tf.reduce_mean(weighted_loss, axis=-1)


class HuberLoss(BaseLoss):
    def __init__(self, delta=1.0, output_weights=None, **kwargs):
        super().__init__(output_weights=output_weights, name="huber_loss", **kwargs)
        self.delta = delta
    
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        diff = y_true - y_pred
        huber_loss_per_output = tf.where(
            tf.abs(diff) <= self.delta,
            0.5 * tf.square(diff),
            self.delta * (tf.abs(diff) - 0.5 * self.delta)
        )
        weighted_loss = huber_loss_per_output * self.output_weights
        return tf.reduce_mean(weighted_loss, axis=-1)


class MeanSquaredLogarithmicError(BaseLoss):
    def call(self, y_true, y_pred):
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        
        log_true = tf.math.log(tf.maximum(y_true, 1e-7))
        log_pred = tf.math.log(tf.maximum(y_pred, 1e-7))
        msle_per_output = tf.square(log_true - log_pred)
        weighted_loss = msle_per_output * self.output_weights
        return tf.reduce_mean(weighted_loss, axis=-1)