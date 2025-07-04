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
    return 100*backend.mean( tf.math.abs( tf.math.reciprocal_no_nan(y_true) * tf.subtract(y_true, y_pred) ) )

