import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow import keras
from keras import backend
from keras import layers
import tf2onnx
import onnx
from keras.metrics import MeanMetricWrapper
from losses import *
tf.config.run_functions_eagerly(True)

@tf.keras.utils.register_keras_serializable()
class MyBlock(layers.Layer):

    def __init__(self,
                 neurons=256,
                 l2=1e-3,
                 activation='relu',
                 batch_norm=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons
        self.l2 = l2
        self.activation = None

        self.dense = keras.layers.Dense(neurons, activation=None, kernel_regularizer=keras.regularizers.L2(l2))
        
        if batch_norm:
            self.batchnorm = keras.layers.BatchNormalization()
        else:
            self.batchnorm = keras.layers.Identity()

        if activation.strip() == 'relu':
            self.activation = keras.layers.Activation(tf.nn.relu)
        elif activation.strip() == 'tanh':
            self.activation = keras.layers.Activation(tf.nn.tanh)
        elif activation.strip() == 'elu':
            self.activation = keras.layers.Activation(tf.nn.elu)
        else:
            raise ValueError(f'Unknown activation function: {activation}')

    def call(self, inputs, batch_size=None):
        xx = self.dense(inputs)
        xx = self.batchnorm(xx)
        xx = self.activation(xx)
        return xx

    def get_config(self):
        config = super().get_config()
        config['neurons'] = self.neurons
        config['l2'] = self.l2
        config['activation'] = self.activation
        config['name'] = self.name
        return config

@tf.keras.utils.register_keras_serializable()
class MyModelNN(keras.Model):

    def __init__(self,
                 input_shape,
                 neurons=256,
                 blocks=4,
                 l2=1e-3,
                 activation='relu',
                 loss='MSE',
                 output_size=4,
                 batch_norm=False,
                 **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons
        self.blocks = blocks
        self.l2 = l2
        self.output_size = output_size

        self.input_layer = keras.layers.InputLayer(input_shape=input_shape, name='input_1')
        self.dense_layers = [MyBlock(neurons, l2, activation, batch_norm) for ii in range(blocks)]
        self.output_layer = keras.layers.Dense(self.output_size, activation='linear')

        if loss == 'MSE':
            self.loss_metric = mean_squared_error_loss
        elif loss == 'M4E':
            self.loss_metric = mixed_loss
        elif loss == 'MAE':
            self.loss_metric = mean_absolute_error_loss
        elif loss == 'MAPE':
            self.loss_metric = mean_absolute_percentage_error
        else:
            raise ValueError('[ERROR] Unknown loss!')
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.mape_metric = keras.metrics.MeanAbsolutePercentageError(name="mape")
        self.mse_metric = keras.metrics.MeanSquaredError(name="mse")

    def call(self, inputs):
        xx = self.input_layer(inputs)
        for layer in self.dense_layers:
            xx = layer(xx)
        xx = self.output_layer(xx)
        xx = tf.cast(xx, tf.float64)
        return xx

    @tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, weights = data
            weights = tf.cast(tf.sqrt(weights), tf.float64)
        elif len(data) == 2:
            x, y = data
            tensor_dim = tf.shape(x)[0]
            weights = tf.ones(shape=(tensor_dim, self.output_size), dtype=tf.float64)
        else:
            return ValueError('[ERROR] Data should be 2 or 3 dimensional (samples, labels, weights)!')

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.loss_metric(y_true=y * weights, y_pred=y_pred * weights)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.mse_metric.update_state(y, y_pred)
        self.mae_metric.update_state(y, y_pred)
        self.mape_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mse": self.mse_metric.result(), "mae": self.mae_metric.result(), "mape": self.mape_metric.result()}

    @tf.function
    def test_step(self, data):
        if len(data) == 3:
            x, y, weights = data
            weights = tf.cast(tf.sqrt(weights), tf.float64)
        elif len(data) == 2:
            x, y = data
            tensor_dim = tf.shape(x)[0]
            weights = tf.ones(shape=(tensor_dim, self.output_size), dtype=tf.float64)
        else:
            return ValueError('[ERROR] Data should be 2 or 3 dimensional (samples, labels, weights)!')

        y_pred = self(x, training=False)
        loss = self.loss_metric(y_true=y * weights, y_pred=y_pred * weights)

        self.loss_tracker.update_state(loss)
        self.mse_metric.update_state(y, y_pred)
        self.mae_metric.update_state(y, y_pred)
        self.mape_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mse": self.mse_metric.result(), "mae": self.mae_metric.result(), "mape": self.mape_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mse_metric, self.mae_metric, self.mape_metric]
