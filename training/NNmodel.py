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
# tf.config.run_functions_eagerly(True)


@tf.keras.utils.register_keras_serializable()
class MyBlock(layers.Layer):

    def __init__(self,
                 neurons=256,
                 l2=1e-3,
                 activation='relu',
                 batch_norm=False,
                 dropout_rate=0.0,
                 use_residual = False,
                 **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons
        self.l2 = l2
        self.activation_name = activation
        self.activation = None
        self.dropout_rate = dropout_rate
        self.use_residual = use_residual

        self.dense = keras.layers.Dense(neurons, 
                                        activation=None, 
                                        kernel_regularizer=keras.regularizers.L2(l2), 
                                        name=f"{self.name}_dense")
        
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
        elif activation.strip() == 'relu6':
            self.activation = keras.layers.Activation(tf.nn.relu6)
        elif activation.strip() == 'swish':
            self.activation = keras.layers.Activation(tf.nn.swish)
        else:
            raise ValueError(f'Unknown activation function: {activation}')

        self.dropout = keras.layers.Dropout(self.dropout_rate)

        self.projection = None
    
    def build(self, input_shape):
        # Add projection if dimensions don't match for residual
        if self.use_residual and input_shape[-1] != self.neurons:
            self.projection = keras.layers.Dense(
                self.neurons,
                kernel_regularizer=keras.regularizers.L2(self.l2),
                use_bias=False
            )
        super().build(input_shape)

    @tf.function(reduce_retracing=True)  
    def call(self, inputs, batch_size=None, training=None):
        xx = self.dense(inputs)
        xx = self.batchnorm(xx, training=training)
        xx = self.activation(xx)
        xx = self.dropout(xx, training=training)

        # Residual connection
        if self.use_residual:
            if self.projection is not None:
                inputs = self.projection(inputs)
            xx = xx + inputs
        return xx

    def get_config(self):
        config = super().get_config()
        config['neurons'] = self.neurons
        config['l2'] = self.l2
        config['activation'] = self.activation_name
        config['name'] = self.name
        config['dropout_rate'] = self.dropout_rate
        config['use_residual'] = self.use_residual
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
                 dropout_rate=0.0,
                 use_residual = False,
                 width = 'equal',
                 **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons
        self.blocks = blocks
        self.l2 = l2
        self.output_size = output_size

        self.input_layer = keras.layers.InputLayer(input_shape=input_shape, name='input_1')
        dense_layers = []
        for ii in range(blocks):
            if width == 'equal':
                power = 0
            elif width == 'contract':
                power = -1*ii
            elif width == 'expand':
                power = ii 
            else:
                raise ValueError(f"Unrecognised width option: {width}. Available options: equal, expand, contract") 
            dense_layers.append(MyBlock( max(neurons*int(2**power), 1),  
                    l2, activation, batch_norm, dropout_rate, use_residual,
                    name=f"block_{ii}"))

        self.dense_layers = dense_layers
        self.output_layer = keras.layers.Dense(self.output_size, activation='linear', name='output_layer')

        if loss == 'MSE':
            self.loss_metric = mean_squared_error_loss
        elif loss == 'M4E':
            self.loss_metric = mixed_loss
        elif loss == 'MAE':
            self.loss_metric = mean_absolute_error_loss
        elif loss == 'MAPE':
            self.loss_metric = mean_absolute_percentage_error
        elif loss == 'hybrid':
            self.loss_metric = hybrid_loss
        elif loss == 'huber':
            self.loss_metric = keras.losses.Huber(delta=1.0)
        elif loss == 'log_cosh':
            self.loss_metric = log_cosh_loss
        else:
            raise ValueError('[ERROR] Unknown loss!')
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.mape_metric = keras.metrics.MeanAbsolutePercentageError(name="mape")
        self.mse_metric = keras.metrics.MeanSquaredError(name="mse")
    
    @tf.function(reduce_retracing=True)  
    def call(self, inputs, training=None):
        xx = self.input_layer(inputs)
        for layer in self.dense_layers:
            xx = layer(xx, training=training)
        xx = self.output_layer(xx)
        xx = tf.cast(xx, tf.float32)
        return xx

    @tf.function(reduce_retracing=True)  
    def train_step(self, data):
        if len(data) == 3:
            x, y, sample_weights = data
            sample_weights = tf.cast(sample_weights, tf.float32)
        elif len(data) == 2:
            x, y = data
            sample_weights = None
        else:
            raise ValueError('[ERROR] Data should be 2 or 3 dimensional!')
        
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            # Compute per-sample loss
            loss_per_sample = self.loss_metric(y_true=y, y_pred=y_pred)
            
            # Apply sample weights if provided
            if sample_weights is not None:
                loss_per_sample = loss_per_sample * sample_weights
            
            # Compute mean loss
            loss = tf.reduce_mean(loss_per_sample)
            
            # Add regularization losses
            if self.losses:
                loss += tf.add_n(self.losses)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        # Gradient clipping to prevent exploding gradients
        gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        # Update metrics without weights for fair comparison
        self.loss_tracker.update_state(loss)
        self.mse_metric.update_state(y, y_pred)
        self.mae_metric.update_state(y, y_pred)
        self.mape_metric.update_state(y, y_pred)
        
        return {
            "loss": self.loss_tracker.result(), 
            "mse": self.mse_metric.result(), 
            "mae": self.mae_metric.result(), 
            "mape": self.mape_metric.result()
        }

    @tf.function(reduce_retracing=True)  
    def test_step(self, data):
        if len(data) == 3:
            x, y, weights = data
            weights = tf.cast(tf.sqrt(weights), tf.float32)
        elif len(data) == 2:
            x, y = data
            tensor_dim = tf.shape(x)[0]
            weights = tf.ones(shape=(tensor_dim, self.output_size), dtype=tf.float64)
        else:
            return ValueError('[ERROR] Data should be 2 or 3 dimensional (samples, labels, weights)!')
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        weights = tf.cast(weights, tf.float32)

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
