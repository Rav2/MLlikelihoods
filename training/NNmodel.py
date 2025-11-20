import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow import keras
from keras import backend
from keras import layers
from keras.metrics import MeanMetricWrapper
from losses import *


@tf.keras.utils.register_keras_serializable()
class MyBlock(layers.Layer):
    """
    A reusable neural network block with dense layer, batch normalization, 
    activation, and optional residual connections.
    
    This block encapsulates a common pattern: dense -> batch norm -> activation -> dropout,
    with optional skip connections for residual learning.
    """

    def __init__(self,
                 neurons=256,
                 l2=1e-3,
                 activation='relu',
                 batch_norm=False,
                 dropout_rate=0.0,
                 use_residual = False,
                 **kwargs):
        """
        Initialize a MyBlock layer.
        
        Args:
            neurons (int): Number of output units in the dense layer. Default: 256.
            l2 (float): L2 regularization coefficient. Default: 1e-3.
            activation (str): Activation function name ('relu', 'tanh', 'elu', 'relu6', 'swish'). Default: 'relu'.
            batch_norm (bool): Whether to apply batch normalization. Default: False.
            dropout_rate (float): Dropout rate between 0 and 1. Default: 0.0.
            use_residual (bool): Whether to add residual/skip connections. Default: False.
            **kwargs: Additional keyword arguments passed to parent Layer class.
        
        Raises:
            ValueError: If activation function is not recognized.
        """
        super().__init__(**kwargs)
        self.neurons = neurons
        self.l2 = l2
        self.activation_name = activation
        self.batch_norm_flag = batch_norm
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
        """
        Forward pass of the block.
        
        Args:
            inputs (Tensor): Input tensor.
            batch_size (int, optional): Batch size (unused, kept for compatibility). Default: None.
            training (bool, optional): Whether in training mode (affects dropout and batch norm). Default: None.
        
        Returns:
            Tensor: Output tensor after applying dense, batch norm, activation, dropout, and optional residual.
        """
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
        config['batch_norm'] = self.batch_norm_flag
        config['dropout_rate'] = self.dropout_rate
        config['use_residual'] = self.use_residual
        return config

@tf.keras.utils.register_keras_serializable()
class MyModelNN(keras.Model):
    """
    A flexible neural network model composed of stacked MyBlock layers with optional separate output heads.
    
    Supports various configurations including residual connections, batch normalization, different 
    loss functions, and customizable network architectures via neuron specifications.
    """

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
                 use_residual=False,
                 separate_heads=False,
                 head_size=256,
                 head_batch_norm=False,
                 gradient_clipping=1.0,
                 **kwargs):
        """
        Initialize the MyModelNN model.
        
        Args:
            input_shape (tuple): Shape of input features (excluding batch dimension).
            neurons (int or list): Number of neurons per block. Can be:
                - An integer: all blocks use this number of neurons. Default: 256.
                - A list of integers: each block uses the corresponding neuron count.
                  If provided, 'blocks' parameter is ignored with a warning.
            blocks (int): Number of stacked blocks (ignored if neurons is a list). Default: 4.
            l2 (float): L2 regularization coefficient. Default: 1e-3.
            activation (str): Activation function name. Default: 'relu'.
            loss (str): Loss function ('MSE', 'MAE', 'MAPE', 'hybrid', 'huber', 'log_cosh', 'triple', 'MSLE', 'M4E'). Default: 'MSE'.
            output_size (int): Number of output units. Default: 4.
            batch_norm (bool): Whether to apply batch normalization in blocks. Default: False.
            dropout_rate (float): Dropout rate in blocks. Default: 0.0.
            use_residual (bool): Whether to use residual connections in blocks. Default: False.
            separate_heads (bool): Whether to use separate output heads per output dimension. Default: False.
            head_size (int): Number of neurons in each separate head. Default: 256.
            head_batch_norm (bool): Whether to apply batch normalization in separate heads. Default: False.
            gradient_clipping (float): Global norm threshold for gradient clipping. Must be positive. Default: 1.0.
            **kwargs: Additional keyword arguments passed to parent Model class.
        
        Raises:
            ValueError: If gradient_clipping is not positive or if loss option is unrecognized.
        """
        super().__init__(**kwargs)
        
        # Handle neurons specification
        if isinstance(neurons, list):
            neuron_list = neurons
            actual_blocks = len(neuron_list)
            if blocks != actual_blocks:
                from logger import logging
                logging.warning(f"neurons is specified as a list with {actual_blocks} elements. "
                               f"Ignoring blocks={blocks} and using {actual_blocks} blocks instead.")
        else:
            neuron_list = [neurons] * blocks
            actual_blocks = blocks
        
        self.neurons = neurons
        self.blocks = actual_blocks
        self.l2 = l2
        self.output_size = output_size
        self.separate_heads = separate_heads
        self.head_size = head_size
        self.head_batch_norm = head_batch_norm
        
        if gradient_clipping <= 0:
            raise ValueError('gradient_clipping must be a positive float value')
        self.gradient_clipping = gradient_clipping

        self.input_layer = keras.layers.InputLayer(input_shape=input_shape, name='input_1')
        dense_layers = []
        for ii in range(actual_blocks):
            dense_layers.append(MyBlock(neuron_list[ii],  
                    l2, activation, batch_norm, dropout_rate, use_residual,
                    name=f"block_{ii}"))

        self.dense_layers = dense_layers
        self.heads = []
        self.head_outputs = []
        if self.separate_heads:
            for hh in range(0, self.output_size):
                self.heads.append(MyBlock(self.head_size,  
                    l2, activation, self.head_batch_norm, 0.0, False,
                    name=f"head_{hh}"))
                self.head_outputs.append(keras.layers.Dense(1, activation='linear', name=f'head_output_{hh}'))
            self.output_layer = keras.layers.Concatenate(name='output_layer')
        else:
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
        elif loss == 'triple':
            self.loss_metric = triple_loss
        elif loss == 'MSLE':
            self.loss_metric = keras.losses.MeanSquaredLogarithmicError()
        elif loss == 'hybrid-weighted':
            self.loss_metric = weighted_hybrid_loss
        elif loss == 'adaptive-weighted':
            self.loss_metric = adaptive_weighted_loss
        elif loss == 'log_cosh-weighted':
            self.loss_metric = weighted_log_cosh_loss
        elif loss == 'triple-weighted':
            self.loss_metric = weighted_triple_loss
        else:
            raise ValueError('[ERROR] Unknown loss!')
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.mape_metric = keras.metrics.MeanAbsolutePercentageError(name="mape")
        self.mse_metric = keras.metrics.MeanSquaredError(name="mse")
    
    @tf.function(reduce_retracing=True)  
    def call(self, inputs, training=None):
        """
        Forward pass of the model.
        
        Args:
            inputs (Tensor): Input tensor with shape matching input_shape.
            training (bool, optional): Whether in training mode. Default: None.
        
        Returns:
            Tensor: Output predictions with shape (batch_size, output_size).
        """
        xx = self.input_layer(inputs)
        for layer in self.dense_layers:
            xx = layer(xx, training=training)
        if self.separate_heads:
            head_outputs = []
            for ii in range(len(self.heads)):
                # Process through head block, then through corresponding output layer
                head_out = self.heads[ii](xx, training=training)
                head_out = self.head_outputs[ii](head_out)
                head_outputs.append(head_out)
            xx = self.output_layer(head_outputs)  # concatenates
        else:    
            xx = self.output_layer(xx)
        xx = tf.cast(xx, tf.float32)
        return xx

    @tf.function(reduce_retracing=True)  
    def train_step(self, data):
        """
        Custom training step with support for sample weights and per-sample loss weighting.
        
        Args:
            data (tuple): Either (x, y) or (x, y, sample_weights) where:
                - x: Input features
                - y: Target labels
                - sample_weights (optional): Per-sample weights for weighted loss
        
        Returns:
            dict: Dictionary containing loss, mse, mae, and mape metrics.
        
        Raises:
            ValueError: If data is not 2 or 3 dimensional.
        """
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
        gradients, _ = tf.clip_by_global_norm(gradients, self.gradient_clipping)
        
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
        """
        Custom evaluation step with support for sample weights.
        
        Args:
            data (tuple): Either (x, y) or (x, y, weights) where:
                - x: Input features
                - y: Target labels
                - weights (optional): Per-sample weights for weighted loss computation
        
        Returns:
            dict: Dictionary containing loss, mse, mae, and mape metrics.
        
        Raises:
            ValueError: If data is not 2 or 3 dimensional.
        """
        if len(data) == 3:
            x, y, weights = data
            weights = tf.cast(weights, tf.float32)
        elif len(data) == 2:
            x, y = data
            tensor_dim = tf.shape(x)[0]
            weights = tf.ones(shape=(tensor_dim,), dtype=tf.float32)
        else:
            raise ValueError('[ERROR] Data should be 2 or 3 dimensional (samples, labels, weights)!')
        
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        weights = tf.cast(weights, tf.float32)

        y_pred = self(x, training=False)
        # Apply weights consistently: per-sample weighting applied after loss computation
        loss_per_sample = self.loss_metric(y_true=y, y_pred=y_pred)
        loss = tf.reduce_mean(loss_per_sample * weights)

        self.loss_tracker.update_state(loss)
        self.mse_metric.update_state(y, y_pred)
        self.mae_metric.update_state(y, y_pred)
        self.mape_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mse": self.mse_metric.result(), "mae": self.mae_metric.result(), "mape": self.mape_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mse_metric, self.mae_metric, self.mape_metric]