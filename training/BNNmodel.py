import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from tensorflow import keras
from keras import layers
import tf2onnx
import onnx
from NNmodel import MyBlock
tf.config.run_functions_eagerly(True)

# Define the prior weight distribution as Normal of mean=0 and stddev=1.
# Note that, in this example, the we prior distribution is not trainable,
# as we fix its parameters.
def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n)
                )
            )
        ]
    )
    return prior_model


# Define variational posterior weight distribution as multivariate Gaussian.
# Note that the learnable parameters for this distribution are the means,
# variances, and covariances.
def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    #print('dims', kernel_size, bias_size, n)
    params_size = tfp.layers.IndependentNormal.params_size(n, name='posterior_param_size')
    #print('params_size', params_size)
    posterior_model = keras.Sequential(
        [
            tfp.layers.VariableLayer(params_size, name='posterior_var'),
            tfp.layers.IndependentNormal(n, name='posterior_norm'),
        ]
    )
    return posterior_model



@tf.keras.utils.register_keras_serializable()
class MyBlockBNN(layers.Layer):

    def __init__(self,
                 neurons = 256, 
                 l2=1e-3,
                 activation = 'relu',
                 name='bnn',
                 **kwargs):
        super().__init__(**kwargs)
        self.neurons = neurons
        self.l2 = l2 
        self.activation = activation

        self.dense = tfp.layers.DenseVariational(units=neurons, 
                                                activation=None,
                                                make_prior_fn=prior,
                                                make_posterior_fn=posterior,
                                                use_bias=True,
                                                name=name,
                                                )

        self.batchnorm = keras.layers.BatchNormalization(name='batchnorm')
        if activation == 'relu':
            self.activation = keras.layers.Activation(tf.nn.relu, name=str(activation))
        else:
            raise ValueError(f'Unknown activation function: {activation}')



    def call(self, inputs, batch_size=None):
        kl_weight = 1 / batch_size if batch_size is not None else 1
        # print('KL weight', kl_weight)
        self.dense.kl_weight = kl_weight  # Update KL weight dynamically
        # print('input shape', tf.shape(inputs))
        # print('batch shape', batch_size)
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
class MyModelBNN(keras.Model):

    def __init__(self,
                input_shape,
                neurons = 256, 
                blocks=4, 
                l2=1e-3,
                activation = 'relu',
                loss='MSE',
                **kwargs):
        super().__init__(**kwargs)
        if loss!='MSE':
            raise NotImplementedError('Support is only for MSE!')

        self.neurons = neurons
        self.blocks = blocks
        self.l2 = l2 

        self.input_layer =  keras.layers.InputLayer(input_shape=input_shape, name='input_1')
        self.dense_layers = [MyBlockBNN(neurons, l2, activation, name=f'bnn_layer_{ii+1}') for ii in range(blocks)]
        self.output_layer = tfp.layers.DenseVariational(units=2, activation='relu',
                                                make_prior_fn=prior,
                                                make_posterior_fn=posterior,
                                                name='output_nLL_mu1'
                                                )

        self.nLL_exp_mu0 = tf.Variable(initial_value=tf.zeros(shape=(1,), dtype=tf.float64), trainable=False, dtype=tf.float64, name='nLL_exp_mu0')
        self.nLL_obs_mu0 = tf.Variable(initial_value=tf.zeros(shape=(1,), dtype=tf.float64), trainable=False, dtype=tf.float64, name='nLL_obs_mu0')
        self.first_iteration = True
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.mape_metric = keras.metrics.MeanAbsolutePercentageError(name="mape")

    def call(self, inputs):
        xx = self.input_layer(inputs)
        batch_s = tf.shape(xx)[0] if xx.shape[0] is None else xx.shape[0]
        for ii in range(len(self.dense_layers)):
            # print(f'[LAYER {ii}]')
            xx = self.dense_layers[ii](xx, batch_size=batch_s)
        xx = self.output_layer(xx)
        # batch_s = tf.shape(xx)[0] if xx.shape[0] is None else xx.shape[0]
        exp = tf.repeat(self.nLL_exp_mu0, batch_s, axis=0)
        exp = tf.reshape(exp, shape=(batch_s, 1))
        obs = tf.repeat(self.nLL_obs_mu0, batch_s, axis=0)
        obs = tf.reshape(obs, shape=(batch_s, 1))
        xx = tf.cast(xx, tf.float64)
        res = tf.concat([exp, tf.gather(xx, indices=[0], axis=1), obs, tf.gather(xx, indices=[1], axis=1)], axis=1)
        return res

    @tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, weights = data
            weights = tf.cast(tf.sqrt(weights), tf.float64)
            batch_size = tf.shape(y)[0]
        elif len(data) == 2:
            x, y = data
            batch_size = tf.shape(y)[0]
            tensor_dim = tf.shape(x)[0]
            weights = tf.ones(shape=(tensor_dim, 2), dtype=tf.float64) 
        else:
            return ValueError('[ERROR] Data should be 2 or 3 dimensional (samples, labels, weights)!')

        if self.first_iteration:
            val_exp = tf.stop_gradient(tf.math.reduce_mean(y[:, 0], axis=0))
            if len(val_exp.get_shape().as_list()) == 0:
                val_exp = tf.expand_dims(val_exp, axis=0)
            self.nLL_exp_mu0.assign(val_exp)
            
            val_obs = tf.stop_gradient(tf.math.reduce_mean(y[:, 2], axis=0))
            if len(val_obs.get_shape().as_list()) == 0:
                val_obs = tf.expand_dims(val_obs, axis=0)
            self.nLL_obs_mu0.assign(val_obs)

            tf.config.run_functions_eagerly(False)
            self.first_iteration = False

        y = tf.gather(y, indices=[1,3], axis=1)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_pred = tf.gather(y_pred, indices=[1,3], axis=1)
            loss = tf.keras.metrics.mean_squared_error(y_true=y*(weights), y_pred=y_pred*(weights))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        self.mape_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mape": self.mape_metric.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        y = tf.gather(y, indices=[1,3], axis=1)

        y_pred = self(x, training=False)
        y_pred = tf.gather(y_pred, indices=[1,3], axis=1)
        loss = tf.keras.metrics.mean_squared_error(y_true=y, y_pred=y_pred)

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        self.mape_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mape": self.mape_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric, self.mape_metric]


@tf.keras.utils.register_keras_serializable()
class MyModelBNNSimple(keras.Model):

    def __init__(self,
                input_shape,
                neurons = 256, 
                blocks=4, 
                l2=1e-3,
                activation = 'relu',
                loss='MSE',
                **kwargs):
        super().__init__(**kwargs)
        if loss!='MSE':
            raise NotImplementedError('Support is only for MSE!')

        self.neurons = neurons
        self.blocks = blocks
        self.l2 = l2 

        self.input_layer =  keras.layers.InputLayer(input_shape=input_shape, name='input_1')
        self.dense_layers = [MyBlock(neurons, l2, activation, name=f'bnn_layer_{ii+1}') for ii in range(blocks)]
        self.output_layer = tfp.layers.DenseVariational(units=2, activation='relu',
                                                make_prior_fn=prior,
                                                make_posterior_fn=posterior,
                                                name='output_nLL_mu1'
                                                )

        self.nLL_exp_mu0 = tf.Variable(initial_value=tf.zeros(shape=(1,), dtype=tf.float64), trainable=False, dtype=tf.float64, name='nLL_exp_mu0')
        self.nLL_obs_mu0 = tf.Variable(initial_value=tf.zeros(shape=(1,), dtype=tf.float64), trainable=False, dtype=tf.float64, name='nLL_obs_mu0')
        self.first_iteration = True
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.mae_metric = keras.metrics.MeanAbsoluteError(name="mae")
        self.mape_metric = keras.metrics.MeanAbsolutePercentageError(name="mape")

    def call(self, inputs):
        xx = self.input_layer(inputs)
        batch_s = tf.shape(xx)[0] if xx.shape[0] is None else xx.shape[0]
        for ii in range(len(self.dense_layers)):
            # print(f'[LAYER {ii}]')
            xx = self.dense_layers[ii](xx, batch_size=batch_s)
        xx = self.output_layer(xx)
        # batch_s = tf.shape(xx)[0] if xx.shape[0] is None else xx.shape[0]
        exp = tf.repeat(self.nLL_exp_mu0, batch_s, axis=0)
        exp = tf.reshape(exp, shape=(batch_s, 1))
        obs = tf.repeat(self.nLL_obs_mu0, batch_s, axis=0)
        obs = tf.reshape(obs, shape=(batch_s, 1))
        xx = tf.cast(xx, tf.float64)
        res = tf.concat([exp, tf.gather(xx, indices=[0], axis=1), obs, tf.gather(xx, indices=[1], axis=1)], axis=1)
        return res

    @tf.function
    def train_step(self, data):
        if len(data) == 3:
            x, y, weights = data
            weights = tf.cast(tf.sqrt(weights), tf.float64)
            batch_size = tf.shape(y)[0]
        elif len(data) == 2:
            x, y = data
            batch_size = tf.shape(y)[0]
            tensor_dim = tf.shape(x)[0]
            weights = tf.ones(shape=(tensor_dim, 2), dtype=tf.float64) 
        else:
            return ValueError('[ERROR] Data should be 2 or 3 dimensional (samples, labels, weights)!')

        if self.first_iteration:
            val_exp = tf.stop_gradient(tf.math.reduce_mean(y[:, 0], axis=0))
            if len(val_exp.get_shape().as_list()) == 0:
                val_exp = tf.expand_dims(val_exp, axis=0)
            self.nLL_exp_mu0.assign(val_exp)
            
            val_obs = tf.stop_gradient(tf.math.reduce_mean(y[:, 2], axis=0))
            if len(val_obs.get_shape().as_list()) == 0:
                val_obs = tf.expand_dims(val_obs, axis=0)
            self.nLL_obs_mu0.assign(val_obs)

            tf.config.run_functions_eagerly(False)
            self.first_iteration = False

        y = tf.gather(y, indices=[1,3], axis=1)
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            y_pred = tf.gather(y_pred, indices=[1,3], axis=1)
            loss = tf.keras.metrics.mean_squared_error(y_true=y*(weights), y_pred=y_pred*(weights))

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        self.mape_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mape": self.mape_metric.result()}

    @tf.function
    def test_step(self, data):
        x, y = data
        y = tf.gather(y, indices=[1,3], axis=1)

        y_pred = self(x, training=False)
        y_pred = tf.gather(y_pred, indices=[1,3], axis=1)
        loss = tf.keras.metrics.mean_squared_error(y_true=y, y_pred=y_pred)

        self.loss_tracker.update_state(loss)
        self.mae_metric.update_state(y, y_pred)
        self.mape_metric.update_state(y, y_pred)
        return {"loss": self.loss_tracker.result(), "mae": self.mae_metric.result(), "mape": self.mape_metric.result()}

    @property
    def metrics(self):
        return [self.loss_tracker, self.mae_metric, self.mape_metric]

