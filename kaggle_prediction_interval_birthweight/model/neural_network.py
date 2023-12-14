"""
DenseMissing layer that outputs an expected RELU whenever input features are missing.

Adapted from https://github.com/lstruski/Processing-of-missing-data-by-neural-networks/
    blob/master/mlp.py
Paper: https://arxiv.org/pdf/1805.07405.pdf
EM algorithm: https://arxiv.org/pdf/1209.0521.pdf and https://arxiv.org/pdf/1902.03335.pdf
"""

from typing import List, Optional, Tuple

import numpy as np
import scipy.stats as st
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from kaggle_prediction_interval_birthweight.data.data_processing import SOFTPLUS_SCALE
from kaggle_prediction_interval_birthweight.model.sampling_utils import (
    compute_highest_density_interval,
    np_softplus,
)

TFPI = tf.constant(3.14159265359, dtype=tf.float32)


@tf.function
def mvn_loglikelihood(
    x: tf.Tensor, mu: tf.Tensor, var: tf.Tensor, return_log: bool = True
) -> tf.Tensor:
    """
    Compute the row-wise multivariate normal log-likelihood.

    Parameters
    ----------
    x: tf.Tensor
        Observations, possibly containing missing values.
    mu: tf.Tensor
        means of x
    var: tf.Tensor
        variances of x
    return_log: bool
        if True, the log is returned.

    Returns
    -------
    tf.Tensor
        log likelihood values of x
    """
    q = -tf.pow(x - mu, 2) / var / 2 - tf.math.log(2 * TFPI * var) / 2
    log_likelihood = tf.reduce_sum(
        tf.where(tf.math.is_nan(q), tf.constant(0.0, dtype=tf.float32), q), axis=-1
    )
    return log_likelihood if return_log else tf.math.exp(log_likelihood)


@tf.function
def expected_relu(mu: tf.Tensor, sigma_sq: tf.Tensor) -> tf.Tensor:
    """
    Compute the expected RELU function when integrated over missing values.

    Parameters
    ----------
    mu: tf.Tensor
        Either observed data (where sigma_sq is 0) or means for unobserved (where sigma_sq > 0)
    sigma_sq: tf.Tensor
        Variances

    Returns
    -------
    tf.Tensor
        expected RELU values
    """
    where_missing = tf.not_equal(sigma_sq, tf.constant(0.0, dtype=tf.float32))
    imputed_sigma = tf.where(where_missing, sigma_sq, tf.fill(tf.shape(sigma_sq), 1e-20))
    sqrt_sigma = tf.sqrt(imputed_sigma)

    w = tf.divide(mu, sqrt_sigma)
    expected_relu_values = sqrt_sigma * (
        tf.divide(
            tf.math.exp(tf.divide(-tf.square(w), 2.0)),
            tf.math.sqrt(2 * TFPI),
        )
        + tf.multiply(
            tf.divide(w, 2.0),
            1 + tf.math.erf(tf.divide(w, tf.math.sqrt(tf.constant(2, dtype=tf.float32)))),
        )
    )
    expected_relu_values = tf.where(
        where_missing, expected_relu_values, (mu + tf.math.abs(mu)) / 2.0
    )
    return expected_relu_values


class DenseMissing(tf.keras.layers.Layer):
    """Extend the Dense layer to handle missing data."""

    def __init__(self, units: int, n_components: int = 3, gamma: float = 1e-6, **kwargs) -> None:
        """
        Parameters
        ----------
        units: int
            number of units in the dense layer
        n_components: int
            number of components of the gaussian mixture distribution
        gamma: float
            regularization value for variance estimates
        kwargs: dict
            other arguments passed to tf.keras.layers.Layer
        """
        super(DenseMissing, self).__init__(**kwargs)
        self.units = units
        self.n_components = n_components
        self.gamma = gamma
        self.iteration = 0

    def compute_output_shape(self, input_shape: Tuple[int]) -> Tuple[int]:
        """
        Compute the output shape of the layer.

        Parameters
        ----------
        input_shape: Tuple[int]
            the input shape

        Returns
        -------
        Tuple[int]
            the output shape
        """
        return input_shape[:-1] + (self.units)

    def build(self, input_shape: Tuple[int]) -> None:
        """
        Build the layer.

        Parameters
        ----------
        input_shape: Tuple[int]
            the input shape
        """
        super(DenseMissing, self).build(input_shape)
        self.component_means = tf.Variable(
            tf.zeros((input_shape[-1], self.n_components), dtype=tf.float32),
            trainable=False,
            name="component_means",
        )
        self.component_vars = tf.Variable(
            tf.ones((input_shape[-1], self.n_components), dtype=tf.float32),
            trainable=False,
            name="component_vars",
        )
        self.component_logits = tf.Variable(
            tf.ones((self.n_components,), dtype=tf.float32),
            trainable=False,
            name="component_logits",
        )
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.units),
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.units,),
            initializer="zeros",
            trainable=True,
            name="bias",
        )

    def call(self, inputs: tf.Tensor, training: Optional[bool] = None) -> tf.Tensor:
        """
        Forward propagation.

        Parameters
        ----------
        inputs: tf.Tensor
            the input tensor
        training: bool
            if True, an EM step updates the gaussian mixture parameters.

        Returns
        -------
        tf.Tensor
            layer output
        """
        if training:
            self.iteration += 1

            # Update mixture weights with an expectation step
            reweighted_logits = self.component_logits + mvn_loglikelihood(
                inputs[:, tf.newaxis, :],
                tf.transpose(self.component_means, perm=[1, 0]),
                tf.transpose(self.component_vars, perm=[1, 0]),
            )
            reweighted_p = tf.math.softmax(reweighted_logits)
            batch_logits = tf.reduce_sum(reweighted_p, axis=0) / tf.reduce_sum(reweighted_p)

            logits_running_estimate = (
                (self.iteration - 1) * self.component_logits + batch_logits
            ) / self.iteration
            self.component_logits.assign(logits_running_estimate)

            # Update means and variances with a maximization step
            input_components = tf.where(
                tf.math.is_nan(inputs[..., tf.newaxis]),
                self.component_means,
                inputs[..., tf.newaxis],
            )
            weighted_inputs = tf.einsum("nc,npc->npc", reweighted_p, input_components)
            weight_values = tf.einsum("nc,np->npc", reweighted_p, tf.ones_like(inputs))
            total_weights = tf.reduce_sum(weight_values, axis=0)
            batch_means = tf.reduce_sum(weighted_inputs, axis=0) / total_weights

            means_running_estimate = (
                (self.iteration - 1) * self.component_means + batch_means
            ) / self.iteration
            self.component_means.assign(means_running_estimate)

            weighted_deviations = reweighted_p[:, tf.newaxis, :] * tf.pow(
                input_components - self.component_means, 2
            )

            component_vars = tf.reduce_sum(weighted_deviations, axis=0) / total_weights

            missing_weights = tf.where(
                tf.math.is_nan(inputs[..., tf.newaxis]),
                weight_values,
                tf.constant(0.0, dtype=tf.float32),
            )
            weighted_var_missing = (
                tf.reduce_sum(missing_weights * self.component_vars, axis=0) / total_weights
            )
            batch_vars = component_vars + weighted_var_missing + self.gamma

            vars_running_estimate = (
                (self.iteration - 1) * self.component_vars + batch_vars
            ) / self.iteration
            self.component_vars.assign(vars_running_estimate)

        # now make the forward pass to make the prediction (and gradient descent during training)
        reweighted_logits = self.component_logits + mvn_loglikelihood(
            inputs[:, tf.newaxis, :],
            tf.transpose(self.component_means, perm=[1, 0]),
            tf.transpose(self.component_vars, perm=[1, 0]),
        )

        reweighted_p = tf.math.softmax(reweighted_logits)
        means = tf.where(
            tf.math.is_nan(inputs[..., tf.newaxis]),
            self.component_means,
            inputs[..., tf.newaxis],
        )
        variances = tf.where(
            tf.math.is_nan(inputs[..., tf.newaxis]),
            self.component_vars,
            tf.constant(0.0, dtype=tf.float32),
        )

        layer_mean = tf.einsum("npc,pu->cnu", means, self.kernel) + self.bias
        layer_var = tf.einsum("npc,pu->cnu", variances, tf.math.pow(self.kernel, 2))
        layer_activation = expected_relu(layer_mean, layer_var)
        return tf.einsum("nc,cnu->nu", reweighted_p, layer_activation)


@tf.function
def shash_loss(y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
    """
    Compute the negative log-likelihood of the SHASH distribution.

    Parameters
    ----------
    y_true: tf.Tensor
        Observation tensor
    y_pred: tf.Tensor
        Predictions for each SHASH parameter (four columns)

    Returns
    -------
    tf.Tensor
        SHASH negative log-likelihood
    """
    center = y_pred[:, 0]
    spread = y_pred[:, 1] + 1e-3
    skew = y_pred[:, 2]
    tail = y_pred[:, 3] + 1e-3
    z = (y_true[:, 0] - center) / (spread * tail)
    sz = tf.math.sinh(tail * tf.math.asinh(z) - skew)
    lcz = tf.math.log(1.0 + sz**2.0) / 2.0
    llk = (
        lcz
        - (sz**2.0) / 2.0
        - tf.math.log(spread)
        - tf.math.log(2.0 * TFPI) / 2.0
        - tf.math.log(1.0 + z**2.0) / 2.0
    )

    return -llk


class MissingnessNeuralNet:
    """Class for neural network that can handle missing values."""

    def __init__(
        self,
        units_list: List[int] = [100, 50],
        n_components: int = 3,
        dropout_rate: float = 0.3,
        bayesian: bool = False,
        batch_size: int = 1000,
        n_epochs: int = 1000,
        verbose: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        units_list: int
            number of units in each hidden layer (excluding input and output layers)
        n_components: int
            number of components of the gaussian mixture distribution of the DenseMissing layer
        dropout_rate: float
            dropout rate for regularization
        bayesian: bool
            if True, layer weights are probabilistic and sampling is used for interval predictions
        batch_size: int
            minibatch size for gradient descent
        n_epochs: int
            number of epochs for training the model
        verbose: bool
            controls the verbosity during training (passed to tf.model.fit())
        """
        self.units_list = units_list
        self.n_components = n_components
        self.dropout_rate = dropout_rate
        self.bayesian = True if bayesian else None  # pass what Dropout expects
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.verbose = verbose

    def build_model(self, X: np.ndarray) -> None:
        """
        Build the model.

        Parameters
        ----------
        X: np.ndarray
            Design matrix containing features
        """
        missingness_present = np.isnan(X).any()
        inputs = tf.keras.layers.Input(shape=(X.shape[-1],))

        # first layer must be DenseMissing to accomodate missing values
        if missingness_present:
            next_output = DenseMissing(units=self.units_list[0], n_components=self.n_components)(
                inputs
            )
        else:
            next_output = tf.keras.layers.Dense(units=self.units_list[0], activation=tf.nn.relu)(
                inputs
            )
        next_output = tf.keras.layers.BatchNormalization()(next_output)
        next_output = tf.keras.layers.Dropout(self.dropout_rate)(
            next_output, training=self.bayesian
        )

        if len(self.units_list) > 1:
            for units in self.units_list[1:]:
                # these can be standard layers, there are no missing values from DenseMissing
                next_output = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)(next_output)
                next_output = tf.keras.layers.BatchNormalization()(next_output)
                next_output = tf.keras.layers.Dropout(self.dropout_rate)(
                    next_output, training=self.bayesian
                )

        # the final layers for the mean, scale, skew and tail parameters
        pred_mean = tf.keras.layers.Dense(units=1, activation="linear")(next_output)

        # scale is definitely less than 0.2 (raw sd is 0.1)
        pred_scale = tf.keras.layers.Dense(
            units=1, activation=lambda x: tf.math.sigmoid(x) * 0.2, kernel_initializer="zeros"
        )(next_output)

        # skew is limited between -0.5 and 0.5
        pred_skew = tf.keras.layers.Dense(
            units=1, activation=lambda x: tf.math.sigmoid(x) - 0.5, kernel_initializer="zeros"
        )(next_output)

        # tail is limited between 0.95 and 1.05
        pred_tail = tf.keras.layers.Dense(
            units=1,
            activation=lambda x: tf.math.sigmoid(x) * 0.1 + 0.95,
            kernel_initializer="zeros",
        )(next_output)
        output_layer = tf.keras.layers.Concatenate()([pred_mean, pred_scale, pred_skew, pred_tail])

        self.model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=tf.optimizers.schedules.CosineDecay(
                    initial_learning_rate=0.0001,
                    warmup_target=0.0003,
                    warmup_steps=int(X.shape[0] / self.batch_size * self.n_epochs / 3),
                    decay_steps=int(2 * X.shape[0] / self.batch_size * self.n_epochs / 3),
                    alpha=0.0,
                )
            ),
            loss=shash_loss,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Fit the model.

        Parameters
        ----------
        X: np.ndarray
            Design matrix containing features
        y: np.ndarray
            Array of response values
        """
        self.build_model(X)
        x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=1, test_size=0.3)
        self.model.fit(
            x=tf.convert_to_tensor(x_train),
            y=tf.convert_to_tensor(y_train),
            validation_data=(x_val, y_val),
            batch_size=self.batch_size,
            epochs=self.n_epochs,
            verbose=self.verbose,
            shuffle=True,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    patience=50, restore_best_weights=True, min_delta=0.0001
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
        )

    def predict_intervals(
        self, X: np.ndarray, alpha: float = 0.9, n_samples: int = 1000
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the alpha * 100% interval for birthweight.

        Parameters
        ----------
        X: np.ndarray
            the design matrix used by self.fit(X, y)
        alpha: float
            the significance level of the predicted intervals
        n_samples: int
            if Bayesian, this many samples are drawn from the posterior distribution

        Returns
        -------
        Tuple
            the arrays corresponding to the lower and upper bounds, respectively
        """
        if self.bayesian:
            predicted_samples = np.random.randn(n_samples, X.shape[0])
            for i, sample in tqdm(enumerate(predicted_samples)):
                center, spread, skew, tail = self.model(X).numpy().T
                spread = spread + 1e-3
                tail = tail + 1e-3
                prediction = center + (tail * spread) * (
                    np.sinh((1 / tail) * np.arcsinh(sample) + skew / tail)
                )
                predicted_samples[i] = prediction

            lower, upper = np.apply_along_axis(
                func1d=lambda x: compute_highest_density_interval(
                    np_softplus(x) * SOFTPLUS_SCALE, alpha=alpha
                ),
                axis=0,
                arr=predicted_samples,
            )
            return lower, upper

        else:
            center, spread, skew, tail = self.model.predict(X).T
            spread = spread + 1e-3
            tail = tail + 1e-3

            buffers = np.linspace(-((1 - alpha) / 2) + 1e-6, ((1 - alpha) / 2) - 1e-6, 20)
            widths = []
            for buffer in buffers:
                upper_edge = center + (tail * spread) * np.sinh(
                    (1 / tail) * np.arcsinh(st.norm.ppf(alpha + (1 - alpha) / 2) - buffer)
                    + skew / tail
                )
                lower_edge = center + (tail * spread) * np.sinh(
                    (1 / tail) * np.arcsinh(st.norm.ppf((1 - alpha) / 2) - buffer) + skew / tail
                )
                widths.append((np_softplus(upper_edge) - np_softplus(lower_edge)).reshape((-1, 1)))
            opt_buffers = buffers[np.hstack(widths).argmin(axis=1)]

            lower = center + (tail * spread) * np.sinh(
                (1 / tail) * np.arcsinh(st.norm.ppf((1 - alpha) / 2) - opt_buffers) + skew / tail
            )
            upper = center + (tail * spread) * np.sinh(
                (1 / tail) * np.arcsinh(st.norm.ppf(alpha + (1 - alpha) / 2) - opt_buffers)
                + skew / tail
            )
            return np_softplus(lower) * SOFTPLUS_SCALE, np_softplus(upper) * SOFTPLUS_SCALE
