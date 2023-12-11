"""
DenseMissing layer that outputs an expected RELU whenever input features are missing.

Adapted from https://github.com/lstruski/Processing-of-missing-data-by-neural-networks/
    blob/master/mlp.py
Paper: https://arxiv.org/pdf/1805.07405.pdf
EM algorithm: https://arxiv.org/pdf/1209.0521.pdf and https://arxiv.org/pdf/1902.03335.pdf
"""

from typing import Optional, Tuple

import tensorflow as tf
import tensorflow_probability as tfp

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
            tf.exp(tf.divide(-tf.square(w), 2.0)),
            tf.math.sqrt(tf.constant(2 * TFPI, dtype=tf.float32)),
        )
        + tf.multiply(
            tf.divide(w, 2.0),
            1 + tf.math.erf(tf.divide(w, tf.math.sqrt(tf.constant(2, dtype=tf.float32)))),
        )
    )
    expected_relu_values = tf.where(where_missing, expected_relu_values, (mu + tf.abs(mu)) / 2.0)
    return expected_relu_values


class DenseMissing(tf.keras.layers.Layer):
    """Extend the Dense layer to handle missing data."""

    def __init__(self, n_units: int, n_components: int = 3, gamma: float = 1e-6, **kwargs) -> None:
        """
        Parameters
        ----------
        n_units: int
            number of units in the dense layer
        n_components: int
            number of components of the gaussian mixture distribution
        gamma: float
            regularization value for variance estimates
        kwargs: dict
            other arguments passed to tf.keras.layers.Layer
        """
        super(DenseMissing, self).__init__(**kwargs)
        self.n_units = n_units
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
        return input_shape[:-1] + (self.n_units)

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
            tf.zeros((input_shape[:-1], self.n_components), dtype=tf.float32),
            trainable=False,
            name="component_means",
        )
        self.component_vars = tf.Variable(
            tf.ones((input_shape[:-1], self.n_components), dtype=tf.float32),
            trainable=False,
            name="component_vars",
        )
        self.component_logits = tf.Variable(
            tf.ones((self.n_components,), dtype=tf.float32),
            trainable=False,
            name="component_logits",
        )
        self.kernel = self.add_weight(
            shape=(input_shape[-1], self.n_units),
            trainable=True,
            name="kernel",
        )
        self.bias = self.add_weight(
            shape=(self.n_units,),
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


class BayesianNeuralNet:
    """Class for bayesian neural network that can handle missing values."""

    def __init__(self, n_units: int, n_layers: int = 1) -> None:
        """
        Parameters
        ----------
        n_units: int
            number of units in each hidden layer
        n_layers: int
            number of hidden layers (does not include input features or final output layer)
        """
        self.n_units = n_units
        self.n_layers = n_layers

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
        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Input(shape=(X.shape[1],)))

        # first layer must be DenseMissing to accomodate missing values
        self.model.add(DenseMissing(n_units=self.n_units))
        if self.n_layers > 1:
            self.model.add(tfp.layers.DenseFlipout(units=self.n_units, activation="relu"))

        # the last layer is the probabilistic DenseFlipout layer, with two outputs-- one
        # each for the mean and the standard deviation of a normal distribution
        self.model.add(tfp.layers.DenseFlipout(units=2), activation="linear")
        self.model.add(
            tfp.layers.DistributionLambda(
                lambda x: tfp.distributions.Normal(
                    loc=x[:, 0], scale=1e-3 + tf.math.softplus(x[:, 1]), name="output_layer"
                )
            )
        )

        self.model.compile(
            optimizer=tf.optimizers.Adam(
                learning_rate=tf.optimizers.schedules.PolynomialDecay(
                    initial_learning_rate=0.01,
                    decay_steps=10000,
                    end_learning_rate=0.00001,
                )
            ),
            loss=lambda y, p_y: -p_y.log_prob(y),
        )
        self.model.fit(x=X, y=y, batch_size=100, epochs=100)
