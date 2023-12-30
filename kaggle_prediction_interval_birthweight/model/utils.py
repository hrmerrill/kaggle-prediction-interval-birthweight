from typing import Optional, Tuple

import numpy as np
import tensorflow as tf


def np_softplus(x: np.ndarray) -> np.ndarray:
    """Helper function for computing softplus without overflow."""
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0)


def np_softplus_inv(x: np.ndarray) -> np.ndarray:
    """Compute the inverse softplus without overflow."""
    return np.log1p(-np.exp(-np.abs(x))) + np.maximum(x, 0)


def compute_highest_density_interval(samples: np.ndarray, alpha: float = 0.9) -> Tuple[float]:
    """
    Get the compact HDI region from a set of samples.

    Adapted from https://bebi103.caltech.edu.s3-website-us-east-1.amazonaws.com/2015/tutorials/
        l06_credible_regions.html

    Parameters
    ----------
    samples: np.ndarray
        array of samples used to compute the HDI
    alpha: float
        the confidence level

    Returns
    -------
    Tuple
        the lower and upper bounds of the HDI
    """
    samples = np.sort(samples.copy())
    n_samples_in_hdi = np.floor(alpha * len(samples)).astype(int)

    # get the widths of candidate intervals that contain n_samples_in_hdi samples
    widths = samples[n_samples_in_hdi:] - samples[: len(samples) - n_samples_in_hdi]
    smallest_interval = np.argmin(widths)

    return samples[smallest_interval], samples[smallest_interval + n_samples_in_hdi]


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

    def __init__(self, units: int, n_components: int = 3, gamma: float = 1e-5, **kwargs) -> None:
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
        Predictions for each SHASH parameter (three or four columns)

    Returns
    -------
    tf.Tensor
        SHASH negative log-likelihood
    """
    center = y_pred[:, 0]
    spread = y_pred[:, 1] + 1e-3
    skew = y_pred[:, 2]
    if y_pred.shape[-1] == 4:
        tail = y_pred[:, 3]
    else:
        tail = 1.0

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


@tf.function
def eim_loss(
    y_true: tf.Tensor, y_pred: tf.Tensor, alpha: float = 0.9, delta: float = 0.03
) -> tf.Tensor:
    """
    Compute the Expanded Interval Minimization loss.

    https://arxiv.org/pdf/1806.11222.pdf Eqns 11-14

    If y_pred has three columns, the third is assumed to be the predicted median.

    Parameters
    ----------
    y_true: tf.Tensor
        Observation tensor
    y_pred: tf.Tensor
        Interval predictions (2 columns, first column >= second)
    alpha: float
        Desired coverage percentage of interval predictions
    delta: float
        Eqn 14. Expansion factor is an average of the [alpha - delta, alpha + delta] quantiles


    Returns
    -------
    tf.Tensor
        EIM loss
    """
    widths = y_pred[:, 1] - y_pred[:, 0]
    expansion_vals = tf.sort(
        tf.math.abs((y_pred[:, 1] + y_pred[:, 0] - 2.0 * tf.squeeze(y_true)) / widths),
        direction="ASCENDING",
    )
    lower_q_index = tf.round(tf.cast(tf.shape(y_pred)[0] - 1, dtype=tf.float32) * (alpha - delta))
    upper_q_index = tf.round(tf.cast(tf.shape(y_pred)[0] - 1, dtype=tf.float32) * (alpha + delta))
    lower_k = expansion_vals[tf.cast(lower_q_index, dtype=tf.int32)]
    upper_k = expansion_vals[tf.cast(upper_q_index, dtype=tf.int32)]
    keepers = tf.cast(
        tf.where(expansion_vals > upper_k, 0, tf.where(expansion_vals < lower_k, 0, 1)),
        dtype=tf.float32,
    )
    k_b = tf.reduce_sum(keepers * expansion_vals) / tf.reduce_sum(keepers)
    loss = k_b * widths

    if y_pred.shape[-1] == 3:
        loss = loss + tf.reduce_mean(tf.math.abs(tf.squeeze(y_true) - y_pred[:, 2]))
    return loss
