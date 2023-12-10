"""
DenseMissing layer that outputs an expected RELU whenever input features are missing.

Adapted from https://github.com/lstruski/Processing-of-missing-data-by-neural-networks/
    blob/master/mlp.py
Paper: https://arxiv.org/pdf/1805.07405.pdf
EM algorithm: https://arxiv.org/pdf/1209.0521.pdf and https://arxiv.org/pdf/1902.03335.pdf
"""

import tensorflow as tf

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
        """ """
