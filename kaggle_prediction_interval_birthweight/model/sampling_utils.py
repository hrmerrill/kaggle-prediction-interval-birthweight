from typing import Tuple

import numpy as np


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
