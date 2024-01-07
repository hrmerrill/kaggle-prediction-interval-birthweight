from typing import List, Optional, Tuple

import numpy as np
import scipy.stats as st
import tensorflow as tf
from mapie.regression import MapieQuantileRegressor
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from kaggle_prediction_interval_birthweight.data.constants import BIN_LABELS, SOFTPLUS_SCALE
from kaggle_prediction_interval_birthweight.utils.utils import (
    DenseMissing,
    compute_highest_density_interval,
    eim_loss,
    np_softplus,
    shash_loss,
)


class MissingnessNeuralNetRegressor:
    """Class for neural network regressor that can handle missing values."""

    def __init__(
        self,
        units_list: List[int] = [235, 235],
        n_components: int = 3,
        dropout_rate: float = 0.8,
        bayesian: bool = False,
        fit_tail: bool = True,
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
            if True, layer weights are probabilistic and sampling is used for interval predictions.
            This is expensive
        fit_tail: bool
            if True, the SHASH tail parameter is also modeled as a function of the inputs
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
        self.fit_tail = fit_tail
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

        # first & skip layers must be DenseMissing to accomodate missing values
        if missingness_present:
            next_output = DenseMissing(units=self.units_list[0], n_components=self.n_components)(
                inputs
            )
            skip_output = DenseMissing(units=self.units_list[-1], n_components=self.n_components)(
                inputs
            )
        else:
            next_output = tf.keras.layers.Dense(units=self.units_list[0], activation=tf.nn.relu)(
                inputs
            )
            skip_output = tf.keras.layers.Dense(units=self.units_list[-1], activation=tf.nn.relu)(
                inputs
            )
        next_output = tf.keras.layers.BatchNormalization(scale=False)(next_output)
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
        next_output = next_output + skip_output

        # the final layers for the mean, scale, skew and tail parameters
        pred_mean = tf.keras.layers.Dense(units=1, activation="linear")(next_output)

        # scale is definitely less than 1 (raw sd is about 0.6)
        pred_scale = tf.keras.layers.Dense(
            units=1, activation=lambda x: tf.math.sigmoid(x), kernel_initializer="zeros"
        )(next_output)

        # skew is limited between -0.5 and 0.5
        pred_skew = tf.keras.layers.Dense(
            units=1, activation=lambda x: tf.math.sigmoid(x) - 0.5, kernel_initializer="zeros"
        )(next_output)

        if self.fit_tail:
            # tail is really unstable, so it is limited to values between 0.95 and 1.05
            pred_tail = tf.keras.layers.Dense(
                units=1,
                activation=lambda x: tf.math.sigmoid(x) * 0.1 + 0.95,
                kernel_initializer="zeros",
            )(next_output)
            output_layer = tf.keras.layers.Concatenate()(
                [pred_mean, pred_scale, pred_skew, pred_tail]
            )
        else:
            output_layer = tf.keras.layers.Concatenate()([pred_mean, pred_scale, pred_skew])

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
                    patience=10,
                    restore_best_weights=True,
                    min_delta=0.0001,
                    start_from_epoch=50,
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
            # create white noise from which to draw SHASH samples
            predicted_samples = np.random.randn(n_samples, X.shape[0])
            for i, sample in tqdm(
                enumerate(predicted_samples), total=n_samples, desc="Sampling from model"
            ):
                # make a probabilistic prediction (dropout is still turned on)
                model_outputs = self.model(X).numpy()
                if model_outputs.shape[-1] == 4:
                    center, spread, skew, tail = model_outputs.T
                else:
                    center, spread, skew = model_outputs.T
                    tail = 1.0
                spread = spread + 1e-3

                # now draw the SHASH variable from that prediction
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
            model_outputs = self.model(X).numpy()
            if model_outputs.shape[-1] == 4:
                center, spread, skew, tail = model_outputs.T
            else:
                center, spread, skew = model_outputs.T
                tail = 1.0

            # Here instead of using the 5 and 95% quantiles as endpoints, we can search for the
            # 95% interval that is smallest (e.g., maybe [2%, 92%] is smaller than [5%, 95%])
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

            # for each row, optimal buffer percentage is the one that produces the shortest width
            opt_buffers = buffers[np.hstack(widths).argmin(axis=1)]

            lower = center + (tail * spread) * np.sinh(
                (1 / tail) * np.arcsinh(st.norm.ppf((1 - alpha) / 2) - opt_buffers) + skew / tail
            )
            upper = center + (tail * spread) * np.sinh(
                (1 / tail) * np.arcsinh(st.norm.ppf(alpha + (1 - alpha) / 2) - opt_buffers)
                + skew / tail
            )
            return np_softplus(lower) * SOFTPLUS_SCALE, np_softplus(upper) * SOFTPLUS_SCALE


class MapieHelper:
    """Helper class for using Mapie for calibration of predicted intervals."""

    def __init__(
        self,
        model: tf.keras.models.Model,
        which_type: str,
        which_pred: int,
        bin_values: Optional[np.ndarray] = None,
        alpha: float = 0.9,
    ) -> None:
        """
        Parameters
        ----------
        model: tf.keras.models.Model
            The trained tensorflow model
        which_type: str
            One of "classifier" or "EIM"
        which_pred: int
            one of 0 (lower bound), 1 (upper bound), or 2 (median)
        bin_values: np.ndarray
            the numeric values corresponding to each category (e.g., bin midpoint). Required if
            self.which_type == "classifier".
        alpha: float
            the significance level of the predicted intervals. Relevant only for
            self.which_type == "classifier".
        """
        self.model = model
        self.which_type = which_type
        self.which_pred = which_pred
        self.bin_values = bin_values
        self.alpha = alpha
        self.__sklearn_is_fitted__ = lambda: True

    def fit(self, X: tf.Tensor, y: tf.Tensor) -> None:
        """Doesn't do anything, just required by Mapie."""

    def predict(self, X: tf.Tensor) -> tf.Tensor:
        """
        Predict from the model.

        Parameters
        ----------
        X: tf.Tensor
            the input tensor

        Returns
        -------
        tf.Tensor
            the lower bounds, upper bounds, or medians, if self.which_pred is 0, 1, or 2,
            respectively.
        """
        # for the classifier, we'll search for the narrowest interval that contains 95% mass
        if self.which_type == "classifier":

            def get_smallest_interval_rowwise(row, alpha=self.alpha, bin_values=self.bin_values):
                """helper function to apply row-wise to get smallest the interval."""
                upper_indices, widths = [], []
                for lower_index in range(len(row)):
                    cumulative_probs = row[lower_index:].cumsum()
                    if cumulative_probs[-1] <= alpha:
                        break
                    else:
                        upper_index = np.where(cumulative_probs >= alpha)[0].min() + lower_index - 1
                        widths.append(bin_values[upper_index] - bin_values[lower_index])
                        upper_indices.append(upper_index)
                return np.array(widths).argmin(), upper_indices[np.array(widths).argmin()]

            probs = self.model.predict(X)
            lower_inds, upper_inds = [], []
            for probs_row in tqdm(
                probs, total=probs.shape[0], desc="Searching for smallest interval"
            ):
                lower_ind, upper_ind = get_smallest_interval_rowwise(probs_row)
                lower_inds.append(lower_ind)
                upper_inds.append(upper_ind)
            lower, upper = self.bin_values[lower_inds], self.bin_values[upper_inds]
            median = self.bin_values[np.abs(probs.cumsum(axis=1) - 0.5).argmin(axis=1)]
            outputs = np.hstack(
                [lower.reshape((-1, 1)), upper.reshape((-1, 1)), median.reshape((-1, 1))]
            )

        # the EIM model predicts intervals directly.
        elif self.which_type == "EIM":
            outputs = self.model.predict(X)

        return outputs[:, self.which_pred]


class MissingnessNeuralNetClassifier:
    """Class for neural network classifier that can handle missing values."""

    def __init__(
        self,
        bin_values: np.ndarray = BIN_LABELS,
        units_list: List[int] = [5, 5, 5],
        n_components: int = 3,
        dropout_rate: float = 0.4,
        batch_size: int = 1000,
        alpha: float = 0.9,
        n_epochs: int = 1000,
        verbose: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        bin_values: np.ndarray
            the numeric values corresponding to each category (e.g., bin midpoint)
        units_list: int
            number of units in each hidden layer (excluding input and output layers)
        n_components: int
            number of components of the gaussian mixture distribution of the DenseMissing layer
        dropout_rate: float
            dropout rate for regularization
        batch_size: int
            minibatch size for gradient descent
        alpha: float
            Desired coverage level of predicted intervals
        n_epochs: int
            number of epochs for training the model
        verbose: bool
            controls the verbosity during training (passed to tf.model.fit())
        """
        self.bin_values = bin_values
        self.units_list = units_list
        self.n_components = n_components
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.alpha = alpha
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

        # first and skip layers must be DenseMissing to accomodate missing values
        if missingness_present:
            next_output = DenseMissing(units=self.units_list[0], n_components=self.n_components)(
                inputs
            )
            skip_output = DenseMissing(units=self.units_list[-1], n_components=self.n_components)(
                inputs
            )
        else:
            next_output = tf.keras.layers.Dense(units=self.units_list[0], activation=tf.nn.relu)(
                inputs
            )
            skip_output = tf.keras.layers.Dense(units=self.units_list[-1], activation=tf.nn.relu)(
                inputs
            )
        next_output = tf.keras.layers.BatchNormalization(scale=False)(next_output)
        next_output = tf.keras.layers.Dropout(self.dropout_rate)(next_output)

        if len(self.units_list) > 1:
            for units in self.units_list[1:]:
                # these can be standard layers, there are no missing values from DenseMissing
                next_output = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)(next_output)
                next_output = tf.keras.layers.BatchNormalization()(next_output)
                next_output = tf.keras.layers.Dropout(self.dropout_rate)(next_output)
        next_output = next_output + skip_output

        # the final layers for the predictions
        probs = tf.keras.layers.Dense(units=len(self.bin_values), activation="softmax")(next_output)
        self.model = tf.keras.models.Model(inputs=inputs, outputs=probs)
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
            loss="sparse_categorical_crossentropy",
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
                    patience=10,
                    restore_best_weights=True,
                    min_delta=0.0001,
                    start_from_epoch=50,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
        )
        print("Calibrating with Mapie.")
        self.calibrator = MapieQuantileRegressor(
            [
                MapieHelper(
                    model=self.model,
                    which_type="classifier",
                    which_pred=i,
                    alpha=self.alpha,
                    bin_values=self.bin_values,
                )
                for i in range(3)
            ],
            alpha=1 - self.alpha,
            cv="prefit",
        )
        self.calibrator.fit(x_val, self.bin_values[y_val.squeeze().astype(int)])

    def predict_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the alpha * 100% interval for birthweight.

        Parameters
        ----------
        X: np.ndarray
            the design matrix used by self.fit(X, y)

        Returns
        -------
        Tuple
            the arrays corresponding to the lower and upper bounds, respectively
        """
        _, intervals = self.calibrator.predict(X)
        lower, upper = intervals.squeeze().T
        return lower.squeeze(), upper.squeeze()


class MissingnessNeuralNetEIM:
    """Class for neural network interval predictor that can handle missing values."""

    def __init__(
        self,
        units_list: List[int] = [275, 275, 275, 275],
        n_components: int = 3,
        dropout_rate: float = 0.4,
        batch_size: int = 1000,
        alpha: float = 0.9,
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
        batch_size: int
            minibatch size for gradient descent
        alpha: float
            Desired coverage level of predicted intervals
        n_epochs: int
            number of epochs for training the model
        verbose: bool
            controls the verbosity during training (passed to tf.model.fit())
        """
        self.units_list = units_list
        self.n_components = n_components
        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.alpha = alpha
        self.n_epochs = n_epochs
        self.verbose = verbose

    def build_model(self, X: np.ndarray) -> tf.keras.models.Model:
        """
        Build the model.

        Parameters
        ----------
        X: np.ndarray
            Design matrix containing features

        Returns
        -------
        tf.keras.models.Model
            the uncompiled tensorflow model.
        """
        missingness_present = np.isnan(X).any()
        inputs = tf.keras.layers.Input(shape=(X.shape[-1],))

        # first and skip layers must be DenseMissing to accomodate missing values
        if missingness_present:
            next_output = DenseMissing(units=self.units_list[0], n_components=self.n_components)(
                inputs
            )
            skip_output = DenseMissing(units=self.units_list[-1], n_components=self.n_components)(
                inputs
            )
        else:
            next_output = tf.keras.layers.Dense(units=self.units_list[0], activation=tf.nn.relu)(
                inputs
            )
            skip_output = tf.keras.layers.Dense(units=self.units_list[-1], activation=tf.nn.relu)(
                inputs
            )
        next_output = tf.keras.layers.BatchNormalization(scale=False)(next_output)
        next_output = tf.keras.layers.Dropout(self.dropout_rate)(next_output)

        if len(self.units_list) > 1:
            for units in self.units_list[1:]:
                # these can be standard layers, there are no missing values from DenseMissing
                next_output = tf.keras.layers.Dense(units=units, activation=tf.nn.relu)(next_output)
                next_output = tf.keras.layers.BatchNormalization()(next_output)
                next_output = tf.keras.layers.Dropout(self.dropout_rate)(next_output)
        next_output = next_output + skip_output

        # the final layers for the lower and upper bounds and median
        pred_lower = tf.keras.layers.Dense(units=1, activation="linear")(next_output)
        pred_median = pred_lower + tf.keras.layers.Dense(units=1, activation=tf.math.softplus)(
            next_output
        )
        pred_upper = pred_median + tf.keras.layers.Dense(units=1, activation=tf.math.softplus)(
            next_output
        )
        output_layer = tf.keras.layers.Concatenate()([pred_lower, pred_upper, pred_median])
        model = tf.keras.models.Model(inputs=inputs, outputs=output_layer)
        return model

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
        x_train, x_val, y_train, y_val = train_test_split(X, y, random_state=1, test_size=0.3)

        # start with a blank model and pretrain it to something more or less useful
        print("Beginning warm-start.")
        blank_model = self.build_model(X)
        lower_boundary = y.mean() - np.quantile(y, 0.05)
        upper_boundary = np.quantile(y, 0.95) - y.mean()
        warmup_loss = (
            lambda y, p_y: (y - lower_boundary - p_y[:, 0]) ** 2
            + (y + upper_boundary - p_y[:, 1]) ** 2
            + tf.math.abs(y - p_y[:, 2])
        )
        blank_model.compile(optimizer=tf.keras.optimizers.legacy.Adam(0.003), loss=warmup_loss)
        blank_model.fit(
            x=tf.convert_to_tensor(x_train),
            y=tf.convert_to_tensor(y_train),
            batch_size=self.batch_size,
            epochs=100,
            shuffle=True,
            verbose=0,
        )

        # use this model as a warm-start for the EIM model
        print("Training warmed-up model.")
        self.model = self.build_model(X)
        self.model.set_weights(blank_model.get_weights())

        def eim_alpha_loss(y: tf.Tensor, p_y: tf.Tensor) -> tf.Tensor:
            return eim_loss(y, p_y, alpha=self.alpha)

        self.model.compile(
            optimizer=tf.keras.optimizers.legacy.Adam(
                learning_rate=tf.optimizers.schedules.CosineDecay(
                    initial_learning_rate=0.0001,
                    warmup_target=0.0003,
                    warmup_steps=int(x_train.shape[0] / self.batch_size * self.n_epochs / 3),
                    decay_steps=int(2 * x_train.shape[0] / self.batch_size * self.n_epochs / 3),
                    alpha=0.0,
                )
            ),
            loss=eim_alpha_loss,
        )

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
                    patience=10,
                    restore_best_weights=True,
                    min_delta=0.0001,
                    start_from_epoch=50,
                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ],
        )
        print("Calibrating with Mapie.")
        self.calibrator = MapieQuantileRegressor(
            [
                MapieHelper(model=self.model, which_type="EIM", which_pred=i, alpha=self.alpha)
                for i in range(3)
            ],
            alpha=1 - self.alpha,
            cv="prefit",
        )
        self.calibrator.fit(x_val, y_val.squeeze())

    def predict_intervals(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the alpha * 100% interval for birthweight.

        Parameters
        ----------
        X: np.ndarray
            the design matrix used by self.fit(X, y)

        Returns
        -------
        Tuple
            the arrays corresponding to the lower and upper bounds, respectively
        """
        _, intervals = self.calibrator.predict(X)
        lower, upper = intervals.squeeze().T
        lower = np_softplus(lower) * SOFTPLUS_SCALE
        upper = np_softplus(upper) * SOFTPLUS_SCALE
        return lower.squeeze(), upper.squeeze()
