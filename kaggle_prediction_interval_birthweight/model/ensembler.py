from typing import Tuple

import numpy as np
import pandas as pd
from mapie.regression import MapieQuantileRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import d2_pinball_score, make_scorer
from sklearn.model_selection import GridSearchCV, train_test_split
from tqdm import tqdm

from kaggle_prediction_interval_birthweight.data.constants import SOFTPLUS_SCALE
from kaggle_prediction_interval_birthweight.data.data_processing import DataProcessor
from kaggle_prediction_interval_birthweight.model.hist_gradient_boosting import HistBoostRegressor
from kaggle_prediction_interval_birthweight.model.linear_regression import RidgeRegressor
from kaggle_prediction_interval_birthweight.model.neural_network import (
    MissingnessNeuralNetClassifier,
    MissingnessNeuralNetEIM,
    MissingnessNeuralNetRegressor,
)
from kaggle_prediction_interval_birthweight.model.wildwood import WildWoodRegressor
from kaggle_prediction_interval_birthweight.utils.utils import (
    compute_highest_density_interval,
    np_softplus,
    np_softplus_inv,
)


class BaseEnsembler:
    """
    Base ensemble model class with common methods.
    """

    def __init__(self, n_folds: int = 3, alpha: float = 0.9) -> None:
        """
        Parameters
        ----------
        n_folds: int
            number of folds to use for held-out training
        alpha: float
            significance level for prediction intervals
        """
        self.alpha = alpha
        self.n_folds = n_folds

        ridge_predictions = ["lower_ridge", "mean_ridge", "upper_ridge"]
        boost_predictions = ["lower_boost", "upper_boost"]
        wildwood_predictions = ["lower_wildwood", "mean_wildwood", "upper_wildwood"]
        nn_predictions = ["center_nn", "scale_nn", "skew_nn", "tail_nn", "lower_nn", "upper_nn"]
        nnc_predictions = ["lower_nnc", "mode_nnc", "upper_nnc"]
        eim_predictions = ["lower_eim", "median_eim", "upper_eim"]
        self.upstream_predictions = (
            ridge_predictions
            + boost_predictions
            + wildwood_predictions
            + nn_predictions
            + nnc_predictions
            + eim_predictions
        )

    def prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit upstream models and prepare data for the ensembler.

        Parameters
        ----------
        df: pd.DataFrame
            The input data

        Returns
        -------
        pd.DataFrame
            data frame with predictions from upstream models
        """
        df = df.copy()

        # save and initialize data processors, so standardization parameters are available later
        self.ridge_data_processor = DataProcessor(model_type="RidgeRegressor")
        self.boost_data_processor = DataProcessor(model_type="HistBoostRegressor")
        self.wildwood_data_processor = DataProcessor(model_type="WildWoodRegressor")
        self.nn_data_processor = DataProcessor(model_type="MissingnessNeuralNetRegressor")
        self.nnc_data_processor = DataProcessor(model_type="MissingnessNeuralNetClassifier")
        self.eim_data_processor = DataProcessor(model_type="MissingnessNeuralNetEIM")
        _ = self.ridge_data_processor(df)
        _ = self.boost_data_processor(df)
        _ = self.wildwood_data_processor(df)
        _ = self.nn_data_processor(df)
        _ = self.nnc_data_processor(df)
        _ = self.eim_data_processor(df)

        # initialize all models
        self.histboosters = [
            HistBoostRegressor(
                alpha=self.alpha,
                categorical_feature_mask=self.boost_data_processor.categorical_features,
            )
            for _ in range(self.n_folds)
        ]
        self.wildwood_regressors = [
            WildWoodRegressor(
                alpha=self.alpha,
                categorical_feature_mask=self.wildwood_data_processor.categorical_features,
            )
            for _ in range(self.n_folds)
        ]
        self.ridge_regressors = [RidgeRegressor() for _ in range(self.n_folds)]
        self.nn_regressors = [
            MissingnessNeuralNetRegressor(bayesian=False, fit_tail=True)
            for _ in range(self.n_folds)
        ]
        self.nn_classifiers = [MissingnessNeuralNetClassifier() for _ in range(self.n_folds)]
        self.nn_eims = [MissingnessNeuralNetEIM() for _ in range(self.n_folds)]

        # create some information for holding new data in the data frame
        df["fold"] = np.random.choice(self.n_folds, df.shape[0])
        for feature in self.upstream_predictions:
            df[feature] = np.nan

        # loop across splits, fit and predict from upstream models
        for k in range(self.n_folds):
            print(f"Ensembler fold {k+1} of {self.n_folds} begins.")

            # split into train and test sets
            df_train = df.query("fold != @k")
            df_test = df.query("fold == @k")

            # prepare the data for each model
            xr_train, yr_train = self.ridge_data_processor(df_train)
            xr_test = self.ridge_data_processor(df_test.drop("DBWT", axis=1))

            xb_train, yb_train = self.boost_data_processor(df_train)
            xb_test = self.boost_data_processor(df_test.drop("DBWT", axis=1))

            xw_train, yw_train = self.wildwood_data_processor(df_train)
            xw_test = self.wildwood_data_processor(df_test.drop("DBWT", axis=1))

            xn_train, yn_train = self.nn_data_processor(df_train)
            xn_test = self.nn_data_processor(df_test.drop("DBWT", axis=1))

            xnc_train, ync_train = self.nnc_data_processor(df_train)
            xnc_test = self.nnc_data_processor(df_test.drop("DBWT", axis=1))

            xeim_train, yeim_train = self.eim_data_processor(df_train)
            xeim_test = self.eim_data_processor(df_test.drop("DBWT", axis=1))

            print("Training the ridge regression model.")
            self.ridge_regressors[k].fit(xr_train, yr_train)
            ridge_mean = self.ridge_regressors[k].predict(xr_test)
            ridge_lower, ridge_upper = self.ridge_regressors[k].predict_intervals(
                xr_test, alpha=self.alpha
            )
            df.loc[df["fold"] == k, "lower_ridge"] = ridge_lower.squeeze() / SOFTPLUS_SCALE
            df.loc[df["fold"] == k, "mean_ridge"] = ridge_mean.squeeze() / SOFTPLUS_SCALE
            df.loc[df["fold"] == k, "upper_ridge"] = ridge_upper.squeeze() / SOFTPLUS_SCALE

            print("Training the histogram boosting model.")
            self.histboosters[k].fit(xb_train, yb_train)
            boost_lower, boost_upper = self.histboosters[k].predict_intervals(xb_test)
            df.loc[df["fold"] == k, "lower_boost"] = boost_lower.squeeze() / SOFTPLUS_SCALE
            df.loc[df["fold"] == k, "upper_boost"] = boost_upper.squeeze() / SOFTPLUS_SCALE

            print("Training the wildwood model.")
            self.wildwood_regressors[k].fit(xw_train, yw_train)
            wildwood_mean = self.wildwood_regressors[k].regressor.predict(xw_test)
            wildwood_lower, wildwood_upper = self.wildwood_regressors[k].predict_intervals(xw_test)
            df.loc[df["fold"] == k, "lower_wildwood"] = wildwood_lower.squeeze() / SOFTPLUS_SCALE
            df.loc[df["fold"] == k, "mean_wildwood"] = wildwood_mean.squeeze() / SOFTPLUS_SCALE
            df.loc[df["fold"] == k, "upper_wildwood"] = wildwood_upper.squeeze() / SOFTPLUS_SCALE

            print("Training the neural network regressor.")
            self.nn_regressors[k].fit(xn_train, yn_train)
            center, spread, skew, tail = self.nn_regressors[k].model(xn_test).numpy().T
            nn_lower, nn_upper = self.nn_regressors[k].predict_intervals(xn_test, alpha=self.alpha)
            df.loc[df["fold"] == k, "center_nn"] = center.squeeze()
            df.loc[df["fold"] == k, "scale_nn"] = spread.squeeze()
            df.loc[df["fold"] == k, "skew_nn"] = skew.squeeze()
            df.loc[df["fold"] == k, "tail_nn"] = tail.squeeze()
            df.loc[df["fold"] == k, "lower_nn"] = nn_lower.squeeze() / SOFTPLUS_SCALE
            df.loc[df["fold"] == k, "upper_nn"] = nn_upper.squeeze() / SOFTPLUS_SCALE

            print("Training the neural network classifier.")
            self.nn_classifiers[k].fit(xnc_train, ync_train)
            nnc_modes = self.nnc_data_processor.bin_values[
                self.nn_classifiers[k].model.predict(xnc_test).argmax(axis=1)
            ]
            nnc_lower, nnc_upper = self.nn_classifiers[k].predict_intervals(xnc_test)
            df.loc[df["fold"] == k, "lower_nnc"] = nnc_lower.squeeze() / SOFTPLUS_SCALE
            df.loc[df["fold"] == k, "mode_nnc"] = nnc_modes.squeeze() / SOFTPLUS_SCALE
            df.loc[df["fold"] == k, "upper_nnc"] = nnc_upper.squeeze() / SOFTPLUS_SCALE

            print("Training the neural network EIM.")
            self.nn_eims[k].fit(xeim_train, yeim_train)
            eim_median = self.nn_eims[k].model.predict(xeim_test)[:, 2]
            eim_lower, eim_upper = self.nn_eims[k].predict_intervals(xeim_test)
            df.loc[df["fold"] == k, "lower_eim"] = eim_lower.squeeze() / SOFTPLUS_SCALE
            df.loc[df["fold"] == k, "median_eim"] = eim_median.squeeze()
            df.loc[df["fold"] == k, "upper_eim"] = eim_upper.squeeze() / SOFTPLUS_SCALE

        return df


class HistBoostEnsembler(BaseEnsembler):
    """
    Create an ensemble model that combines the other models into a HistBoostRegressor.
    """

    def __init__(self, **kwargs) -> None:
        """
        Parameters
        ----------
        **kwargs: dict
            arguments passed to BaseEnsembler
        """
        super(HistBoostEnsembler, self).__init__(**kwargs)

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the histboost ensembler.

        Parameters
        ----------
        df: pd.DataFrame
            The input data
        """
        df = self.prepare_data(df)
        # update the categorical feature mask (upstream predictions are numeric)
        categorical_feature_mask = np.concatenate(
            [
                np.array([False] * len(self.upstream_predictions)),
                self.boost_data_processor.categorical_features,
            ]
        )
        param_grid = {
            "l2_regularization": [0, 1, 2],
            "learning_rate": [0.15, 0.2],
        }
        self.lower_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                quantile=(1 - self.alpha) / 2,
                loss="quantile",
                max_iter=1000,
                categorical_features=categorical_feature_mask,
                max_leaf_nodes=21,
                max_depth=4,
                min_samples_leaf=100,
            ),
            param_grid=param_grid,
            scoring=make_scorer(lambda o, p: d2_pinball_score(o, p, alpha=(1 - self.alpha) / 2)),
            verbose=1,
        )
        self.upper_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                quantile=self.alpha + (1 - self.alpha) / 2,
                loss="quantile",
                max_iter=1000,
                categorical_features=categorical_feature_mask,
                max_leaf_nodes=21,
                max_depth=4,
                min_samples_leaf=100,
            ),
            param_grid=param_grid,
            scoring=make_scorer(
                lambda o, p: d2_pinball_score(o, p, alpha=self.alpha + (1 - self.alpha) / 2)
            ),
            verbose=1,
        )
        self.median_regressor = GridSearchCV(
            estimator=HistGradientBoostingRegressor(
                quantile=0.5,
                loss="quantile",
                max_iter=1000,
                categorical_features=categorical_feature_mask,
                max_leaf_nodes=21,
                max_depth=4,
                min_samples_leaf=100,
            ),
            param_grid=param_grid,
            scoring=make_scorer(lambda o, p: d2_pinball_score(o, p, alpha=0.5)),
            verbose=1,
        )

        print("Training the ensemble model.")
        x_ens = np.hstack(
            [
                df[self.upstream_predictions].values,
                self.boost_data_processor(df.drop(["DBWT"], axis=1)),
            ]
        )
        y_ens = df["DBWT"].values.squeeze()
        xtr, xval, ytr, yval = train_test_split(x_ens, y_ens, random_state=1, test_size=0.3)

        self.median_regressor.fit(xtr, ytr.squeeze())
        self.lower_regressor.fit(xtr, ytr.squeeze())
        self.upper_regressor.fit(xtr, ytr.squeeze())

        print("Calibrating with Mapie.")
        self.calibrator = MapieQuantileRegressor(
            [
                self.lower_regressor.best_estimator_,
                self.upper_regressor.best_estimator_,
                self.median_regressor.best_estimator_,
            ],
            alpha=1 - self.alpha,
            cv="prefit",
        )
        self.calibrator.fit(xval, yval.squeeze())

    def predict_intervals(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict the alpha * 100% interval for birthweight.

        Parameters
        ----------
        df: pd.DataFrame
            The input data

        Returns
        -------
        Tuple
            the arrays corresponding to the lower and upper bounds, respectively
        """
        df = df.copy()
        if "DBWT" in df.columns:
            df = df.drop("DBWT", axis=1)

        xr = self.ridge_data_processor(df)
        xb = self.boost_data_processor(df)
        xw = self.wildwood_data_processor(df)
        xn = self.nn_data_processor(df)
        xnc = self.nnc_data_processor(df)
        xeim = self.eim_data_processor(df)

        lowers, uppers = [], []
        for k in range(self.n_folds):
            rm = self.ridge_regressors[k].predict(xr)
            rl, ru = self.ridge_regressors[k].predict_intervals(xr)
            hl, hu = self.histboosters[k].predict_intervals(xb)
            wm = self.wildwood_regressors[k].regressor.predict(xw)
            wl, wu = self.wildwood_regressors[k].predict_intervals(xw)
            nc, ns, nk, nt = self.nn_regressors[k].model(xn).numpy().T
            nl, nu = self.nn_regressors[k].predict_intervals(xn)
            nncm = self.nnc_data_processor.bin_values[
                self.nn_classifiers[k].model.predict(xnc).argmax(axis=1)
            ]
            nncl, nncu = self.nn_classifiers[k].predict_intervals(xnc)
            em = self.nn_eims[k].model.predict(xeim)[:, 2]
            el, eu = self.nn_eims[k].predict_intervals(xeim)
            x = np.hstack(
                [
                    rl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    rm.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    ru.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    hl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    hu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    wl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    wm.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    wu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nc.reshape((-1, 1)),
                    ns.reshape((-1, 1)),
                    nk.reshape((-1, 1)),
                    nt.reshape((-1, 1)),
                    nl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nncl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nncm.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nncu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    el.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    em.reshape((-1, 1)),
                    eu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                ]
            )
            _, intervals = self.calibrator.predict(np.hstack([x, xb]))
            lower, upper = intervals.squeeze().T
            lowers.append(lower)
            uppers.append(upper)

        lower = np.hstack([item.reshape((-1, 1)) for item in lowers]).mean(axis=1)
        upper = np.hstack([item.reshape((-1, 1)) for item in uppers]).mean(axis=1)
        return lower.squeeze(), upper.squeeze()


class NeuralNetEnsembler(BaseEnsembler):
    """
    Create an ensemble model that combines the other models into a MissingnessNeuralNetRegressor.
    """

    def __init__(self, n_folds: int = 3, alpha: float = 0.9, **kwargs) -> None:
        """
        Parameters
        ----------
        n_folds: int
            number of folds to use for held-out training
        alpha: float
            significance level for prediction intervals
        **kwargs: dict
            arguments passed to MissingnessNeuralNetRegressor
        """
        super(NeuralNetEnsembler, self).__init__(n_folds, alpha)
        self.neural_net = MissingnessNeuralNetRegressor(bayesian=True, fit_tail=True, **kwargs)

    def fit(self, df: pd.DataFrame) -> None:
        """
        Fit the neural network ensembler.

        Parameters
        ----------
        df: pd.DataFrame
            The input data
        """
        df = self.prepare_data(df)

        print("Training the ensemble model.")
        x_ens = np.hstack(
            [
                df[self.upstream_predictions].values,
                self.nn_data_processor(df.drop(["DBWT"], axis=1)),
            ]
        )
        y_ens = np_softplus_inv(df["DBWT"].values.squeeze() / SOFTPLUS_SCALE)
        self.neural_net.fit(x_ens, y_ens)

    def predict_intervals(
        self, df: pd.DataFrame, alpha: float = 0.9, n_samples: int = 200
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
                Predict the alpha * 100% interval for birthweight.
        nn_classifiers
                Parameters
                ----------
                df: pd.DataFrame
                    The input data
                alpha: float
                    significance level for prediction intervals. Can be different from the value
                    passed at initialization.
                n_samples: int
                    This many samples are drawn from the posterior distribution

                Returns
                -------
                Tuple
                    the arrays corresponding to the lower and upper bounds, respectively
        """
        df = df.copy()
        if "DBWT" in df.columns:
            df = df.drop("DBWT", axis=1)

        xr = self.ridge_data_processor(df)
        xb = self.boost_data_processor(df)
        xw = self.wildwood_data_processor(df)
        xn = self.nn_data_processor(df)
        xnc = self.nnc_data_processor(df)
        xeim = self.eim_data_processor(df)

        predicted_samples_list = []
        for k in range(self.n_folds):
            rm = self.ridge_regressors[k].predict(xr)
            rl, ru = self.ridge_regressors[k].predict_intervals(xr)
            hl, hu = self.histboosters[k].predict_intervals(xb)
            wm = self.wildwood_regressors[k].regressor.predict(xw)
            wl, wu = self.wildwood_regressors[k].predict_intervals(xw)
            nc, ns, nk, nt = self.nn_regressors[k].model(xn).numpy().T
            nl, nu = self.nn_regressors[k].predict_intervals(xn)
            nncm = self.nnc_data_processor.bin_values[
                self.nn_classifiers[k].model.predict(xnc).argmax(axis=1)
            ]
            nncl, nncu = self.nn_classifiers[k].predict_intervals(xnc)
            em = self.nn_eims[k].model.predict(xeim)[:, 2]
            el, eu = self.nn_eims[k].predict_intervals(xeim)
            x = np.hstack(
                [
                    rl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    rm.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    ru.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    hl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    hu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    wl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    wm.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    wu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nc.reshape((-1, 1)),
                    ns.reshape((-1, 1)),
                    nk.reshape((-1, 1)),
                    nt.reshape((-1, 1)),
                    nl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nncl.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nncm.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    nncu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    el.reshape((-1, 1)) / SOFTPLUS_SCALE,
                    em.reshape((-1, 1)),
                    eu.reshape((-1, 1)) / SOFTPLUS_SCALE,
                ]
            )
            x_inputs = np.hstack([x, xn])

            predicted_samples = np.random.randn(n_samples, x_inputs.shape[0])
            for i, sample in tqdm(
                enumerate(predicted_samples), total=n_samples, desc="Sampling from ensembler"
            ):
                center, spread, skew, tail = self.neural_net.model(x_inputs).numpy().T
                spread = spread + 1e-3
                predicted_samples[i] = center + (tail * spread) * (
                    np.sinh((1 / tail) * np.arcsinh(sample) + skew / tail)
                )
            predicted_samples_list.append(predicted_samples)

        all_predicted_samples = np.vstack(predicted_samples_list)
        lower, upper = np.apply_along_axis(
            func1d=lambda x: compute_highest_density_interval(
                np_softplus(x) * SOFTPLUS_SCALE, alpha=alpha
            ),
            axis=0,
            arr=all_predicted_samples,
        )
        return lower, upper
