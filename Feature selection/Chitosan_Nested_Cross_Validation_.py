import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from ngboost import NGBRegressor


class nested_cross_validation:

    def __init__(
        self,
        datafile="Dataset_14_features.xlsx",
        model_type=None
    ):

        self.df = pd.read_excel(datafile)

        if model_type == 'MLR':
            self.user_defined_model = LinearRegression()

            self.p_grid = {
                'fit_intercept': [True, False],
                'positive': [True, False]
            }

        elif model_type == 'lasso':
            self.user_defined_model = linear_model.Lasso()

            self.p_grid = {
                'alpha': [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0],
                'positive': [True, False]
            }

        elif model_type == 'kNN':
            self.user_defined_model = KNeighborsRegressor()

            self.p_grid = {
                'n_neighbors': [2, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30, 50],
                'weights': ["uniform", 'distance'],
                'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
                'leaf_size': [10, 30, 50, 75, 100],
                'p': [1, 2],
                'metric': ['minkowski']
            }

        elif model_type == 'PLS':
            self.user_defined_model = PLSRegression()

            self.p_grid = {
                'n_components': [2, 4, 6],
                'max_iter': [250, 500, 750, 1000]
            }

        elif model_type == 'SVR':
            self.user_defined_model = SVR()

            self.p_grid = {
                'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                'degree': [2, 3, 4, 5, 6],
                'gamma': ['scale', 'auto'],
                'C': [0.1, 0.5, 1, 2],
                'epsilon': [0.001, 0.005, 0.01, 0.05, 0.1, 0.15, 0.2],
                'shrinking': [True, False]
            }

        elif model_type == 'DT':
            self.user_defined_model = DecisionTreeRegressor(random_state=4)

            self.p_grid = {
                'criterion': [
                    'squared_error',
                    'friedman_mse',
                    'absolute_error',
                    'poisson'
                ],
                'splitter': ['best', 'random'],
                'max_depth': [None],
                'min_samples_split': [2, 4, 6],
                'min_samples_leaf': [1, 2, 4],
                'max_features': [None, 'auto', 'sqrt', 'log2'],
                'ccp_alpha': [0, 0.05, 0.1, 0.15]
            }

        elif model_type == 'RF':
            self.user_defined_model = RandomForestRegressor(
                random_state=4
            )

            self.p_grid = {
                'n_estimators': [100, 300, 400],
                'criterion': [
                    'squared_error',
                    'absolute_error'
                ],
                'max_depth': [None],
                'min_samples_split': [2, 4, 6, 8],
                'min_samples_leaf': [1, 2, 4],
                'min_weight_fraction_leaf': [0.0],
                'max_features': ['auto', 'sqrt'],
                'max_leaf_nodes': [None],
                'min_impurity_decrease': [0.0],
                'bootstrap': [True],
                'oob_score': [True],
                'ccp_alpha': [0, 0.005, 0.01]
            }

        elif model_type == 'LGBM':
            self.user_defined_model = LGBMRegressor(
                random_state=4
            )

            self.p_grid = {
                "n_estimators": [100, 150, 200, 250, 300, 400, 500, 600],
                'boosting_type': ['gbdt', 'dart', 'goss'],
                'num_leaves': [16, 32, 64, 128, 256],
                'learning_rate': [0.1, 0.01, 0.001, 0.0001],
                'min_child_weight': [
                    0.001, 0.01, 0.1, 1.0, 10.0
                ],
                'subsample': [0.4, 0.6, 0.8, 1.0],
                'min_child_samples': [2, 10, 20, 40, 100],
                'reg_alpha': [0, 0.005, 0.01, 0.015],
                'reg_lambda': [0, 0.005, 0.01, 0.015]
            }

        elif model_type == 'XGB':
            self.user_defined_model = XGBRegressor(
                objective='reg:squarederror'
            )

            self.p_grid = {
                'booster': ['gbtree', 'gblinear', 'dart'],
                "n_estimators": [100, 150, 300, 400],
                'max_depth': [3, 4, 5, 6, 7, 8, 9, 10],
                'gamma': [0, 2, 4, 6, 8, 10],
                'learning_rate': [0.3, 0.2, 0.1, 0.05, 0.01],
                'subsample': [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                'min_child_weight': [1.0, 2.0, 4.0, 5.0],
                'max_delta_step': [1, 2, 4, 6, 8, 10],
                'reg_alpha': [0.001, 0.01, 0.1],
                'reg_lambda': [0.001, 0.01, 0.1]
            }

        elif model_type == 'NGB':

            b1 = DecisionTreeRegressor(
                criterion='squared_error',
                max_depth=2
            )

            b2 = DecisionTreeRegressor(
                criterion='squared_error',
                max_depth=4
            )

            b3 = DecisionTreeRegressor(
                criterion='squared_error',
                max_depth=8
            )

            b4 = DecisionTreeRegressor(
                criterion='squared_error',
                max_depth=12
            )

            b5 = DecisionTreeRegressor(
                criterion='squared_error',
                max_depth=16
            )

            b6 = DecisionTreeRegressor(
                criterion='squared_error',
                max_depth=32
            )

            self.user_defined_model = NGBRegressor()

            self.p_grid = {
                'n_estimators': [
                    100, 200, 300, 400, 500, 600, 800
                ],
                'learning_rate': [
                    0.1,0.01, 0.001
                ],
                'minibatch_frac': [
                   1.0, 0.8, 0.5
                ],
                'col_sample': [
                   1.0, 0.8, 0.5
                ],
                'Base': [
                    b1, b2, b3, b4, b5, b6
                ]
            }

        else:
            print(
                "Please select one of these models: "
                "MLR, lasso, kNN, PLS, SVR, DT, RF, "
                "LGBM, XGB, NGB"
            )

    def input_target(self):

        # Dataset_14_features.xlsx should already contain
        # the 14 predictors.

        X = self.df.drop(
            [
                'Formulation_number',
                'Modification',
                'Drug',
                'Cross_linker',
                'DL'
            ],
            axis='columns'
        )

        print("\nNumber of input features:", X.shape[1])

        print("Input features:")
        print(list(X.columns))

        # Keep the same scaling approach as your previous code
        stdScale = StandardScaler().fit(X)

        self.X = stdScale.transform(X)

        self.Y = self.df['DL']

    def cross_validation(self, input_value):

        if input_value is None:
            NUM_TRIALS = 10
        else:
            NUM_TRIALS = input_value

        self.itr_number = []

        # Inner-loop metrics
        self.inner_results = []
        self.inner_rmse_results = []
        self.inner_r2_results = []

        # Outer-loop metrics
        self.outer_results = []
        self.outer_rmse_results = []
        self.outer_r2_results = []

        self.model_params = []

        # Stored outer-test values
        self.y_test_list = []
        self.pred_list = []

        # NGB prediction intervals
        self.pred_mean_list = []
        self.pred_scale_list = []
        self.pred_lower_list = []
        self.pred_upper_list = []

        for i in range(NUM_TRIALS):

            # ==================================================
            # OUTER CV
            # ==================================================

            cv_outer = KFold(
                n_splits=10,
                shuffle=True,
                random_state=i
            )

            for train_index, test_index in cv_outer.split(
                self.X,
                self.Y
            ):

                X_train = self.X[train_index]
                X_test = self.X[test_index]

                y_train = self.Y[train_index]
                y_test = self.Y[test_index]

                y_test = np.array(y_test)

                self.y_test_list.append(y_test)

                # ==================================================
                # INNER CV
                # ==================================================

                cv_inner = KFold(
                    n_splits=15
                )

                # MAE remains the main model-selection metric.
                #
                # RMSE and R2 are also calculated for the selected
                # hyperparameter combination.

                scoring = {
                    'MAE': 'neg_mean_absolute_error',
                    'RMSE': 'neg_root_mean_squared_error',
                    'R2': 'r2'
                }

                search = RandomizedSearchCV(
                    self.user_defined_model,
                    self.p_grid,
                    n_iter=40,
                    verbose=1,
                    scoring=scoring,
                    refit='MAE',
                    cv=cv_inner,
                    n_jobs=-1
                )

                result = search.fit(
                    X_train,
                    y_train
                )

                # ==================================================
                # BEST MODEL FROM INNER CV
                # ==================================================

                best_model = result.best_estimator_

                best_index = result.best_index_

                # Main metric: validation MAE
                best_score = abs(
                    result.cv_results_[
                        'mean_test_MAE'
                    ][best_index]
                )

                # Additional validation metrics
                best_rmse = abs(
                    result.cv_results_[
                        'mean_test_RMSE'
                    ][best_index]
                )

                best_r2 = (
                    result.cv_results_[
                        'mean_test_R2'
                    ][best_index]
                )

                self.inner_results.append(
                    best_score
                )

                self.inner_rmse_results.append(
                    best_rmse
                )

                self.inner_r2_results.append(
                    best_r2
                )

                # ==================================================
                # OUTER TEST PREDICTION
                # ==================================================

                yhat = best_model.predict(
                    X_test
                )

                self.pred_list.append(
                    yhat
                )

                # ==================================================
                # OUTER TEST METRICS
                # ==================================================

                test_mae = mean_absolute_error(
                    y_test,
                    yhat
                )

                test_rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        yhat
                    )
                )

                test_r2 = r2_score(
                    y_test,
                    yhat
                )

                self.outer_results.append(
                    test_mae
                )

                self.outer_rmse_results.append(
                    test_rmse
                )

                self.outer_r2_results.append(
                    test_r2
                )

                # ==================================================
                # NGBOOST PREDICTION INTERVALS
                # ==================================================

                if self.user_defined_model.__class__.__name__ == "NGBRegressor":

                    try:

                        pred_dist = best_model.pred_dist(
                            X_test
                        )

                        # Predictive mean
                        pred_mean = np.asarray(
                            pred_dist.mean()
                        )

                        # Predictive scale
                        pred_scale = np.asarray(
                            pred_dist.scale
                        )

                        # 95% prediction interval
                        lower = np.asarray(
                            pred_dist.ppf(0.025)
                        )

                        upper = np.asarray(
                            pred_dist.ppf(0.975)
                        )

                        self.pred_mean_list.append(
                            pred_mean
                        )

                        self.pred_scale_list.append(
                            pred_scale
                        )

                        self.pred_lower_list.append(
                            lower
                        )

                        self.pred_upper_list.append(
                            upper
                        )

                    except Exception as e:

                        print(
                            "\nWARNING: Could not calculate "
                            "NGBoost prediction interval."
                        )

                        print(
                            type(e).__name__,
                            e
                        )

                        self.pred_mean_list.append(
                            np.full(len(y_test), np.nan)
                        )

                        self.pred_scale_list.append(
                            np.full(len(y_test), np.nan)
                        )

                        self.pred_lower_list.append(
                            np.full(len(y_test), np.nan)
                        )

                        self.pred_upper_list.append(
                            np.full(len(y_test), np.nan)
                        )

                else:

                    # Empty placeholders for non-NGB models
                    self.pred_mean_list.append(
                        np.full(len(y_test), np.nan)
                    )

                    self.pred_scale_list.append(
                        np.full(len(y_test), np.nan)
                    )

                    self.pred_lower_list.append(
                        np.full(len(y_test), np.nan)
                    )

                    self.pred_upper_list.append(
                        np.full(len(y_test), np.nan)
                    )

                # ==================================================
                # SAVE PARAMETERS
                # ==================================================

                self.model_params.append(
                    result.best_params_
                )

                self.itr_number.append(
                    i + 1
                )

                # ==================================================
                # PRINT RESULTS
                # ==================================================

                print(
                    '\n########################################################'
                )

                print(
                    'Iteration %d completed'
                    % (i + 1)
                )

                print(
                    'Validation MAE: %.3f'
                    % best_score
                )

                print(
                    'Validation RMSE: %.3f'
                    % best_rmse
                )

                print(
                    'Test MAE: %.3f'
                    % test_mae
                )

                print(
                    'Test RMSE: %.3f'
                    % test_rmse
                )

                print(
                    'Test R2: %.3f'
                    % test_r2
                )

                print(
                    '\nBest Model Parameters:\n%s'
                    % result.best_params_
                )

                print(
                    '########################################################\n'
                )

    def results(
        self,
        file_path='results.xlsx'
    ):

        # ==================================================
        # CREATE RESULTS DATAFRAME
        # ==================================================

        list_of_tuples = list(
            zip(
                self.itr_number,
                self.inner_results,
                self.inner_rmse_results,
                self.inner_r2_results,
                self.outer_results,
                self.outer_rmse_results,
                self.outer_r2_results,
                self.model_params,
                self.y_test_list,
                self.pred_list,
                self.pred_mean_list,
                self.pred_scale_list,
                self.pred_lower_list,
                self.pred_upper_list
            )
        )

        CV_dataset = pd.DataFrame(
            list_of_tuples,
            columns=[
                'Iter',
                'Valid Score',
                'Valid RMSE',
                'Valid R2',
                'Test Score',
                'Test RMSE',
                'Test R2',
                'Model Parms',
                'Experimental_Drug_Loading',
                'Predicted_Drug_Loading',
                'NGB_Predicted_Mean',
                'NGB_Predicted_Scale',
                'NGB_PI_Lower_95',
                'NGB_PI_Upper_95'
            ]
        )

        # Same sorting logic as your previous code
        CV_dataset['Score_difference'] = abs(
            CV_dataset['Valid Score'] -
            CV_dataset['Test Score']
        )

        CV_dataset.sort_values(
            by=[
                'Score_difference',
                'Test Score'
            ],
            ascending=True,
            inplace=True
        )

        CV_dataset = CV_dataset.reset_index(
            drop=True
        )

        # Save Excel
        CV_dataset.to_excel(
            file_path,
            index=False
        )

        self.CV_dataset = CV_dataset

    def best_model(self):

        # ==================================================
        # SELECT BEST HYPERPARAMETERS
        # ==================================================

        best_model_params = (
            self.CV_dataset.iloc[0, 7]
        )

        best_model = (
            self.user_defined_model.set_params(
                **best_model_params
            )
        )

        # Fit final NGB model to complete 14-feature dataset
        self.best_model = best_model.fit(
            self.X,
            self.Y
        )