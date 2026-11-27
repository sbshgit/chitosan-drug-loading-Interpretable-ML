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
        datafile="Dataset_15_features.xlsx",
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
                'alpha': [
                    0.01, 0.02, 0.05, 0.1,
                    0.25, 0.5, 1.0
                ],
                'positive': [True, False]
            }

        elif model_type == 'kNN':

            self.user_defined_model = KNeighborsRegressor()

            self.p_grid = {
                'n_neighbors': [
                    2, 4, 5, 6, 8, 10, 12,
                    15, 20, 25, 30, 50
                ],
                'weights': [
                    "uniform", 'distance'
                ],
                'algorithm': [
                    'auto',
                    'ball_tree',
                    'kd_tree',
                    'brute'
                ],
                'leaf_size': [
                    10, 30, 50, 75, 100
                ],
                'p': [1, 2],
                'metric': ['minkowski']
            }

        elif model_type == 'PLS':

            self.user_defined_model = PLSRegression()

            self.p_grid = {
                'n_components': [2, 4, 6],
                'max_iter': [
                    250, 500, 750, 1000
                ]
            }

        elif model_type == 'SVR':

            self.user_defined_model = SVR()

            self.p_grid = {
                'kernel': [
                    'linear', 'poly',
                    'rbf', 'sigmoid'
                ],
                'degree': [2, 3, 4, 5, 6],
                'gamma': ['scale', 'auto'],
                'C': [0.1, 0.5, 1, 2],
                'epsilon': [
                    0.001, 0.005, 0.01,
                    0.05, 0.1, 0.15, 0.2
                ],
                'shrinking': [True, False]
            }

        elif model_type == 'DT':

            self.user_defined_model = DecisionTreeRegressor(
                random_state=4
            )

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
                'max_features': [
                    None, 'auto',
                    'sqrt', 'log2'
                ],
                'ccp_alpha': [
                    0, 0.05, 0.1, 0.15
                ]
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
                'min_samples_split': [
                    2, 4, 6, 8
                ],
                'min_samples_leaf': [
                    1, 2, 4
                ],
                'min_weight_fraction_leaf': [0.0],
                'max_features': [
                    'auto', 'sqrt'
                ],
                'max_leaf_nodes': [None],
                'min_impurity_decrease': [0.0],
                'bootstrap': [True],
                'oob_score': [True],
                'ccp_alpha': [
                    0, 0.005, 0.01
                ]
            }

        elif model_type == 'LGBM':

            self.user_defined_model = LGBMRegressor(
                random_state=4
            )

            self.p_grid = {
                "n_estimators": [
                    100, 150, 200, 250,
                    300, 400, 500, 600
                ],
                'boosting_type': [
                    'gbdt', 'dart', 'goss'
                ],
                'num_leaves': [
                    16, 32, 64, 128, 256
                ],
                'learning_rate': [
                    0.1, 0.01, 0.001, 0.0001
                ],
                'min_child_weight': [
                    0.001, 0.01, 0.1,
                    1.0, 10.0
                ],
                'subsample': [
                    0.4, 0.6, 0.8, 1.0
                ],
                'min_child_samples': [
                    2, 10, 20, 40, 100
                ],
                'reg_alpha': [
                    0, 0.005, 0.01, 0.015
                ],
                'reg_lambda': [
                    0, 0.005, 0.01, 0.015
                ]
            }

        elif model_type == 'XGB':

            self.user_defined_model = XGBRegressor(
                objective='reg:squarederror'
            )

            self.p_grid = {
                'booster': [
                    'gbtree', 'gblinear', 'dart'
                ],
                "n_estimators": [
                    100, 150, 300, 400
                ],
                'max_depth': [
                    3, 4, 5, 6, 7, 8, 9, 10
                ],
                'gamma': [
                    0, 2, 4, 6, 8, 10
                ],
                'learning_rate': [
                    0.3, 0.2, 0.1, 0.05, 0.01
                ],
                'subsample': [
                    0.5, 0.6, 0.7,
                    0.8, 0.9, 1.0
                ],
                'min_child_weight': [
                    1.0, 2.0, 4.0, 5.0
                ],
                'max_delta_step': [
                    1, 2, 4, 6, 8, 10
                ],
                'reg_alpha': [
                    0.001, 0.01, 0.1
                ],
                'reg_lambda': [
                    0.001, 0.01, 0.1
                ]
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
                    100
                ],
                'learning_rate': [
                    0.1
                ],
                'minibatch_frac': [
                    0.8
                ],
                'col_sample': [
                     0.8
                ],
                'Base': [
                    b1, b2, b3, b4, b5, b6
                ]
            }

        else:

            print(
                "Please select one of these models: "
                "MLR, lasso, kNN, PLS, SVR, DT, "
                "RF, LGBM, XGB, NGB"
            )

    def input_target(self):

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

        stdScale = StandardScaler().fit(X)

        self.X = stdScale.transform(X)

        self.Y = self.df['DL']

    def cross_validation(self, input_value):

        if input_value is None:
            NUM_TRIALS = 10
        else:
            NUM_TRIALS = input_value

        # ----------------------------------------------------
        # Main MAE results
        # ----------------------------------------------------

        self.itr_number = []
        self.inner_results = []
        self.outer_results = []

        # ----------------------------------------------------
        # Additional inner-loop metrics
        # ----------------------------------------------------

        self.inner_rmse_results = []
        self.inner_r2_results = []

        # ----------------------------------------------------
        # Additional outer-loop metrics
        # ----------------------------------------------------

        self.outer_rmse_results = []
        self.outer_r2_results = []

        # ----------------------------------------------------
        # Other stored results
        # ----------------------------------------------------

        self.model_params = []
        self.y_test_list = []
        self.pred_list = []

        for i in range(NUM_TRIALS):

            # ==================================================
            # OUTER LOOP
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

                # Store test set
                y_test = np.array(y_test)

                self.y_test_list.append(y_test)

                # ==================================================
                # INNER LOOP
                # ==================================================

                cv_inner = KFold(
                    n_splits=15
                )

                # MAE remains the PRIMARY model-selection metric.
                #
                # RMSE and R2 are calculated additionally.

                scoring = {
                    'MAE': 'neg_mean_absolute_error',
                    'RMSE': 'neg_root_mean_squared_error',
                    'R2': 'r2'
                }

                search = RandomizedSearchCV(
                    self.user_defined_model,
                    self.p_grid,
                    n_iter=40,
                    verbose=0,
                    scoring=scoring,
                    cv=cv_inner,
                    n_jobs=-1,
                    refit='MAE'
                )

                # Execute search
                result = search.fit(
                    X_train,
                    y_train
                )

                # ==================================================
                # BEST MODEL
                # ==================================================

                best_model = result.best_estimator_

                # Index of MAE-selected hyperparameter combination
                best_index = result.best_index_

                # --------------------------------------------------
                # INNER MAE
                # --------------------------------------------------

                best_score = abs(
                    result.cv_results_[
                        'mean_test_MAE'
                    ][best_index]
                )

                self.inner_results.append(
                    best_score
                )

                # --------------------------------------------------
                # INNER RMSE
                # --------------------------------------------------

                best_rmse = abs(
                    result.cv_results_[
                        'mean_test_RMSE'
                    ][best_index]
                )

                self.inner_rmse_results.append(
                    best_rmse
                )

                # --------------------------------------------------
                # INNER R2
                # --------------------------------------------------

                best_r2 = (
                    result.cv_results_[
                        'mean_test_R2'
                    ][best_index]
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

                self.pred_list.append(yhat)

                # ==================================================
                # OUTER TEST MAE
                # ==================================================

                test_mae = mean_absolute_error(
                    y_test,
                    yhat
                )

                self.outer_results.append(
                    test_mae
                )

                # ==================================================
                # OUTER TEST RMSE
                # ==================================================

                test_rmse = np.sqrt(
                    mean_squared_error(
                        y_test,
                        yhat
                    )
                )

                self.outer_rmse_results.append(
                    test_rmse
                )

                # ==================================================
                # OUTER TEST R2
                # ==================================================

                test_r2 = r2_score(
                    y_test,
                    yhat
                )

                self.outer_r2_results.append(
                    test_r2
                )

                # ==================================================
                # STORE ITERATION AND PARAMETERS
                # ==================================================

                self.itr_number.append(
                    i + 1
                )

                self.model_params.append(
                    result.best_params_
                )

                # ==================================================
                # ONLY PRINT PROGRESS
                # ==================================================

                print(
                    f"Iteration {i + 1} of "
                    f"{NUM_TRIALS} runs completed"
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
                self.pred_list
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
                'Predicted_Drug_Loading'
            ]
        )

        # ==================================================
        # SCORE DIFFERENCE
        # ==================================================

        CV_dataset['Score_difference'] = abs(
            CV_dataset['Valid Score'] -
            CV_dataset['Test Score']
        )

        # ==================================================
        # SORT RESULTS
        # ==================================================

        CV_dataset.sort_values(
            by=[
                'Score_difference',
                'Test Score'
            ],
            ascending=True,
            inplace=True
        )

        CV_dataset.reset_index(
            drop=True,
            inplace=True
        )

        # ==================================================
        # SAVE EXCEL
        # ==================================================

        CV_dataset.to_excel(
            file_path,
            index=False
        )

        self.CV_dataset = CV_dataset

    def best_model(self):

        # ==================================================
        # BEST HYPERPARAMETERS
        # ==================================================

        best_model_params = (
            self.CV_dataset.iloc[0]['Model Parms']
        )

        best_model = (
            self.user_defined_model.set_params(
                **best_model_params
            )
        )

        self.best_model = best_model.fit(
            self.X,
            self.Y
        )