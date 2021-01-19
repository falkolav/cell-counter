from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor

import xgboost as xgb
import pandas as pd
import numpy as np

class MLRegressor():
    def __init__(self, train_x, train_y, test_x, test_y):
        self.results = pd.DataFrame(columns=['SVR', 'XGB', 'RR', 'NNR', 'GTB'], index=['prediction', 'error'])
        self.train_x = train_x
        self.test_x = test_x
        self.train_y = train_y
        self.test_y = test_y

    def run_SVR(self):
        # SVM Regression
        print("SVM Regression:")
        model = 'SVR'

        regr = make_pipeline(StandardScaler(), SVR())
        regr.fit(self.train_x, self.train_y)

        predict = regr.predict(self.test_x)
        self.results.loc['prediction', model] = predict
        error = abs(predict - np.asarray(self.test_y, dtype=np.float64))
        self.results.loc['error', model] = error
        mae = round(np.mean(error), 0)
        self.results.loc['mae', model] = mae

        print('Mean Absolute Error:', mae, 'cells.')

    def run_XGBoost(self):
        # XGBoost
        print("XBBoost:")
        model = 'XGB'

        regr = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
                                max_depth=5, alpha=10, n_estimators=10)
        regr.fit(self.train_x, self.train_y)
        predict = regr.predict(self.test_x)
        self.results.loc['prediction', model] = predict
        error = abs(predict - np.asarray(self.test_y, dtype=np.float64))
        self.results.loc['error', model] = error
        mae = round(np.mean(error), 0)
        self.results.loc['mae', model] = mae

        print('Mean Absolute Error:', mae, 'cells.')

    def run_RR(self):
        # Ridge Regress
        print("Ridge_Regression:")
        model = 'RR'

        regr = make_pipeline(StandardScaler(), linear_model.Ridge())
        regr.fit(self.train_x, self.train_y)
        predict = regr.predict(self.test_x)
        self.results.loc['prediction', model] = predict
        error = abs(predict - np.asarray(self.test_y, dtype=np.float64))
        self.results.loc['error', model] = error
        mae = round(np.mean(error), 0)
        self.results.loc['mae', model] = mae

        print('Mean Absolute Error:', mae, 'cells.')

    def run_NNR(self):
        # Ridge Regression
        print("Nearest Neighbor Regression:")
        model = 'NNR'

        regr = KNeighborsRegressor(n_neighbors=3)
        regr.fit(self.train_x, self.train_y)
        predict = regr.predict(self.test_x)
        self.results.loc['prediction', model] = predict
        error = abs(predict - np.asarray(self.test_y, dtype=np.float64))
        self.results.loc['error', model] = error
        mae = round(np.mean(error), 0)
        self.results.loc['mae', model] = mae

        print('Mean Absolute Error:', mae, 'cells.')

    def run_GTB(self):
        # Gradient Tree Boosting
        print("Gradient Tree Boosting:")
        model = 'GTB'

        regr = GradientBoostingRegressor(n_estimators=50, learning_rate=0.1, max_depth=1, random_state=0,
                                         loss='ls')
        regr.fit(self.train_x, self.train_y)
        predict = regr.predict(self.test_x)
        self.results.loc['prediction', model] = predict
        error = abs(predict - np.asarray(self.test_y, dtype=np.float64))
        self.results.loc['error', model] = error
        mae = round(np.mean(error), 0)
        self.results.loc['mae', model] = mae

        print('Mean Absolute Error:', mae, 'cells.')

    def run_all_tests(self):
        self.run_GTB()
        self.run_NNR()
        self.run_RR()
        self.run_SVR()
        self.run_XGBoost()