# Copyright 2021 The CGLB Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


def meanpred_baseline(dataset):
    _, Y_train = dataset.train
    _, Y_test = dataset.test
    pf = np.mean(Y_train)
    pv = np.var(Y_train)
    lml = np.sum(stats.norm.logpdf(Y_train, pf, pv ** 0.5))
    rmse = np.mean((Y_test - pf) ** 2.0) ** 0.5
    lpd = np.mean(stats.norm.logpdf(Y_test, pf, pv ** 0.5))
    return dict(lml=lml, rmse=rmse, lpd=lpd)


def linear_baseline(dataset):
    X_train, Y_train = dataset.train
    X_test, Y_test = dataset.test
    reg = LinearRegression().fit(X_train, Y_train)
    residuals = reg.predict(X_train) - Y_train
    pred_var = np.var(residuals)
    lml = np.sum(stats.norm.logpdf(residuals, scale=pred_var ** 0.5))
    residuals_test = reg.predict(X_test) - Y_test
    rmse = np.mean(residuals_test ** 2.0) ** 0.5
    lpd = np.mean(stats.norm.logpdf(residuals_test, scale=pred_var ** 0.5))
    return dict(lml=lml, rmse=rmse, lpd=lpd)
