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
