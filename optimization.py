# optimization.py
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.metrics import make_scorer
import numpy as np

def adjust_initial_search_space(param_space, complexity_factor):
    if complexity_factor < 10:
        param_space['n_estimators'] = Integer(50, 100)
        param_space['learning_rate'] = Real(1e-4, 1e-2, prior='log-uniform')
    elif 10 <= complexity_factor < 100:
        param_space['n_estimators'] = Integer(100, 200)
        param_space['learning_rate'] = Real(1e-5, 1e-2, prior='log-uniform')
    else:
        param_space['n_estimators'] = Integer(200, 400)
        param_space['learning_rate'] = Real(1e-6, 1e-2, prior='log-uniform')
    return param_space

def evaluate_performance(model, X_test, y_test):
    y_pred = model.predict(X_test)
    return np.mean((y_test - y_pred) ** 2)
