
# Beginning of optimization.py
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# Define hyperparameter space for Bayesian optimization
param_space = {
    'learning_rate': Real(1e-6, 1e-2, prior='log-uniform'),
    'n_estimators': Integer(50, 200),
    'max_depth': Integer(5, 15),
    'subsample': Real(0.5, 1.0),
    'min_samples_split': Integer(2, 10),
    'min_samples_leaf': Integer(1, 10),
}

# Define a function to adjust the search space
def adjust_search_space(current_space, performance, threshold=0.1):
    adjusted_space = current_space.copy()
    if performance > threshold:
        adjusted_space['learning_rate'] = Real(current_space['learning_rate'].low * 0.1, 
                                               current_space['learning_rate'].high * 10, 
                                               prior='log-uniform')
    else:
        adjusted_space['learning_rate'] = Real(current_space['learning_rate'].low, 
                                               current_space['learning_rate'].high * 0.1, 
                                               prior='log-uniform')
    return adjusted_space
# End of optimization.py
