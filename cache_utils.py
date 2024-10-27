
# cache_utils.py
import functools
import hashlib

def compute_hash(data):
    return hashlib.sha256(str(data).encode()).hexdigest()

cache_conditions = {
    'X_train_hash': None,
    'y_train_hash': None,
    'hyperparameters_hash': None,
}

def invalidate_cache_if_changed(current_X_train, current_y_train, current_hyperparameters):
    current_X_train_hash = compute_hash(current_X_train)
    current_y_train_hash = compute_hash(current_y_train)
    current_hyperparameters_hash = compute_hash(current_hyperparameters)

    if (cache_conditions['X_train_hash'] != current_X_train_hash or
        cache_conditions['y_train_hash'] != current_y_train_hash or
        cache_conditions['hyperparameters_hash'] != current_hyperparameters_hash):
        
        cached_bayesian_fit.cache_clear()
        
        cache_conditions['X_train_hash'] = current_X_train_hash
        cache_conditions['y_train_hash'] = current_y_train_hash
        cache_conditions['hyperparameters_hash'] = current_hyperparameters_hash