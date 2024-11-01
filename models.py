# models.py
class BaseModel:
    def __init__(self):
        self.model = None

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError

### Start Model Validation and Monitoring.
@dataclass
class ModelMetrics:
    mae: float
    mse: float
    r2: float
    training_time: float
    memory_usage: float
    prediction_latency: float

class ModelValidator:
    def __init__(self):
        self.metrics_history = defaultdict(list)
        
    def validate_model(
        self,
        model: Any,
        X_val: np.ndarray,
        y_val: np.ndarray,
        model_key: str
    ) -> ModelMetrics:
        start_time = time.time()
        memory_usage = memory_profiler.memory_usage()
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = ModelMetrics(
            mae=mean_absolute_error(y_val, y_pred),
            mse=mean_squared_error(y_val, y_pred),
            r2=r2_score(y_val, y_pred),
            training_time=time.time() - start_time,
            memory_usage=max(memory_usage) - min(memory_usage),
            prediction_latency=self._measure_prediction_latency(model, X_val)
        )
        
        # Store metrics
        self.metrics_history[model_key].append(metrics)
        
        return metrics
        
    def _measure_prediction_latency(
        self,
        model: Any,
        X: np.ndarray,
        n_iterations: int = 100
    ) -> float:
        latencies = []
        for _ in range(n_iterations):
            start_time = time.time()
            model.predict(X[:100])  # Use small batch for latency test
            latencies.append(time.time() - start_time)
        return np.mean(latencies)

#### End Model Validation and Monitoring.


# Simple 2nd base ModelSelector ####
 class ModelSelector:
    def __init__(self):
        self.model_configs = {
            'simple': ModelConfig(
                model_class=LinearRegression,
                default_params={
                    'fit_intercept': True,
                    'normalize': False
                },
                complexity_level='low',
                validation_metrics=['mae', 'mse'],
                memory_requirement='low'
            ),
            'medium': ModelConfig(
                model_class=RandomForestRegressor,
                default_params={
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'n_jobs': -1
                },
                complexity_level='medium',
                validation_metrics=['mae', 'mse', 'r2'],
                memory_requirement='medium'
            ),
            'complex': ModelConfig(
                model_class=MLPRegressor,
                default_params={
                    'hidden_layer_sizes': (100, 50),
                    'max_iter': 1000,
                    'early_stopping': True,
                    'validation_fraction': 0.1,
                    'learning_rate': 'adaptive',
                    'solver': 'adam'
                },
                complexity_level='high',
                validation_metrics=['mae', 'mse', 'r2', 'explained_variance'],
                memory_requirement='high'
            )
        }
        self.model_cache = {}
        
    def choose_model_based_on_complexity(
        self, 
        complexity_factor: float,
        data_size: Optional[int] = None,
        available_memory: Optional[float] = None,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        try:
            # Memory-aware model selection
            if available_memory is not None:
                suitable_models = self._filter_models_by_memory(available_memory)
            else:
                suitable_models = self.model_configs
                
            # Select model based on complexity and data size
            model_key = self._select_model_key(
                complexity_factor, 
                data_size, 
                list(suitable_models.keys())
            )
            
            # Cache key generation
            cache_key = self._generate_cache_key(model_key, custom_params)
            
            # Check cache first
            if cache_key in self.model_cache:
                return self.model_cache[cache_key]
                
            config = self.model_configs[model_key]
            params = config.default_params.copy()
            
            if custom_params:
                params.update(custom_params)
                
            # Resource optimization for different scenarios
            params = self._optimize_params_for_resources(
                params, 
                data_size, 
                available_memory
            )
                
            model = config.model_class(**params)
            
            # Cache the model
            self.model_cache[cache_key] = model
            
            logging.info(
                f"Selected {model_key} model (complexity: {config.complexity_level})"
                f" for complexity factor {complexity_factor}"
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Error in model selection: {str(e)}", exc_info=True)
            return self._get_fallback_model()

    def _filter_models_by_memory(
        self, 
        available_memory: float
    ) -> Dict[str, ModelConfig]:
        memory_thresholds = {
            'low': 1.0,     # 1 GB
            'medium': 4.0,  # 4 GB
            'high': 8.0     # 8 GB
        }
        
        return {
            key: config for key, config in self.model_configs.items()
            if memory_thresholds[config.memory_requirement] <= available_memory
        }

    def _select_model_key(
        self, 
        complexity_factor: float,
        data_size: Optional[int],
        available_models: List[str]
    ) -> str:
        if complexity_factor < 10:
            preferred_key = 'simple'
        elif 10 <= complexity_factor < 100:
            preferred_key = 'medium'
        else:
            preferred_key = 'complex'
            
        return preferred_key if preferred_key in available_models else available_models[0]

    def _optimize_params_for_resources(
        self, 
        params: Dict[str, Any],
        data_size: Optional[int],
        available_memory: Optional[float]
    ) -> Dict[str, Any]:
        if data_size and available_memory:
            # Adjust batch size based on data size and available memory
            if 'batch_size' in params:
                params['batch_size'] = min(
                    params['batch_size'],
                    int(data_size * 0.1),  # 10% of data
                    int((available_memory * 0.8 * 1e9) / (data_size * 8))  # 80% of memory
                )
            
            # Adjust model complexity parameters
            if 'n_estimators' in params:
                params['n_estimators'] = min(
                    params['n_estimators'],
                    int(np.sqrt(data_size))
                )
                
        return params

    def _generate_cache_key(
        self, 
        model_key: str,
        custom_params: Optional[Dict[str, Any]]
    ) -> str:
        if custom_params:
            param_str = json.dumps(custom_params, sort_keys=True)
            return f"{model_key}_{hashlib.md5(param_str.encode()).hexdigest()}"
        return model_key

    def _get_fallback_model(self) -> Any:
        return self.model_configs['simple'].model_class(
            **self.model_configs['simple'].default_params
        )

# Simple 2nd mod end.####

# base simple model mod start.

    from sklearn.base import BaseEstimator, RegressorMixin
import numpy as np

class SimpleModel(BaseModel, BaseEstimator, RegressorMixin):
    def __init__(self, learning_rate: float = 0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SimpleModel':
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for _ in range(100):  # Simple iteration count
            y_pred = self.predict(X)
            error = y_pred - y
            
            # Update weights and bias
            self.weights -= self.learning_rate * (2/n_samples) * X.T.dot(error)
            self.bias -= self.learning_rate * (2/n_samples) * np.sum(error)
            
        self.is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        return np.dot(X, self.weights) + self.bias

# base simple model mod end.

class MediumModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Medium model implementation

class ComplexModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Complex model implementation

@dataclass
class ModelConfig:
    model_class: Any
    default_params: Dict[str, Any]
    complexity_level: str
    
class ModelSelector:
    def __init__(self):
        self.model_configs = {
            'simple': ModelConfig(
                model_class=LinearRegression,
                default_params={},
                complexity_level='low'
            ),
            'medium': ModelConfig(
                model_class=RandomForestRegressor,
                default_params={
                    'n_estimators': 100,
                    'max_depth': 10
                },
                complexity_level='medium'
            ),
            'complex': ModelConfig(
                model_class=MLPRegressor,
                default_params={
                    'hidden_layer_sizes': (100, 50),
                    'max_iter': 1000,
                    'early_stopping': True
                },
                complexity_level='high'
            )
        }
        
    def choose_model_based_on_complexity(
        self, 
        complexity_factor: float,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """
        Select and instantiate appropriate model based on complexity factor.
        
        Args:
            complexity_factor: Float indicating data/task complexity
            custom_params: Optional dictionary of model parameters to override defaults
            
        Returns:
            Instantiated model object
        """
        try:
            if complexity_factor < 10:
                model_key = 'simple'
            elif 10 <= complexity_factor < 100:
                model_key = 'medium'
            else:
                model_key = 'complex'
                
            config = self.model_configs[model_key]
            params = config.default_params.copy()
            
            if custom_params:
                params.update(custom_params)
                
            model = config.model_class(**params)
            
            logging.info(
                f"Selected {model_key} model with complexity level "
                f"{config.complexity_level} for complexity factor {complexity_factor}"
            )
            
            return model
            
        except Exception as e:
            logging.error(f"Error in model selection: {str(e)}", exc_info=True)
            # Fallback to simple model if error occurs
            return self.model_configs['simple'].model_class()

    def get_model_complexity_level(self, model: Any) -> str:
        """Get complexity level of a model instance."""
        for config in self.model_configs.values():
            if isinstance(model, config.model_class):
                return config.complexity_level
        return 'unknown'

