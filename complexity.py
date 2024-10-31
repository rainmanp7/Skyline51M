# Beginning of complexity.py
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable

@dataclass
class ModelConfig:
    model_class: Any
    default_params: Dict[str, Any]
    complexity_level: str
    suggested_iterations: int
    suggested_metric: Callable

class EnhancedModelSelector(ModelSelector):
    def __init__(self):
        super().__init__()
        self.metric_configs = {
            'low': (mean_squared_error, 100),    # Simple metric, fewer iterations
            'medium': (mean_absolute_error, 500), # Balanced metric, medium iterations
            'high': (r2_score, 1000)             # Complex metric, more iterations
        }
    
    def choose_model_and_config(
        self,
        complexity_factor: float,
        custom_params: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, Callable, int]:
        """
        Enhanced model selection that also chooses evaluation metric and iterations.
        
        Returns:
            Tuple[model, evaluation_metric, num_iterations]
        """
        try:
            # Get base model using parent class method
            model = super().choose_model_based_on_complexity(
                complexity_factor, custom_params
            )
            
            # Determine complexity level
            if complexity_factor < 10:
                level = 'low'
            elif 10 <= complexity_factor < 100:
                level = 'medium'
            else:
                level = 'high'
            
            # Get corresponding metric and iterations
            metric, iterations = self.metric_configs[level]
            
            logging.info(
                f"Selected model configuration: {level} complexity\n"
                f"Metric: {metric.__name__}\n"
                f"Iterations: {iterations}"
            )
            
            return model, metric, iterations
            
        except Exception as e:
            logging.error(f"Error in enhanced model selection: {str(e)}", exc_info=True)
            # Fallback to simplest configuration
            return (
                self.model_configs['simple'].model_class(),
                mean_squared_error,
                100
            )
# End of complexity.py
