
# main.py
import asyncio
import json
from logging_config import setup_logging
from optimization import parallel_bayesian_optimization
from knowledge_base import SimpleKnowledgeBase
from models import choose_model_based_on_complexity
from skopt.space import Real, Integer
from sklearn.model_selection import train_test_split

async def main():
    logger = setup_logging()
    
    # Load configuration
    with open('config.json', 'r') as f:
        config = json.load(f)
    
    # Initialize knowledge base
    knowledge_base = SimpleKnowledgeBase()
    
    # Setup model and optimization parameters
    param_space = {
        'learning_rate': Real(1e-6, 1e-2, prior='log-uniform'),
        'n_estimators': Integer(50, 200),
        'max_depth': Integer(5, 15),
        'subsample': Real(0.5, 1.0),
        'min_samples_split': Integer(2, 10),
        'min_samples_leaf': Integer(1, 10),
    }

    
    
    # Run optimization
    best_params, best_score = await parallel_bayesian_optimization(
        param_space, X_train, y_train, X_test, y_test, n_iterations=5
    )
    
    if best_params is not None:
        logger.info(f"Optimization complete. Best parameters: {best_params}")
        logger.info(f"Best MSE: {best_score}")
    else:
        logger.error("Optimization failed to produce valid results.")

if __name__ == "__main__":
    asyncio.run(main())

