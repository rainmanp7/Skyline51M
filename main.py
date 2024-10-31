
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

process_manager = AsyncProcessManager()
    
    # Create tasks
    tasks = [
        ProcessTask(
            name="model_training",
            priority=1,
            function=model.fit,
            args=(X_train, y_train),
            kwargs={}
        ),
        ProcessTask(
            name="hyperparameter_optimization",
            priority=2,
            function=optimizer.optimize,
            args=(param_space,),
            kwargs={}
        )
    ]
    
    # Submit and run tasks
    for task in tasks:
        await process_manager.submit_task(task)
    
    results = await process_manager.run_tasks()
    await process_manager.cleanup()
    
    return results

# Run the async process
results = asyncio.run(main())

from sklearn.model_selection import train_test_split

# Assuming X and y are your feature matrix and target vector
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Perform parallel Bayesian optimization with dynamic complexity
best_params, best_score = parallel_bayesian_optimization(
    initial_param_space, X_train, y_train, X_test, y_test, n_iterations=5
)

# Train final model with best parameters
if best_params is not None:
    final_model = YourModelClass().set_params(**best_params)
    final_model.fit(X_train, y_train)
    final_performance = evaluate_performance(final_model, X_test, y_test)
    logging.info(f"Final model MSE on test set: {final_performance}")
else:
    logging.error("Optimization failed to produce valid results.")
    
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

