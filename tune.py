"""
Hyperparameter Tuning Module

Uses Ray Tune with ASHA scheduler for distributed hyperparameter optimization.
"""

import tempfile
from pathlib import Path
from typing import Dict, Any

import joblib
import lightgbm as lgb
import numpy as np
import ray
from ray import tune, train
from ray.tune.schedulers import ASHAScheduler
from sklearn.metrics import accuracy_score, f1_score
import typer
from typing_extensions import Annotated

from config import (
    FEATURES_DIR, MODELS_DIR, LGBM_PARAMS, N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS, RANDOM_SEED, RAY_NUM_TRIALS,
    RAY_MAX_CONCURRENT, set_seed
)


# Initialize Typer app
app = typer.Typer()


def short_trial_dirname_creator(trial):
    """Create short trial names to avoid Windows path length limit."""
    return f"trial_{trial.trial_id[:8]}"


def train_lightgbm_ray(config: Dict[str, Any]) -> None:
    """
    Ray Tune training function.
    
    This function is called by Ray Tune for each hyperparameter trial.
    It MUST use train.report() (not tune.report()) per Ray 2.0+ API.
    """
    # Load preprocessed data
    data = joblib.load(FEATURES_DIR / 'preprocessed_data_ray.pkl')
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Build LightGBM parameters
    lgbm_params = {
        'objective': 'multiclass',
        'num_class': len(np.unique(y_train)),
        'metric': 'multi_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': config['num_leaves'],
        'learning_rate': config['learning_rate'],
        'feature_fraction': config['feature_fraction'],
        'bagging_fraction': config['bagging_fraction'],
        'min_child_samples': config['min_child_samples'],
        'random_state': RANDOM_SEED,
        'n_jobs': 1,  # Each trial uses 1 CPU
        'verbose': -1
    }
    
    # Train model
    model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, **lgbm_params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)]
    )
    
    # Evaluate
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1_macro = f1_score(y_val, y_pred, average='macro')
    
    # Report metrics (Ray 2.0+ API requires dict)
    train.report({"accuracy": accuracy, "f1_macro": f1_macro})


@app.command()
def tune_hyperparameters(
    num_trials: Annotated[int, typer.Option(help="Number of Ray Tune trials")] = RAY_NUM_TRIALS,
    max_concurrent: Annotated[int, typer.Option(help="Max concurrent trials")] = RAY_MAX_CONCURRENT,
    output_file: Annotated[str, typer.Option(help="Output file for best config")] = "best_config_ray.pkl"
) -> None:
    """
    Main hyperparameter tuning workload using Ray Tune.
    
    This is a stateless function that:
    1. Initializes Ray
    2. Defines hyperparameter search space
    3. Runs distributed trials with ASHA scheduler
    4. Saves best configuration
    """
    print("="*80)
    print("HYPERPARAMETER TUNING WORKLOAD (Ray Tune)")
    print("="*80)
    
    set_seed()
    
    # Initialize Ray
    ray.shutdown()
    ray.init(ignore_reinit_error=True)
    print(f"\nRay initialized for tuning")
    
    # Define search space
    search_space = {
        'num_leaves': tune.choice([15, 31, 63]),
        'learning_rate': tune.choice([0.01, 0.05, 0.1]),
        'feature_fraction': tune.uniform(0.6, 0.9),
        'bagging_fraction': tune.uniform(0.6, 0.9),
        'min_child_samples': tune.choice([10, 20, 30])
    }
    
    # Configure ASHA scheduler
    scheduler = ASHAScheduler(
        metric="accuracy",
        mode="max",
        max_t=N_ESTIMATORS,
        grace_period=50,
        reduction_factor=2
    )
    
    print(f"\nStarting Ray Tune hyperparameter optimization...")
    print(f"  Number of trials: {num_trials}")
    print(f"  Max concurrent trials: {max_concurrent}")
    print(f"  Search space: {list(search_space.keys())}\n")
    
    # Setup storage path (Windows-safe)
    ray_storage_path = Path(tempfile.gettempdir()) / "ray_tune_music"
    ray_storage_path.mkdir(exist_ok=True)
    print(f"  Ray Tune storage path: {ray_storage_path}\n")
    
    # Run Ray Tune
    try:
        analysis = tune.run(
            train_lightgbm_ray,
            config=search_space,
            num_samples=num_trials,
            scheduler=scheduler,
            resources_per_trial={"cpu": 1},
            max_concurrent_trials=max_concurrent,
            verbose=1,
            storage_path=str(ray_storage_path),
            trial_dirname_creator=short_trial_dirname_creator,
            progress_reporter=tune.CLIReporter(metric_columns=["accuracy", "f1_macro"]),
            checkpoint_config=tune.CheckpointConfig(
                num_to_keep=1,
                checkpoint_frequency=0
            )
        )
        
        print("\nRay Tune completed successfully!")
        
        # Get best configuration
        best_config = analysis.get_best_config(metric="accuracy", mode="max")
        best_result = analysis.best_result
        
        print("\n" + "="*80)
        print("RAY TUNE RESULTS")
        print("="*80)
        print(f"\nBest configuration found:")
        for key, value in best_config.items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        
        print(f"\nBest validation accuracy: {best_result['accuracy']:.4f}")
        print(f"Best validation F1 (macro): {best_result['f1_macro']:.4f}")
        
        # Save best config
        save_path = MODELS_DIR / output_file
        joblib.dump(best_config, save_path)
        print(f"\nBest configuration saved to {save_path}")
        
    except Exception as e:
        print(f"\nRay Tune failed: {e}")
        print("\nFalling back to default parameters...")
        best_config = LGBM_PARAMS.copy()
        save_path = MODELS_DIR / output_file
        joblib.dump(best_config, save_path)
    
    # Shutdown Ray
    ray.shutdown()
    print("\nRay shutdown complete")
    
    print("\n" + "="*80)
    print("HYPERPARAMETER TUNING COMPLETE!")
    print("="*80)
    print(f"\nNext step: Train model with tuned hyperparameters:")
    print(f"  python psmdsbe102_2526A/train.py --use-tuned-params")


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()  # Initialize Typer CLI