"""
Model Training Module

Trains LightGBM model with MLflow experiment tracking.
Supports both default hyperparameters and Ray Tune optimized configs.
"""

import time
from pathlib import Path
from typing import Optional

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import typer
from typing_extensions import Annotated

from config import (
    FEATURES_DIR, MODELS_DIR, LGBM_PARAMS, N_ESTIMATORS,
    EARLY_STOPPING_ROUNDS, MLFLOW_EXPERIMENT_NAME, RANDOM_SEED, set_seed
)


# Initialize Typer app
app = typer.Typer()


@app.command()
def train_model(
    experiment_name: Annotated[str, typer.Option(help="MLflow experiment name")] = MLFLOW_EXPERIMENT_NAME,
    data_file: Annotated[str, typer.Option(help="Preprocessed data pickle file")] = "preprocessed_data_ray.pkl",
    use_tuned_params: Annotated[bool, typer.Option(help="Use Ray Tune optimized hyperparameters")] = False,
    model_name: Annotated[str, typer.Option(help="Saved model filename")] = "lightgbm_model.pkl"
) -> None:
    """
    Main training workload: Train LightGBM model with MLflow tracking.
    
    This is a stateless function that:
    1. Loads preprocessed data
    2. Optionally loads tuned hyperparameters from Ray Tune
    3. Trains LightGBM model with early stopping
    4. Logs all metrics and parameters to MLflow
    5. Saves trained model and artifacts
    """
    print("="*80)
    print("MODEL TRAINING WORKLOAD")
    print("="*80)
    
    set_seed()
    
    # Set MLflow experiment
    mlflow.set_experiment(experiment_name)
    
    # Load preprocessed data
    data_path = FEATURES_DIR / data_file
    print(f"\nLoading data from {data_path}")
    data = joblib.load(data_path)
    
    X_train = data['X_train']
    y_train = data['y_train']
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    label_encoder = data['label_encoder']
    scaler = data['scaler']
    
    print(f"Data loaded:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    print(f"  Features: {X_train.shape}")
    print(f"  Genres: {len(label_encoder.classes_)}")
    
    # Load hyperparameters
    if use_tuned_params:
        tuned_config_path = MODELS_DIR / 'best_config_ray.pkl'
        if tuned_config_path.exists():
            print(f"\nLoading tuned hyperparameters from {tuned_config_path}")
            best_config = joblib.load(tuned_config_path)
        else:
            print(f"\nTuned config not found, using defaults")
            best_config = LGBM_PARAMS.copy()
    else:
        print("\nUsing default hyperparameters")
        best_config = LGBM_PARAMS.copy()
    
    # Update config for LightGBM
    lgbm_params = best_config.copy()
    lgbm_params.update({
        'objective': 'multiclass',
        'num_class': len(label_encoder.classes_),
        'metric': 'multi_logloss',
        'random_state': RANDOM_SEED,
        'n_jobs': -1,
        'verbose': -1
    })
    
    # Start MLflow run
    with mlflow.start_run() as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")
        
        # Log parameters
        mlflow.log_params(lgbm_params)
        mlflow.log_param("n_estimators", N_ESTIMATORS)
        mlflow.log_param("early_stopping_rounds", EARLY_STOPPING_ROUNDS)
        
        # Train model
        print("\n" + "="*80)
        print("Training LightGBM model...")
        print("="*80)
        
        start_time = time.time()
        
        model = lgb.LGBMClassifier(n_estimators=N_ESTIMATORS, **lgbm_params)
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['train', 'val'],
            eval_metric='multi_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=True),
                lgb.log_evaluation(period=50)
            ]
        )
        
        training_time = (time.time() - start_time) / 60
        print(f"\nTraining completed in {training_time:.1f} minutes")
        print(f"Best iteration: {model.best_iteration_}")
        
        # Evaluate on all sets
        print("\n" + "="*80)
        print("Evaluating model...")
        print("="*80)
        
        # Train metrics
        y_train_pred = model.predict(X_train)
        train_acc = accuracy_score(y_train, y_train_pred)
        train_f1_macro = f1_score(y_train, y_train_pred, average='macro')
        train_f1_weighted = f1_score(y_train, y_train_pred, average='weighted')
        
        # Val metrics
        y_val_pred = model.predict(X_val)
        val_acc = accuracy_score(y_val, y_val_pred)
        val_f1_macro = f1_score(y_val, y_val_pred, average='macro')
        val_f1_weighted = f1_score(y_val, y_val_pred, average='weighted')
        
        # Test metrics
        y_test_pred = model.predict(X_test)
        test_acc = accuracy_score(y_test, y_test_pred)
        test_f1_macro = f1_score(y_test, y_test_pred, average='macro')
        test_f1_weighted = f1_score(y_test, y_test_pred, average='weighted')
        
        print(f"  TRAIN: Accuracy={train_acc:.4f}, F1(macro)={train_f1_macro:.4f}")
        print(f"  VAL:   Accuracy={val_acc:.4f}, F1(macro)={val_f1_macro:.4f}")
        print(f"  TEST:  Accuracy={test_acc:.4f}, F1(macro)={test_f1_macro:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_f1_macro", train_f1_macro)
        mlflow.log_metric("train_f1_weighted", train_f1_weighted)
        
        mlflow.log_metric("val_accuracy", val_acc)
        mlflow.log_metric("val_f1_macro", val_f1_macro)
        mlflow.log_metric("val_f1_weighted", val_f1_weighted)
        
        mlflow.log_metric("test_accuracy", test_acc)
        mlflow.log_metric("test_f1_macro", test_f1_macro)
        mlflow.log_metric("test_f1_weighted", test_f1_weighted)
        
        mlflow.log_metric("training_time_minutes", training_time)
        mlflow.log_metric("best_iteration", model.best_iteration_)
        
        # Save model
        model_path = MODELS_DIR / model_name
        joblib.dump(model, model_path)
        print(f"\nModel saved to {model_path}")
        
        # Log model to MLflow
        mlflow.sklearn.log_model(model, "model")
        
        # Save artifacts
        artifacts = {
            'model': model,
            'label_encoder': label_encoder,
            'scaler': scaler,
            'hyperparameters': lgbm_params,
            'metrics': {
                'test_accuracy': test_acc,
                'test_f1_macro': test_f1_macro,
                'test_f1_weighted': test_f1_weighted
            }
        }
        
        artifacts_path = MODELS_DIR / 'model_artifacts.pkl'
        # Add to train.py after saving artifacts (line ~127):
        joblib.dump(model, MODELS_DIR / 'lightgbm_model_ray.pkl')
        joblib.dump(scaler, MODELS_DIR / 'scaler_ray.pkl')  
        joblib.dump(label_encoder, MODELS_DIR / 'label_encoder_ray.pkl')

        joblib.dump(artifacts, artifacts_path)
        print(f"Artifacts saved to {artifacts_path}")
        
        print(f"\nMLflow Run ID: {run.info.run_id}")
        print(f"View results: mlflow ui")
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\nFinal Test Results:")
    print(f"  Accuracy:        {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  F1 (macro):      {test_f1_macro:.4f}")
    print(f"  F1 (weighted):   {test_f1_weighted:.4f}")


if __name__ == "__main__":
    app()  # Initialize Typer CLI