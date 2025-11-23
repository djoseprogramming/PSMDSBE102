"""
Model Evaluation Module

Comprehensive evaluation including per-class metrics, confusion matrix,
feature importance, Cleanlab label quality analysis, and data slicing.
"""

from pathlib import Path
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix
)
from cleanlab.classification import CleanLearning
import typer
from typing_extensions import Annotated

from config import FEATURES_DIR, MODELS_DIR, RESULTS_DIR, set_seed


# Initialize Typer app
app = typer.Typer()


@app.command()
def evaluate_model(
    model_file: Annotated[str, typer.Option(help="Trained model artifacts file")] = "model_artifacts.pkl",
    data_file: Annotated[str, typer.Option(help="Preprocessed data file")] = "preprocessed_data_ray.pkl",
    save_results: Annotated[bool, typer.Option(help="Save evaluation results")] = True
) -> None:
    """
    Main evaluation workload: Comprehensive model evaluation.
    
    This is a stateless function that:
    1. Loads trained model and test data
    2. Computes per-class metrics
    3. Generates confusion matrix
    4. Analyzes feature importance
    5. Runs Cleanlab label quality analysis
    6. Performs data slicing (high vs low confidence)
    7. Saves all results
    """
    print("="*80)
    print("MODEL EVALUATION WORKLOAD")
    print("="*80)
    
    set_seed()
    
    # Load model artifacts
    artifacts_path = MODELS_DIR / model_file
    print(f"\nLoading model from {artifacts_path}")
    artifacts = joblib.load(artifacts_path)
    
    model = artifacts['model']
    label_encoder = artifacts['label_encoder']
    scaler = artifacts['scaler']
    
    # Load test data
    data_path = FEATURES_DIR / data_file
    print(f"Loading data from {data_path}")
    data = joblib.load(data_path)
    
    X_test = data['X_test']
    y_test = data['y_test']
    
    print(f"\nLoaded model and data")
    print(f"  Test samples: {len(X_test):,}")
    print(f"  Features: {X_test.shape}")
    print(f"  Genres: {len(label_encoder.classes_)}")
    
    # Make predictions
    print("\n" + "="*80)
    print("Computing predictions...")
    print("="*80)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # Overall metrics
    test_acc = accuracy_score(y_test, y_pred)
    test_f1_macro = f1_score(y_test, y_pred, average='macro')
    test_f1_weighted = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\nOverall Test Performance:")
    print(f"  Accuracy:        {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  F1 (macro):      {test_f1_macro:.4f}")
    print(f"  F1 (weighted):   {test_f1_weighted:.4f}")
    
    # Per-class metrics
    print("\n" + "="*80)
    print("Per-Class Metrics:")
    print("="*80)
    
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average=None
    )
    
    metrics_df = pd.DataFrame({
        'Genre': label_encoder.classes_,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'Support': support
    })
    
    metrics_df = metrics_df.sort_values('F1-Score', ascending=False)
    print("\n", metrics_df.to_string(index=False))
    
    # Confusion matrix
    print("\n" + "="*80)
    print("Confusion Matrix:")
    print("="*80)
    
    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    print(f"\nTop 5 most confused pairs:")
    confused_pairs = []
    for i in range(len(cm)):
        for j in range(len(cm)):
            if i != j:
                confused_pairs.append((
                    label_encoder.classes_[i],
                    label_encoder.classes_[j],
                    cm[i, j],
                    cm_normalized[i, j]
                ))
    
    confused_pairs.sort(key=lambda x: x, reverse=True)
    for true_genre, pred_genre, count, pct in confused_pairs[:5]:
        print(f"  {true_genre} → {pred_genre}: {count} ({pct*100:.1f}%)")
    
    # Feature importance
    print("\n" + "="*80)
    print("Feature Importance (Top 10):")
    print("="*80)
    
    feature_importance = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': [f'Feature_{i}' for i in range(len(feature_importance))],
        'Importance': feature_importance
    }).sort_values('Importance', ascending=False)
    
    print("\n", importance_df.head(10).to_string(index=False))
    
    # Cleanlab label quality analysis
    print("\n" + "="*80)
    print("Cleanlab Label Quality Analysis:")
    print("="*80)
    
    try:
        from cleanlab.filter import find_label_issues
        
        # Find label issues using Cleanlab 2.x API
        label_issues_mask = find_label_issues(
            labels=y_test,
            pred_probs=y_proba,
            return_indices_ranked_by='self_confidence'
        )
        
        num_issues = sum(label_issues_mask)
        pct_issues = (num_issues / len(y_test)) * 100
        
        print(f"\nFound {num_issues} potential label issues ({pct_issues:.1f}%)")
        
        # Show examples of label issues
        if num_issues > 0:
            issue_indices = np.where(label_issues_mask)[0][:10]
            
            print(f"\nTop 10 potential label issues:")
            print(f"{'Index':<8} {'True Genre':<20} {'Predicted Genre':<20} {'Confidence':<12}")
            print("-" * 60)
            
            for idx in issue_indices:
                true_genre = label_encoder.classes_[y_test[idx]]
                pred_genre = label_encoder.classes_[y_pred[idx]]
                confidence = y_proba[idx][y_pred[idx]]
                print(f"{idx:<8} {true_genre:<20} {pred_genre:<20} {confidence:<12.4f}")
        
    except Exception as e:
        print(f"\nCleanlab analysis failed: {e}")
        
        num_issues = sum(label_issues)
        pct_issues = (num_issues / len(y_test)) * 100
        
        print(f"\nFound {num_issues} potential label issues ({pct_issues:.1f}%)")
        
        # Show examples of label issues
        if num_issues > 0:
            issue_indices = np.where(label_issues)[:10]
            
            print(f"\nTop 10 potential label issues:")
            print(f"{'Index':<8} {'True Genre':<20} {'Predicted Genre':<20} {'Confidence':<12}")
            print("-" * 60)
            
            for idx in issue_indices:
                true_genre = label_encoder.classes_[y_test[idx]]
                pred_genre = label_encoder.classes_[y_pred[idx]]
                confidence = y_proba[idx][y_pred[idx]]
                print(f"{idx:<8} {true_genre:<20} {pred_genre:<20} {confidence:<12.4f}")
        
    except Exception as e:
        print(f"\nCleanlab analysis failed: {e}")
    
    # Data slicing: High vs Low confidence
    print("\n" + "="*80)
    print("Data Slicing (Confidence-based):")
    print("="*80)
    
    max_probs = np.max(y_proba, axis=1)
    high_confidence_mask = max_probs > 0.7
    low_confidence_mask = max_probs <= 0.7
    
    high_conf_acc = accuracy_score(y_test[high_confidence_mask], y_pred[high_confidence_mask])
    low_conf_acc = accuracy_score(y_test[low_confidence_mask], y_pred[low_confidence_mask])
    
    high_conf_f1 = f1_score(y_test[high_confidence_mask], y_pred[high_confidence_mask], average='macro')
    low_conf_f1 = f1_score(y_test[low_confidence_mask], y_pred[low_confidence_mask], average='macro')
    
    print(f"\nHigh Confidence (>70%):")
    print(f"  Samples: {sum(high_confidence_mask):,}")
    print(f"  Accuracy: {high_conf_acc:.4f} ({high_conf_acc*100:.2f}%)")
    print(f"  F1 (macro): {high_conf_f1:.4f}")
    
    print(f"\nLow Confidence (≤70%):")
    print(f"  Samples: {sum(low_confidence_mask):,}")
    print(f"  Accuracy: {low_conf_acc:.4f} ({low_conf_acc*100:.2f}%)")
    print(f"  F1 (macro): {low_conf_f1:.4f}")
    
    # Save results
    if save_results:
        print("\n" + "="*80)
        print("Saving Results:")
        print("="*80)
        
        results = {
            'overall_metrics': {
                'accuracy': test_acc,
                'f1_macro': test_f1_macro,
                'f1_weighted': test_f1_weighted
            },
            'per_class_metrics': metrics_df,
            'confusion_matrix': cm,
            'feature_importance': importance_df,
            'data_slicing': {
                'high_confidence_accuracy': high_conf_acc,
                'low_confidence_accuracy': low_conf_acc
            }
        }
        
        results_path = RESULTS_DIR / 'evaluation_results.pkl'
        joblib.dump(results, results_path)
        print(f"\nResults saved to {results_path}")
        
        # Save metrics CSV
        metrics_csv_path = RESULTS_DIR / 'per_class_metrics.csv'
        metrics_df.to_csv(metrics_csv_path, index=False)
        print(f"Per-class metrics saved to {metrics_csv_path}")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    app()  # Initialize Typer CLI