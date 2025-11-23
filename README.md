# Music Genre Classification with Distributed Computing and MLOps

**Course**: PSMDSBE 102: Special Topics in Data Science  
**Institution**: Technological Institute of the Philippines (TIP)  
**Students**: Daryll Jose, Martin Basbacio  
**Date**: November 2025

---

## Contact Information

- **Daryll Jose**
  - Email: [qdajose@tip.edu.ph](mailto:qdajose@tip.edu.ph)
  - GitHub: [@djoseprogramming](https://github.com/djoseprogramming)

- **Martin Basbacio**
  - Email: [qmlsbasbacio01@tip.edu.ph](mailto:qmlsbasbacio01@tip.edu.ph)
  - GitHub: [@mahteenbash](https://github.com/mahteenbash)

---

## Course Overview

**PSMDSBE 102: Special Topics in Data Science** covers advanced topics including:
- **Activity 1**: MLOps Design & Architecture
- **Activity 2**: Data Engineering with Ray Distributed Computing
- **Activity 3**: Experiment Tracking (MLflow), Hyperparameter Tuning, and Model Serving
- **Activity 4**: Software Engineering (Refactoring, CLI Development, Stateless Functions)

This project demonstrates **end-to-end MLOps pipeline** development using the FMA (Free Music Archive) dataset.

---

## Project Summary

### Objective
Build a production-grade music genre classification system that:
- Processes 8,000 audio tracks efficiently using distributed computing (Ray)
&nbsp;&nbsp;1. [fma_small](https://os.unil.cloud.switch.ch/fma/fma_small.zip)  
&nbsp;&nbsp;2. [fma_medium](https://os.unil.cloud.switch.ch/fma/fma_medium.zip)  
&nbsp;&nbsp;3. [fma_metadata](https://os.unil.cloud.switch.ch/fma/fma_metadata.zip)
- Extracts 89 audio features per track (MFCCs, spectral features, tempo, etc.)
- Trains LightGBM classifier with comprehensive evaluation
- Provides CLI interface for inference and analysis
- Tracks all experiments with MLflow
- Demonstrates MLOps best practices

### Key Results (Actual Execution)
- **Test Accuracy**: 54.33% (8 balanced genres)
- **Macro F1-Score**: 0.5405 (fair evaluation across all genres)
- **Training Accuracy**: 99.84% (indicates some overfitting)
- **Validation Accuracy**: 55.83%
- **Best Iteration**: 117 (early stopping prevented further overfitting)
- **Processing Speed**: ~20-25 minutes for 8,000 tracks (with Ray parallelization)
- **Model Training**: ~5 minutes with LightGBM
- **Hyperparameter Tuning**: ~40 seconds (5 trials completed)
- **Feature Extraction**: 89 features per track (20 MFCCs, 12 chroma, 6 spectral contrast, tempo, RMS energy, etc.)

---

## Project Architecture

### Directory Structure
```
psmdsbe102_2526A/
├── README.md                      # This file
├── requirements.txt               # Python dependencies
├── config.py                      # Centralized configuration
├── data.py                        # Data loading & feature extraction (Ray)
├── train.py                       # Model training with MLflow
├── tune.py                        # Hyperparameter tuning (Ray Tune)
├── evaluate.py                    # Comprehensive evaluation
├── predict.py                     # Single song prediction (CLI)
├── utils.py                       # Shared utility functions
│
├── fma_small/                     # Audio files (8,000 tracks, 8 genres, balanced)
├── fma_medium/                    # Audio files (25,000 tracks, 16 genres, unbalanced)
├── fma_metadata/                  # FMA metadata CSV files
├── preprocessed_features/         # Extracted features cache
├── saved_models/                  # Trained models & artifacts
├── results/                       # Evaluation results
└── mlruns/                        # MLflow experiment tracking
```

### Genres (8 total, perfectly balanced)
For this documentation, we will use the ```fma_small``` dataset.

&nbsp;1. Hip-Hop  
&nbsp;2. Pop  
&nbsp;3. Folk  
&nbsp;4. Experimental  
&nbsp;5. Rock  
&nbsp;6. International  
&nbsp;7. Electronic  
&nbsp;8. Instrumental  

(1,000 tracks each = 8,000 total, perfectly balanced dataset)

---

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Extract Features (Ray Distributed)
```bash
python data.py
```
- Loads FMA small data and metadata
- Extracts 89 features from 8,000 tracks using Ray parallelization
- Splits into train/val/test (70/15/15)
- Saves to: `preprocessed_features/preprocessed_data_ray.pkl`
- **Time**: ~20-25 minutes

### 3. Train Model with MLflow Tracking
```bash
python train.py
```
- Trains LightGBM with default hyperparameters
- Logs metrics to MLflow: accuracy, F1-macro, F1-weighted
- Saves model, scaler, label encoder
- **Time**: ~5 minutes

### 4. (Optional) Hyperparameter Tuning
```bash
python tune.py
```
- Ray Tune distributed search over 5 hyperparameter configurations
- ASHA scheduler for early stopping
- Saves best configuration
- **Time**: ~40 seconds (5 trials)

### 5. Train with Tuned Hyperparameters
```bash
python train.py --use-tuned-params
```
- Uses best hyperparameters from Ray Tune (if available)
- Falls back to defaults if tuning not completed
- **Time**: ~5 minutes

### 6. Evaluate Model Comprehensively
```bash
python evaluate.py
```
- Per-class metrics (precision, recall, F1)
- Confusion matrix analysis
- Feature importance (Top 10)
- Cleanlab label quality analysis
- Data slicing (confidence-based: >70% vs ≤70%)
- Saves results to: `results/evaluation_results.pkl` and `results/per_class_metrics.csv`
- **Time**: ~3 minutes

### 7. Predict Genre for Any Song
```bash
python predict.py --audio-path "path/to/song.mp3"
```
- Extracts features from single audio file
- Predicts genre with confidence
- Shows top 5 predictions
- **Time**: ~2 seconds

### 8. View Experiment Results
```bash
mlflow ui
```
- Open: http://127.0.0.1:5000
- View all training runs, metrics, parameters, saved models

---

## Actual Execution Results

### Training Results (Default Hyperparameters)
```
Train: Accuracy=99.84%, F1(macro)=0.9984
Val:   Accuracy=55.83%, F1(macro)=0.5556
Test:  Accuracy=54.33%, F1(macro)=0.5405
Best iteration: 117
Training time: 0.0 minutes
```

### Hyperparameter Tuning Results (5 Trials)
```
Trial 0: num_leaves=63, learning_rate=0.01 → Accuracy=0.5558, F1=0.5540
Trial 1: num_leaves=63, learning_rate=0.1  → Accuracy=0.5408, F1=0.5391
Trial 2: num_leaves=31, learning_rate=0.05 → Accuracy=0.5483, F1=0.5465
Trial 3: num_leaves=31, learning_rate=0.05 → Accuracy=0.5475, F1=0.5471
Trial 4: num_leaves=31, learning_rate=0.1  → Accuracy=0.5517, F1=0.5521

Best Trial: num_leaves=63, learning_rate=0.01 (Accuracy=0.5558)
```

### Training with Tuned Hyperparameters
```
Train: Accuracy=99.84%, F1(macro)=0.9984
Val:   Accuracy=55.83%, F1(macro)=0.5556
Test:  Accuracy=54.33%, F1(macro)=0.5405
Best iteration: 117
```

### Evaluation Results

**Overall Test Performance:**
- Accuracy: 54.33%
- Macro F1-Score: 0.5405
- Weighted F1-Score: 0.5405

**Per-Class Metrics:**

| Genre | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| Folk | 0.6335 | 0.6800 | 0.6559 | 150 |
| International | 0.5960 | 0.6000 | 0.5980 | 150 |
| Hip-Hop | 0.5697 | 0.6267 | 0.5968 | 150 |
| Rock | 0.5750 | 0.6133 | 0.5935 | 150 |
| Electronic | 0.5714 | 0.5333 | 0.5517 | 150 |
| Instrumental | 0.5223 | 0.5467 | 0.5342 | 150 |
| Experimental | 0.4961 | 0.4200 | 0.4549 | 150 |
| Pop | 0.3525 | 0.3267 | 0.3391 | 150 |

**Top 5 Most Confused Genre Pairs:**
- Rock → Pop: 16 occurrences (10.7%)
- Rock → International: 12 occurrences (8.0%)
- Rock → Folk: 7 occurrences (4.7%)
- Rock → Instrumental: 5 occurrences (3.3%)
- Rock → Hip-Hop: 5 occurrences (3.3%)

**Feature Importance (Top 10):**  
&nbsp;1. Feature_0: 560  
&nbsp;2. Feature_64: 542  
&nbsp;3. Feature_88: 493  
&nbsp;4. Feature_3: 466  
&nbsp;5. Feature_87: 456  
&nbsp;6. Feature_71: 437  
&nbsp;7. Feature_77: 437  
&nbsp;8. Feature_20: 420  
&nbsp;9. Feature_2: 417  
&nbsp;10. Feature_65: 412

**Data Slicing by Confidence:**
- High Confidence (>70%): 380 samples, Accuracy=79.74%, F1(macro)=0.7383
- Low Confidence (≤70%): 820 samples, Accuracy=42.56%, F1(macro)=0.4271

**Cleanlab Label Quality Analysis:**
- Found 241,632 potential label issues
- Top issue examples with misclassifications found

---

## File Descriptions

### config.py - Centralized Configuration
**Purpose**: Single source of truth for all constants, paths, and settings.

**Key Features**:
- Project paths (DATA_DIR, METADATA_DIR, FEATURES_DIR, MODELS_DIR, RESULTS_DIR)
- Data settings (SAMPLE_RATE=22050, AUDIO_DURATION=30, feature counts)
- Model hyperparameters (LGBM_PARAMS with default values)
- Ray settings (RAY_NUM_CPUS=6, RAY_NUM_TRIALS=5, RAY_MAX_CONCURRENT=2)
- MLflow URI and experiment name
- Utility functions (set_seed, get_audio_path)

**Principle**: No hardcoded values in other scripts; all pulled from here.

---

### data.py - Data Preparation & Feature Extraction
**Purpose**: Load FMA metadata, extract features, prepare datasets.

**CLI Command**: `python data.py --subset small`

**What It Does**:  
&nbsp;1. Loads FMA tracks.csv and genres.csv  
&nbsp;2. Filters to 8,000 balanced tracks (1,000 per genre)  
&nbsp;3. Validates audio file existence (verified: 8,000 valid files)  
&nbsp;4. Parallel feature extraction (Ray distributed):  
&nbsp;&nbsp;- MFCCs (mean + std): 40 features  
&nbsp;&nbsp;- Chroma (mean + std): 24 features  
&nbsp;&nbsp;- Spectral contrast (mean + std): 12 features  
&nbsp;&nbsp;- Zero crossing rate: 2 features  
&nbsp;&nbsp;- Spectral centroid, bandwidth, rolloff: 6 features  
&nbsp;&nbsp;- Tempo: 1 feature  
&nbsp;&nbsp;- RMS energy (mean + std): 2 features  
&nbsp;&nbsp;- Total: 89 features per track  
&nbsp;5. Stratified train/val/test split (70/15/15)  
&nbsp;&nbsp;- Train: 5,597 samples  
&nbsp;&nbsp;- Val: 1,200 samples  
&nbsp;&nbsp;- Test: 1,200 samples  
&nbsp;6. Feature scaling (StandardScaler)  
&nbsp;7. Label encoding  
&nbsp;8. Saves to: `preprocessed_data_ray.pkl`

**Output**: X_train, X_val, X_test with shape (5597, 89), (1200, 89), (1200, 89)

**Key Technology**: Ray @remote decorator for parallel processing

---

### train.py - Model Training
**Purpose**: Train LightGBM classifier with MLflow experiment tracking.

**CLI Command**: 
- `python train.py` (default hyperparameters)
- `python train.py --use-tuned-params` (tuned hyperparameters)

**What It Does**:  
&nbsp;1. Loads preprocessed data (5,597 training samples)  
&nbsp;2. Optionally loads tuned hyperparameters from Ray Tune  
&nbsp;3. Trains LightGBM with early stopping (20 rounds patience)  
&nbsp;4. MLflow logging:  
&nbsp;&nbsp;- All hyperparameters  
&nbsp;&nbsp;- Train/val/test metrics (accuracy, F1-macro, F1-weighted)  
&nbsp;&nbsp;- Best iteration number (117)  
&nbsp;&nbsp;- Training time  
&nbsp;5. Saves artifacts:  
&nbsp;&nbsp;- `lightgbm_model_ray.pkl`  
&nbsp;&nbsp;- `scaler_ray.pkl`  
&nbsp;&nbsp;- `label_encoder_ray.pkl`  
&nbsp;&nbsp;- `model_artifacts.pkl` (all-in-one)

**Key Features**: Early stopping at iteration 117, MLflow integration

---

### tune.py - Hyperparameter Optimization
**Purpose**: Distributed hyperparameter search using Ray Tune.

**CLI Command**: `python tune.py`

**What It Does**:  
&nbsp;1. Defines hyperparameter search space:  
&nbsp;&nbsp;- num_leaves: [31, 63]  
&nbsp;&nbsp;- learning_rate: [0.01, 0.05, 0.1]  
&nbsp;&nbsp;- feature_fraction: [0.6-0.74]  
&nbsp;&nbsp;- bagging_fraction: [0.61-0.84]  
&nbsp;&nbsp;- min_child_samples: [10, 30]  
&nbsp;2. Uses Ray Tune with AsyncHyperBand scheduler  
&nbsp;3. Runs 5 sequential trials (2 max concurrent)  
&nbsp;4. Optimizes on validation F1-macro score  
&nbsp;5. Saves best configuration (Trial 0: num_leaves=63, learning_rate=0.01)

**Results**: Best trial achieved 55.58% accuracy

**Key Technology**: Ray Tune + BasicVariantGenerator + AsyncHyperBand scheduler

---

### evaluate.py - Comprehensive Model Evaluation
**Purpose**: Detailed analysis of model performance and data quality.

**CLI Command**: `python evaluate.py`

**What It Does**:  
&nbsp;1. **Overall Metrics**: Test accuracy 54.33%, F1-macro 0.5405  
&nbsp;2. **Per-Class Analysis**: Individual precision, recall, F1 for each of 8 genres  
&nbsp;&nbsp;3. **Confusion Matrix**: Top confusions are Rock→Pop (10.7%), Rock→International (8.0%)  
&nbsp;4. **Feature Importance**: Top 10 features identified (Feature_0, Feature_64, etc.)  
&nbsp;5. **Cleanlab Analysis**: Identifies potential data quality issues  
&nbsp;6. **Data Slicing**:  
&nbsp;&nbsp;- High confidence predictions (>70%): 380 samples, 79.74% accuracy  
&nbsp;&nbsp;- Low confidence predictions (≤70%): 820 samples, 42.56% accuracy  
&nbsp;7. **Outputs**:  
&nbsp;&nbsp;- `evaluation_results.pkl`  
&nbsp;&nbsp;- `per_class_metrics.csv`  

**Key Features**: Production-grade evaluation with confidence-based analysis

---

### predict.py - Single Song Inference
**Purpose**: CLI interface for genre prediction on new audio files.

**CLI Command**: `python predict.py --audio-path "song.mp3" --top-n 5`

**What It Does**:
&nbsp;1. Loads trained model, scaler, label encoder from `saved_models/`  
&nbsp;2. Extracts 89 features from audio file (same as training pipeline)  
&nbsp;3. Scales features using saved scaler  
&nbsp;4. Predicts genre probabilities  
&nbsp;5. Displays:  
&nbsp;&nbsp;- Predicted genre + confidence  
&nbsp;&nbsp;- Top N predictions with confidence bars  
&nbsp;6. Error handling for missing files or feature extraction failures

**Example Output**:
```
Prediction Results
========================================
  File: Mozart - Lacrimosa.mp3
  Predicted Genre: Instrumental
  Confidence: 84.46%

  Top Predictions:
    Instrumental    [████████████████████████████████] 84.46%
    Experimental    [█████] 10.42%
    Electronic      [█] 2.97%
    International   [ ] 0.72%
    Folk            [ ] 0.68%
========================================
```

---

### utils.py - Utility Functions
**Purpose**: Shared helper functions used across multiple scripts.

**Functions**:
- `set_seed()`: Set random seeds for reproducibility
- `save_pickle()`: Save objects to pickle files
- `load_pickle()`: Load objects from pickle files
- `ensure_dir()`: Create directories if missing

**Principle**: Avoid circular imports by centralizing shared code

---

## Technologies Used

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Audio Processing** | librosa | Extract MFCCs, spectral features, tempo |
| **Data Processing** | pandas, numpy, scikit-learn | Data manipulation, scaling, encoding |
| **Distributed Computing** | Ray | Parallel feature extraction (4-8 CPUs) |
| **Model Training** | LightGBM | Gradient boosting classifier |
| **Hyperparameter Tuning** | Ray Tune | Distributed hyperparameter search |
| **Experiment Tracking** | MLflow | Log parameters, metrics, models |
| **Data Quality** | Cleanlab | Identify potential label issues |
| **CLI** | Typer | User-friendly command-line interface |
| **Configuration** | Python module | Centralized settings management |

---

## MLOps Principles Demonstrated

### 1. Stateless Workloads
- All functions take explicit inputs (no global state)
- Reproducible with seed control (RANDOM_SEED = 42)
- Example: `prepare_data(subset='small')` vs implicit globals

### 2. Configuration Management
- Centralized `config.py` (single source of truth)
- No hardcoded paths or hyperparameters
- Easy to switch datasets, models, or settings
- All changes in one file update entire system

### 3. Experiment Tracking
- MLflow logs all parameters and metrics
- Model versioning with run IDs
- Reproducible runs with complete artifact storage
- Two training runs performed: Default params and Tuned params

### 4. Distributed Computing
- Ray parallelizes feature extraction
- Scales from 1 to 100+ CPUs
- Fault-tolerant distributed tasks
- Extracted 8,000 tracks in ~20-25 minutes

### 5. CLI Interface
- Typer provides user-friendly commands
- Type hints and help documentation
- Production-ready command-line tools
- 5 main CLI scripts: data.py, train.py, tune.py, evaluate.py, predict.py

### 6. Modular Design
- Separate concerns: data, training, tuning, evaluation, inference
- No circular dependencies
- Easy to test, maintain, and extend
- Config module shared across all scripts

---

## Configuration & Customization

### Change Dataset
Edit `config.py`:
```python
DATA_DIR = PROJECT_ROOT / "fma_small"  # Or "fma_medium"
MAX_TRACKS = 8000  # Or 25000
```

### Increase CPU Count
Edit `config.py`:
```python
RAY_NUM_CPUS = 8

### Adjust Hyperparameters
Edit `config.py`:
```python
LGBM_PARAMS = {
    'learning_rate': 0.05,  # Adjust learning rate
    'num_leaves': 31,       # Adjust tree depth
    ...
}
```

### Change MLflow Experiment
Edit `config.py` or pass CLI argument:
```bash
python train.py --experiment-name "my-custom-experiment"
```

---

## Troubleshooting

### Ray Initialization Error
```bash
ray stop
python data.py  # Will reinitialize Ray
```

### Librosa Installation Issues
Ensure FFmpeg is installed:
```bash
# macOS
brew install ffmpeg

# Windows (with conda)
conda install -c conda-forge ffmpeg

# Linux
sudo apt-get install ffmpeg
```

### Module Not Found Error
```bash
export PYTHONPATH=$PYTHONPATH:$PWD
python train.py
```

### MLflow UI Not Accessible
```bash
mlflow ui --host 127.0.0.1 --port 5000
```

## Execution Summary

### Completed Steps
&nbsp;1. Data preprocessing: Extracted 89 features from 8,000 tracks using Ray  
&nbsp;2. Model training (default): Trained LightGBM, achieved 54.33% test accuracy  
&nbsp;3. Hyperparameter tuning: Ran 5 trials, best config saved  
&nbsp;4. Model training (tuned): Re-trained with best hyperparameters  
&nbsp;5. Evaluation: Comprehensive analysis with Cleanlab and data slicing  
&nbsp;6. Results saved: model_artifacts.pkl, evaluation_results.pkl, per_class_metrics.csv  


---

## License

This project is submitted as coursework for **PSMDSBE 102: Special Topics in Data Science**.

---

Last Updated: November 23, 2025  
Execution Date: November 23, 2025  