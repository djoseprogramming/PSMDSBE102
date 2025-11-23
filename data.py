"""
Data Loading and Preprocessing Module

This module handles all data-related operations including:
- Loading FMA metadata
- Filtering tracks by genre
- Parallel feature extraction using Ray
- Train/val/test splitting
- Feature scaling and preprocessing
"""

import time
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
import librosa
import ray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from tqdm import tqdm
import joblib
import typer
from typing_extensions import Annotated

from config import (
    DATA_DIR, METADATA_DIR, FEATURES_DIR, SAMPLE_RATE, AUDIO_DURATION,
    N_MFCC, N_CHROMA, N_SPECTRAL_CONTRAST, MIN_SAMPLES_PER_GENRE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO, RANDOM_SEED, RAY_NUM_CPUS,
    get_audio_path, set_seed
)


# Initialize Typer app
app = typer.Typer()


@ray.remote
def extract_audio_features_ray(
    track_id: int,
    audio_path: Path,
    sr: int = SAMPLE_RATE,
    duration: int = AUDIO_DURATION
) -> Tuple[int, Optional[np.ndarray], Optional[str]]:
    """
    Ray remote function to extract audio features in parallel.
    
    Extracts 89 features from an audio file:
    - 40 MFCC features (20 mean + 20 std)
    - 24 Chroma features (12 mean + 12 std)
    - 14 Spectral contrast features (7 mean + 7 std)
    - 2 Zero crossing rate features (mean + std)
    - 2 Spectral centroid features (mean + std)
    - 2 Spectral bandwidth features (mean + std)
    - 2 Spectral rolloff features (mean + std)
    - 1 Tempo feature
    - 2 RMS energy features (mean + std)
    
    Args:
        track_id: Track ID
        audio_path: Path to audio file
        sr: Sample rate
        duration: Duration in seconds
        
    Returns:
        Tuple of (track_id, features_array, error_message)
    """
    try:
        # Load audio
        y, sr = librosa.load(str(audio_path), sr=sr, duration=duration, mono=True)
        
        features = []
        
        # 1. MFCCs (20 coefficients)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        
        # 2. Chroma (12 features)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_chroma=N_CHROMA)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        
        # 3. Spectral contrast (6 bands)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_bands=N_SPECTRAL_CONTRAST)
        features.extend(np.mean(spectral_contrast, axis=1))
        features.extend(np.std(spectral_contrast, axis=1))
        
        # 4. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        
        # 5. Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
        
        # 6. Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        
        # 7. Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        
        # 8. Tempo (handle array return)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        features.append(tempo)
        
        # 9. RMS energy
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])
        
        return track_id, np.array(features, dtype=np.float32), None
        
    except Exception as e:
        return track_id, None, str(e)


def load_metadata(
    metadata_dir: Path = METADATA_DIR,
    data_dir: Path = DATA_DIR,
    subset: str = "small",
    min_samples: int = MIN_SAMPLES_PER_GENRE
) -> Tuple[pd.DataFrame, List[int], List[str], List[str]]:
    """
    Load and filter FMA metadata.
    
    Args:
        metadata_dir: Directory containing metadata files
        data_dir: Directory containing audio files
        subset: Which FMA subset to use ('small', 'medium', or 'large')
        min_samples: Minimum samples required per genre
        
    Returns:
        Tuple of (tracks_dataframe, track_ids, genre_labels, genre_names)
    """
    print("Loading FMA metadata...")
    
    # Load tracks metadata (multi-level columns)
    tracks_path = metadata_dir / 'tracks.csv'
    tracks = pd.read_csv(tracks_path, index_col=0, header=[0, 1])
    
    print(f"Loaded metadata for {len(tracks):,} tracks")
    
    # Filter to specified subset
    subset_tracks = tracks[tracks[('set', 'subset')] == subset].copy()
    print(f"Filtered to {len(subset_tracks):,} tracks in '{subset}' subset")
    
    # Get tracks with valid top-level genres
    tracks_with_genre = subset_tracks[subset_tracks[('track', 'genre_top')].notna()].copy()
    print(f"Tracks with genre labels: {len(tracks_with_genre):,}")
    
    # Genre distribution
    genre_counts = tracks_with_genre[('track', 'genre_top')].value_counts()
    print(f"\nGenre distribution ({len(genre_counts)} total genres):")
    for genre, count in genre_counts.head(10).items():
        print(f"  {genre:20s} {count:5,}")
    
    # Filter genres with at least min_samples
    valid_genres = genre_counts[genre_counts >= min_samples].index.tolist()
    tracks_filtered = tracks_with_genre[
        tracks_with_genre[('track', 'genre_top')].isin(valid_genres)
    ].copy()
    
    print(f"\nKept {len(valid_genres)} genres with >= {min_samples} samples")
    print(f"Final dataset: {len(tracks_filtered):,} tracks")
    print(f"Genres: {', '.join(sorted(valid_genres))}")
    
    # Verify audio files exist
    print("\nVerifying audio files...")
    track_ids = tracks_filtered.index.tolist()
    genre_labels = tracks_filtered[('track', 'genre_top')].tolist()
    
    valid_track_ids = []
    valid_genre_labels = []
    
    for track_id, genre in tqdm(zip(track_ids, genre_labels), total=len(track_ids), desc="Checking files"):
        audio_path = get_audio_path(track_id, data_dir)
        if audio_path.exists():
            valid_track_ids.append(track_id)
            valid_genre_labels.append(genre)
    
    print(f"Found {len(valid_track_ids):,} valid audio files")
    
    return tracks_filtered, valid_track_ids, valid_genre_labels, valid_genres


@app.command()
def prepare_data(
    subset: Annotated[str, typer.Option(help="FMA subset to use")] = "small",
    output_file: Annotated[str, typer.Option(help="Output pickle file name")] = "preprocessed_data_ray.pkl",
    num_cpus: Annotated[int, typer.Option(help="Number of CPUs for Ray")] = RAY_NUM_CPUS
) -> None:
    """
    Main data preparation workload: load metadata, extract features, split data.
    
    This is a stateless function that performs the complete data preparation pipeline:
    1. Load FMA metadata and filter by genre
    2. Extract audio features in parallel using Ray
    3. Split into train/val/test sets
    4. Scale features and encode labels
    5. Save preprocessed data
    """
    print("="*80)
    print("DATA PREPARATION WORKLOAD")
    print("="*80)
    
    set_seed()
    
    # Initialize Ray
    ray.shutdown()  # Shutdown any existing instance
    ray.init(num_cpus=num_cpus, ignore_reinit_error=True)
    print(f"\nRay initialized with {num_cpus} CPUs")
    
    # Load metadata
    tracks, track_ids, genre_labels, genre_names = load_metadata(subset=subset)
    
    # Extract features using Ray
    print(f"\n{'='*80}")
    print(f"Extracting features from {len(track_ids):,} tracks using Ray...")
    print(f"Parallel workers: {num_cpus}")
    print(f"This will take 20-40 minutes with parallel processing.\n")
    
    start_time = time.time()
    
    # Create Ray tasks
    futures = []
    for track_id, genre in zip(track_ids, genre_labels):
        audio_path = get_audio_path(track_id)
        if audio_path.exists():
            future = extract_audio_features_ray.remote(track_id, audio_path)
            futures.append((future, genre))
    
    # Collect results with progress bar
    features_list = []
    labels_list = []
    successful_ids = []
    failed_count = 0
    
    for future, genre in tqdm(futures, desc="Extracting features"):
        track_id, features, error = ray.get(future)
        
        if features is not None:
            features_list.append(features)
            labels_list.append(genre)
            successful_ids.append(track_id)
        else:
            failed_count += 1
    
    # Convert to arrays
    X = np.array(features_list)
    y_str = np.array(labels_list)
    
    elapsed_time = time.time() - start_time
    
    print(f"\nSuccessfully extracted features from {len(successful_ids):,} tracks")
    print(f"✗ Failed: {failed_count} tracks")
    print(f"Feature shape: {X.shape}")
    print(f"Time elapsed: {elapsed_time/60:.1f} minutes")
    print(f"Average time per track: {elapsed_time/len(successful_ids):.2f} seconds")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_str)
    
    print(f"\nEncoded {len(label_encoder.classes_)} genres:")
    for i, genre in enumerate(label_encoder.classes_):
        print(f"  {i}: {genre}")
    
    # Split data
    X_temp, X_test, y_temp, y_test, ids_temp, ids_test = train_test_split(
        X, y, successful_ids, test_size=TEST_RATIO, stratify=y, random_state=RANDOM_SEED
    )
    
    val_size_adjusted = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
    X_train, X_val, y_train, y_val, ids_train, ids_val = train_test_split(
        X_temp, y_temp, ids_temp, test_size=val_size_adjusted, stratify=y_temp, random_state=RANDOM_SEED
    )
    
    print(f"\nData split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Val:   {len(X_val):,} samples")
    print(f"  Test:  {len(X_test):,} samples")
    
    # Scale features
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nFeatures scaled (StandardScaler)")
    
    # Save preprocessed data
    preprocessed_data = {
        'X_train': X_train_scaled, 'y_train': y_train, 'ids_train': ids_train,
        'X_val': X_val_scaled, 'y_val': y_val, 'ids_val': ids_val,
        'X_test': X_test_scaled, 'y_test': y_test, 'ids_test': ids_test,
        'label_encoder': label_encoder, 'scaler': scaler,
        'genre_names': label_encoder.classes_.tolist()
    }
    
    save_path = FEATURES_DIR / output_file
    joblib.dump(preprocessed_data, save_path)
    print(f"Saved to {save_path}")
    
    # Shutdown Ray
    ray.shutdown()
    print("\nRay shutdown complete")
    print("\n" + "="*80)
    print("DATA PREPARATION COMPLETE!")
    print("="*80)


if __name__ == "__main__":
    if ray.is_initialized():
        ray.shutdown()
    ray.init()
    app()  # Initialize Typer CLI
