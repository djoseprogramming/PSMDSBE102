import typer
from pathlib import Path
import joblib
import numpy as np
import librosa

app = typer.Typer()

# ====== Constants - match your model training ======
N_MFCC = 20
N_CHROMA = 12
N_SPECTRAL_CONTRAST = 6
SAMPLE_RATE = 22050
AUDIO_DURATION = 30

def extract_features_single_song(audio_path: Path, sr=SAMPLE_RATE, duration=AUDIO_DURATION):
    """
    Extract features from a single audio file (same as training pipeline).
    """
    try:
        y, sr_loaded = librosa.load(str(audio_path), sr=sr, duration=duration, mono=True)
        features = []
        # 1. MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr_loaded, n_mfcc=N_MFCC)
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        # 2. Chroma
        chroma = librosa.feature.chroma_stft(y=y, sr=sr_loaded, n_chroma=N_CHROMA)
        features.extend(np.mean(chroma, axis=1))
        features.extend(np.std(chroma, axis=1))
        # 3. Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr_loaded, n_bands=N_SPECTRAL_CONTRAST)
        features.extend(np.mean(spectral_contrast, axis=1))
        features.extend(np.std(spectral_contrast, axis=1))
        # 4. Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)
        features.extend([np.mean(zcr), np.std(zcr)])
        # 5. Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr_loaded)
        features.extend([np.mean(spectral_centroid), np.std(spectral_centroid)])
        # 6. Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr_loaded)
        features.extend([np.mean(spectral_bandwidth), np.std(spectral_bandwidth)])
        # 7. Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr_loaded)
        features.extend([np.mean(spectral_rolloff), np.std(spectral_rolloff)])
        # 8. Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr_loaded)
        if isinstance(tempo, np.ndarray):
            tempo = float(tempo[0]) if len(tempo) > 0 else 120.0
        else:
            tempo = float(tempo)
        features.append(tempo)
        # 9. RMS energy
        rms = librosa.feature.rms(y=y)
        features.extend([np.mean(rms), np.std(rms)])
        return np.array(features, dtype=np.float32)
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

@app.command()
def predict(
    audio_path: str = typer.Option(..., help="Path to .mp3 audio file"),
    top_n: int = typer.Option(5, help="Number of top predictions to show"),
    model_path: str = typer.Option("saved_models/lightgbm_model_ray.pkl", help="Path to trained LightGBM model"),
    scaler_path: str = typer.Option("saved_models/scaler_ray.pkl", help="Path to fitted scaler"),
    encoder_path: str = typer.Option("saved_models/label_encoder_ray.pkl", help="Path to fitted label encoder")
):
    """
    Predict music genre for a given song using a trained LightGBM model.
    """
    # Check file exists
    if not Path(audio_path).exists():
        print(f"File not found: {audio_path}")
        raise typer.Exit(code=1)
    # Load model, scaler, and label encoder
    try:
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        label_encoder = joblib.load(encoder_path)
    except Exception as e:
        print(f"Could not load model/scaler/encoder: {e}")
        raise typer.Exit(code=1)
    # Process features
    features = extract_features_single_song(Path(audio_path))
    if features is None:
        print("Error: Could not extract audio features.")
        raise typer.Exit(code=1)
    features_scaled = scaler.transform(features.reshape(1, -1))
    y_pred = model.predict(features_scaled)[0]
    y_proba = model.predict_proba(features_scaled)[0]
    predicted_genre = label_encoder.classes_[y_pred]
    confidence = y_proba[y_pred]
    top_indices = np.argsort(y_proba)[-top_n:][::-1]
    top_genres = [(label_encoder.classes_[i], y_proba[i]) for i in top_indices]

    print("\nPrediction Results")
    print("="*40)
    print(f"  File: {Path(audio_path).name}")
    print(f"  Predicted Genre: {predicted_genre}")
    print(f"  Confidence: {confidence:.2%}\n")
    print("  Top Predictions:")
    for genre, prob in top_genres:
        bar = "█" * int(prob * 50)
        print(f"    {genre:15s} {bar} {prob:.2%}")
    print("="*40)

if __name__ == "__main__":
    app()