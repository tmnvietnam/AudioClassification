import sounddevice as sd
import numpy as np
import librosa
import tensorflow as tf

SAMPLING_RATE = 22050

# 1. Function to record audio
def record_audio(duration):
    """
    Records audio for a given duration and sample rate.
    Parameters:
        - duration: Duration of the recording in seconds.
        - sr: Sample rate for the recording (default is 22050 Hz).
    Returns:
        - recorded_audio: The recorded audio signal as a NumPy array.
    """
    print(f"Recording for {duration} seconds...")
    recorded_audio = sd.rec(int(duration * SAMPLING_RATE), samplerate=SAMPLING_RATE, channels=1)
    sd.wait()  # Wait for the recording to finish
    recorded_audio = recorded_audio.flatten()  # Convert 2D array to 1D
    print("Recording complete.")
    return recorded_audio

# 2. Preprocess audio (apply FFT)
def preprocess_audio(audio):
    """
    Preprocesses audio by applying FFT to transform it into the frequency domain.
    Parameters:
        - audio: The raw audio signal.
        - sr: Sample rate (default is 22050 Hz).
    Returns:
        - fft_magnitude: Magnitude of the FFT applied to the audio.
    """
    # Apply FFT
    fft = np.fft.fft(audio, n=SAMPLING_RATE)
    fft_magnitude = np.abs(fft[:SAMPLING_RATE // 2])  # Keep only positive frequencies
    return fft_magnitude

# 3. Load model and classify the audio
def classify_audio(model, audio):
    """
    Classifies a preprocessed audio sample using the trained model.
    Parameters:
        - model: The pre-trained classification model.
        - audio: Preprocessed audio data (FFT magnitudes).
        - sr: Sample rate used for preprocessing.
    Returns:
        - prediction: The predicted class label.
    """
    # Preprocess the audio (FFT)
    fft_magnitude = preprocess_audio(audio)
    
    # Normalize the audio features
    fft_magnitude = fft_magnitude / np.max(fft_magnitude)
    
    # Reshape for model input (1, feature_size)
    fft_magnitude = np.expand_dims(fft_magnitude, axis=0)
    
    # Predict using the trained model
    prediction = model.predict(fft_magnitude)
    
    # Return the predicted class
    predicted_label = np.argmax(prediction, axis=1)
    return predicted_label

# 4. Main test function
def main():
    """
    Records audio, preprocesses it, and classifies it using the trained model.
    Parameters:
        - model_path: Path to the pre-trained model file (e.g., 'sound_classification_fft_model.h5').
        - labels: List of class labels (e.g., ['ng', 'ok']).
        - duration: Duration of the audio recording (in seconds).
        - sr: Sample rate (default is 22050 Hz).
    """
    labels = ["ng", "ok"]  # Define the labels used in your model
    
    # Load the trained model
    model = tf.keras.models.load_model("model.h5")
    
    # Record the audio
    recorded_audio = record_audio(0.5)
    
    # Classify the recorded audio
    predicted_label_index = classify_audio(model, recorded_audio)
    
    # Get the label from index
    predicted_label = labels[predicted_label_index[0]]
    print(f"Predicted label: {predicted_label}")
    
    return predicted_label

if __name__ == "__main__":
    main()
