import os
import sys
import queue
import time
import threading
from termcolor import colored
import json

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf    
from sklearn.model_selection import train_test_split

import librosa
import sounddevice as sd


class Config:
    def __init__(self, config_path='config.json'):
        # Load the JSON configuration file
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        # Set attributes from the JSON config
        self.MODEL_FILE = config["MODEL_FILE"]
        self.DATASET_DIR = config["DATASET_DIR"]
        self.LABELS = config["LABELS"]
        self.EPOCHS = config["EPOCHS"]
        self.BATCH_SIZE = config["BATCH_SIZE"]
        self.DURATION = config["DURATION"]
        self.SAMPLE_RATE = config["SAMPLE_RATE"]
        self.WINDOW_SIZE = config["WINDOW_SIZE"]
        self.STEP_SIZE = config["STEP_SIZE"]

# Usage
config = Config()

q = queue.Queue()
stop_flag = threading.Event()  # Create a threading event to signal when to stop recording

   
def extract_features(y):
    sr = config.SAMPLE_RATE
    # Calculate Zero Crossing Rate (ZCR)
    # ZCR measures the rate at which the signal changes sign, indicating the frequency content.
    zcr = np.mean(librosa.feature.zero_crossing_rate(y))  

    # Calculate Root Mean Square (RMS)
    # RMS represents the energy of the audio signal, giving an indication of its loudness.
    rms = np.mean(librosa.feature.rms(y=y))

    # Calculate Peak Amplitude
    # Peak amplitude is the maximum absolute value of the audio signal, representing the loudest point.
    peak_amp = np.max(np.abs(y))    

    # Calculate Mean Amplitude
    # Mean amplitude is the average of the audio signal, giving a general sense of the signal level.
    mean_amp = np.mean(y)
    
    # Calculate the Mel spectrogram of the audio signal
    # The Mel spectrogram is a representation of the short-term power spectrum of sound, which emphasizes perceptually relevant features for audio analysis and processing.
    melspectrogram = np.mean(librosa.feature.melspectrogram(y=y, sr=sr), axis=1)
   
    # Calculate Spectral Bandwidth
    # Spectral bandwidth measures the width of the spectrum and is indicative of the timbre of the sound.
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=y,sr=sr))

    # Calculate Spectral Centroid
    # Spectral centroid indicates where the center of mass of the spectrum is located, often related to the perceived brightness of a sound.
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
    
    
    # Apply Fast Fourier Transform (FFT) to convert to the frequency domain
    # FFT converts the time-domain signal into its frequency components.
    fft = np.fft.fft(y, n=sr)  # Compute the FFT

    # Take magnitude of the FFT and only positive frequencies
    # Since FFT produces complex numbers, we take the absolute value to get the magnitude.
    fft_magnitude = np.abs(fft[:sr // 2])  # Only keep the first half of the FFT result (positive frequencies)

    # Combine all extracted features into a single array and return
    return np.hstack([
        np.array([zcr, rms, peak_amp, mean_amp]),  # Combine temporal features
        # melspectrogram,                            # Add Mel spectrogram
        spectral_centroid,                         # Add spectral centroid
        spectral_bandwidth,                        # Add spectral bandwidth
        fft_magnitude                              # Add FFT magnitude
    ])
    
def load_dataset(data_dir, labels):
    # Initialize lists to store audio features and corresponding labels
    audio_data = []    # Will hold extracted features from audio files
    audio_labels = []  # Will hold labels corresponding to each audio file
    
    # Loop through each label (sub-directory) specified in labels
    for label in labels:
        # Construct the path to the label's directory
        folder_path = os.path.join(data_dir, label)
        
        # Loop through each file in the label's directory
        for file in os.listdir(folder_path):
            # Check if the file is a .wav audio file
            if file.endswith('.wav'):
                # Construct the full file path
                file_path = os.path.join(folder_path, file)
                
                # Load the audio file using librosa
                audio, _ = librosa.load(file_path)
               
                # Extract features from the loaded audio
                features = extract_features(audio)

                # Normalize the features to a range of [0, 1]
                # This prevents large feature values from skewing the data
                features = features / np.max(features)               

                # Append the features and corresponding label index to their respective lists
                audio_data.append(features)
                audio_labels.append(labels.index(label))  # Store the index of the label in the list of labels
    
    # Convert lists to NumPy arrays for easier handling and processing
    audio_data = np.array(audio_data)    # Array of feature vectors
    audio_labels = np.array(audio_labels)  # Array of label indices

    # Return the extracted features and corresponding labels
    return audio_data, audio_labels

def create_dense_model(input_shape, num_classes):
    # model = tf.keras.models.Sequential([ 
    #     tf.keras.layers.InputLayer(input_shape=input_shape), 
    #     tf.keras.layers.Dense(500, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),                  
    #     tf.keras.layers.Dense(300, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),        
    #     tf.keras.layers.Dense(150, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),        
    #     tf.keras.layers.Dense(100, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(60, activation='relu'),
    #     tf.keras.layers.Dropout(0.5),
    #     tf.keras.layers.Dense(num_classes, activation='softmax')
    # ])    
    model = tf.keras.models.Sequential([ 
        tf.keras.layers.InputLayer(input_shape=input_shape), 
        tf.keras.layers.Dense(320, activation='relu'),
        tf.keras.layers.Dropout(0.5),                  
        tf.keras.layers.Dense(220, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(120, activation='relu'),
        tf.keras.layers.Dropout(0.5),         
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])    
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def plot_training_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.savefig("firgure.png") 
    plt.show()

def predict(y,model, labels ):

    # Combine features
    features = extract_features(y)
    
    features = features / np.max(features)
    # Reshape the features to match the input shape of the model
    input_data = features.reshape(1, -1)    
    
    # Predict using the trained model
    predictions = model.predict(input_data, verbose=0)
    
    # Return the predicted class
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    
    # Get the label name from the index
    predicted_label = labels[predicted_label_index]
    
    return predicted_label
    
def sliding_window_detection(model):
    window_size = config.WINDOW_SIZE
    labels = config.LABELS
    step_size = config.STEP_SIZE
    
    buffer = np.array([])  # Initial empty buffer to hold incoming samples
    window_index = 0
    while not stop_flag.is_set() or not q.empty():
        if not q.empty():
            # Get the next chunk of samples from the queue
            new_samples = q.get()

            # Append the new samples to the buffer
            buffer = np.concatenate((buffer, new_samples.flatten()))

            # Process the buffer only if we have at least window_size samples
            while len(buffer) >= window_size:
                # Extract the current window
                current_window = buffer[:window_size]
                
                predicted_label = predict(current_window, model, labels)                    
              
                if predicted_label.startswith("ok"):
                    print(colored('OK', 'green'))
                if predicted_label.startswith("ng"):
                    print(colored('NG', 'red'))   
                                    
                # Slide the window forward by step_size
                buffer = buffer[step_size:]
                window_index = window_index + 1

        else:
            time.sleep(0.1)  # Avoid busy-waiting if the queue is empty

def record_sound(selected_device):
    sr = config.SAMPLE_RATE
    duration = config.DURATION
    
    # Callback function that is called when audio is available
    def callback(indata, frames, time, status):
        # Copy the incoming audio data to the queue
        q.put(indata.copy())  # Store the audio data for further processing

    # Create an InputStream to capture audio from the selected device
    with sd.InputStream(callback=callback, channels=1, samplerate=sr, device=selected_device):
        # Sleep for the duration specified in the configuration to allow for recording
        time.sleep(duration)
    
    # Signal that the recording should stop
    stop_flag.set()  # Set the stop flag to notify other threads or processes that recording has finished

def main_train():    
    dataset_dir = config.DATASET_DIR
    labels = config.LABELS
    epochs = config.EPOCHS
    batch_size = config.BATCH_SIZE
    model_file = config.MODEL_FILE
    
    # Load the audio data and extract frequency components using FFT
    X, y = load_dataset(dataset_dir, labels = labels)   
            
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    input_shape = (X_train.shape[1],)  # Input shape is based on FFT size
    model = create_dense_model(input_shape, len(labels))
    
    # Train the model
    history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=epochs,
                    batch_size=batch_size)
    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f"Test accuracy: {test_acc}")

    # Save the model
    model.save(model_file)

def device_selection():
    return 1
    selected_device = None
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            if("Microphone (Sound Blaster Play!"in device['name'] and ')' not in device['name']  ):
                print(f"Device {i}: {device['name']}")
        
    while selected_device is None:
        try:
            user_input = int(input("Please enter the index of the microphone device you want to use: "))
            if 0 <= user_input < len(devices) and devices[user_input]['max_input_channels'] > 0:
                selected_device = user_input
            else:
                print("Invalid selection. Please select a valid microphone device index.")
        except ValueError:
            print("Please enter a valid number.")
    
    return selected_device
    
def main_test():        
    model_file = config.MODEL_FILE
        
    selected_device = device_selection()
    model = tf.keras.models.load_model(model_file) 
    
    # Create threads for recording and sliding window detection
    detect_thread = threading.Thread(target=sliding_window_detection, args=(model,))
    record_thread = threading.Thread(target=record_sound, args=(selected_device,))
    
    # Start the threads
    detect_thread.start()
    record_thread.start()

    # Wait for the threads to finish
    detect_thread.join()
    record_thread.join()
    
    
if __name__ == "__main__":         
    if len(sys.argv) < 2:        
        function_name = None
    else:
        function_name = sys.argv[1]    

    if(function_name=='train' or function_name==None):
        main_train()
        
    if (function_name=='test' or function_name==None):        
        main_test()