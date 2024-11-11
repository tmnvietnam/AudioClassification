import os
import sys
import queue
import time
import threading
import json
import cv2

from termcolor import colored

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pywt

import sounddevice as sd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from scipy.signal import butter, filtfilt
from scipy.signal import find_peaks
from scipy.signal.windows import gaussian  # Corrected import

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf    
from tensorflow.keras import regularizers
from tensorflow.keras.optimizers import Adam


class Config:
    def __init__(self, config_path='config.json'):
        # Load the JSON configuration file
        with open(config_path, 'r') as file:
            config = json.load(file)
        
     # Set attributes from the JSON config
        self.DATASET_DIR = config["DATASET_DIR"]
        self.LABELS = config["LABELS"]
        self.EPOCHS = config["EPOCHS"]
        self.BATCH_SIZE = config["BATCH_SIZE"]
        self.PATIENCE = config["PATIENCE"]
        self.DURATION = config["DURATION"]
        self.SAMPLE_RATE = config["SAMPLE_RATE"]
        self.SEGMENT_DURATION = config["SEGMENT_DURATION"]
        self.WINDOW_SIZE = config["WINDOW_SIZE"]
        self.STEP_SIZE = config["STEP_SIZE"]     
        self.FRAME_LENGTH = config["FRAME_LENGTH"]
        self.HOP_LENGTH = config["HOP_LENGTH"]         
        self.PEAK_HEIGHT = config["PEAK_HEIGHT"]
        self.PEAK_DISTANCE = config["PEAK_DISTANCE"]
        self.PEAK_THRESHOLD = config["PEAK_THRESHOLD"]
        self.MAX_NUM_PEAKS = config["MAX_NUM_PEAKS"]
        self.HIGH_THRESH_AMP = config["HIGH_THRESH_AMP"]
        self.LOW_THRESH_AMP = config["LOW_THRESH_AMP"]

# Usage
config = Config()

N = 256

q = queue.Queue()
stop_flag = threading.Event()  # Create a threading event to signal when to stop recording

def get_num_amp_peaks(y):
    peak_height = config.PEAK_HEIGHT   # Minimum height for peak detection in time domain
    peak_distance = config.PEAK_DISTANCE  # Minimum distance between peaks for peak detection
    peak_threshold = config.PEAK_THRESHOLD  # Minimum distance between peaks for peak detection

    # Detect peaks in the time-domain signal with specified height and distance constraints
    peaks_y, _ = find_peaks(y, height=peak_height, distance=peak_distance , threshold=peak_threshold)
    return len(peaks_y)

def use_machine_learning(y):
    
    result = True      
    max_amp = np.max(np.abs(y))
    num_peaks = get_num_amp_peaks(y)   
    
    # print(max_amp, num_peaks)
    if (max_amp >= config.HIGH_THRESH_AMP or max_amp <= config.LOW_THRESH_AMP):
        result =  False
    
    if (num_peaks <= 0 or num_peaks >= config.MAX_NUM_PEAKS+1):
        result = False        
    
    return result
# Replace STFT-based spectrogram extraction with Wavelet Transform
def extract_scaleogram(y, sr, wavelet='cmor', scales=np.arange(1, 128)):
    # Perform Continuous Wavelet Transform (CWT)
    coefficients, frequencies = pywt.cwt(y, scales, wavelet, 1.0/sr)
    
    # Absolute value of the coefficients
    scaleogram = np.abs(coefficients)
    
    return scaleogram

def save_scaleogram_image(scaleogram, save_path):
    # Use matplotlib to save the scaleogram as an image
    plt.figure(figsize=(4, 4))
    plt.imshow(scaleogram, aspect='auto', origin='lower', cmap='viridis')
    plt.axis('off')  # Turn off axes for a cleaner image
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    

def segment_around_the_peak(signal, sr, duration):
     # Find the peak of the signal
    peak_index = np.argmax(np.abs(signal))  # Index of the maximum absolute value (peak)
    
    # Calculate the number of samples for the desired duration around the peak
    half_window_samples = int((duration * sr) / 2)
    
    # Get the segment around the peak
    start_index = max(0, peak_index - half_window_samples)
    end_index = min(len(signal), peak_index + half_window_samples)
    segment = signal[start_index:end_index]
    
    return segment  
    
def load_dataset(data_dir, labels, save_dir):
    # Initialize lists to store audio features and corresponding labels
    audio_data = []    # Holds extracted features from audio files
    audio_labels = []  # Holds labels corresponding to each audio file

    os.makedirs(save_dir, exist_ok=True)

    # Loop through each label (sub-directory) specified in labels
    for label in labels:
        # Construct the path to the label's directory
        folder_path = os.path.join(data_dir, label)

        # Ensure folder exists
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} does not exist. Skipping this label.")
            continue

        # Loop through each file in the label's directory
        for file in os.listdir(folder_path):
            # Check if the file is a .wav audio file
            if file.endswith('.wav'):
                # Construct the full file path
                file_path = os.path.join(folder_path, file)

                # Load the audio file using librosa
                signal, sr = librosa.load(file_path, sr=config.SAMPLE_RATE)

                # Check if we need to process this file for machine learning
                # Extract scaleogram features
                
                segment = segment_around_the_peak(signal, sr, config.SEGMENT_DURATION)    
                window_size = len(segment)
                gaussian_window = gaussian(window_size, std=window_size/6)
                windowed_segment = segment * gaussian_window
                
                scaleogram = extract_scaleogram(segment, sr)

                # Resize the scaleogram to a fixed shape (e.g., NxN)
                scaleogram_resized = cv2.resize(scaleogram, (N, N))

                # Save the scaleogram image to disk
                save_path = os.path.join(save_dir, f"{label}_{file[:-4]}.png")
                save_scaleogram_image(scaleogram_resized, save_path)

                # Append the features and corresponding label index to their respective lists
                scaleogram_resized = np.expand_dims(scaleogram_resized, axis=-1)  # Add channel dimension

                # Normalize the scaleogram (optional, between 0 and 1)
                scaleogram_resized = scaleogram_resized / np.max(scaleogram_resized)

                # Append the features and corresponding label index to their respective lists
                audio_data.append(scaleogram_resized)
                audio_labels.append(labels.index(label))  # Store the index of the label

    # Convert lists to numpy arrays for model input compatibility
    audio_data = np.array(audio_data)
    audio_labels = np.array(audio_labels)
    return audio_data, audio_labels


def create_dense_model(input_shape, num_classes):    

    model = tf.keras.models.Sequential()               

    # Add Conv2D layers (convolutional layers)
    model.add(tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5)) 
    model.add(tf.keras.layers.Conv2D(64, (3, 3), activation='relu',kernel_regularizer=regularizers.l2(0.01)))
    model.add(tf.keras.layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.5)) 
    model.add(tf.keras.layers.Flatten())  # Flatten the 2D matrix into a 1D vector
    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))  # Fully connected layer
    model.add(tf.keras.layers.Dropout(0.5)) 
    model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # Output layer (e.g., 10 classes)

    model.compile(optimizer=Adam(learning_rate=0.001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

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
    
    plt.savefig("training_history_cnn.png") 
    plt.show()

def plot_confusion_matrix(model, x_val, y_val, labels):
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    plt.savefig("confusion_matrix_cnn.png") 
    plt.show()

    
def predict(y,model, labels ):
    
    if (use_machine_learning(y)):     
        # Extract spectrogram (assuming extract_spectrogram returns a 2D array)
        spectrogram = extract_spectrogram(audio, sample_rate)
        
        # Resize the spectrogram to (N, N)
        spectrogram_resized = cv2.resize(spectrogram, (N, N))
        spectrogram_resized = np.expand_dims(spectrogram_resized, axis=-1)  # Shape to (N, N, 1)
        
        # Normalize the spectrogram
        spectrogram_resized = spectrogram_resized / np.max(spectrogram_resized)
        
        # Add a batch dimension and make prediction
        spectrogram_input = np.expand_dims(spectrogram_resized, axis=0)  # Shape to (1, N, N, 1)
        
        # Predict using the trained model
        predictions = model.predict(spectrogram_input, verbose=0)
        
        # Return the predicted class
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        
        # Get the label name from the index
        predicted_label = labels[predicted_label_index]
        
        return predicted_label
    else:
        return "NG"
    
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

                if predicted_label.startswith("OK"):
                    print(colored(f"OK: {predicted_label}", 'green'))
                if predicted_label.startswith("NG"):
                    print(colored(f'NG {predicted_label}', 'red'))   
                                    
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
    patience = config.PATIENCE   
   
    model_file = "model_cnn.h5"
    save_dir = 'scaleogram'
    
    # Load the audio data and extract frequency components using FFT
    X, y = load_dataset(dataset_dir, labels = labels , save_dir = save_dir)   
            
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    input_shape = (N, N, 1)  # Input shape is based on FFT size
    model = create_dense_model(input_shape, len(labels))
     
    # Train the model
    early_stopping_accuracy = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=patience,
        restore_best_weights=True
    )    

    early_stopping_loss = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience,
        restore_best_weights=True
    )
    
    history = model.fit(X_train, y_train, 
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping_accuracy, early_stopping_loss]
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate the model and plot confusion matrix
    plot_confusion_matrix(model, X_val, y_val, labels)

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
    model_file = 'model_cnn.h5'
        
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

    if(function_name=='train'):
        main_train()
        
    if (function_name=='test'):
        main_test()