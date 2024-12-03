import os
import sys
import queue
import time
import threading
import json
import argparse

import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import librosa
import pywt

import sounddevice as sd
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from scipy.signal import find_peaks, hilbert, windows

from termcolor import colored
from datetime import datetime


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf    
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*n_fft.*')

# Get the absolute path of the current script
current_file_path = os.path.abspath(__file__)
# Get the directory of the current script
source_directory = os.path.dirname(current_file_path)    

# Initialize the parser
parser = argparse.ArgumentParser(description='A script for training two models with specific arguments')

# Adding arguments with short names
parser.add_argument('-f', '--function', type=str, choices=['train', 'test'], default='test', help='Specify the function to use')
parser.add_argument('-m', '--model', type=int, choices=[0, 1], default=0 , help='Specify the model to use')

# Parse arguments
args = parser.parse_args()

q = queue.Queue()
stop_flag = threading.Event()  # Create a threading event to signal when to stop recording

class Config:
    def __init__(self, config_path=os.path.join(source_directory, '../config.json')):
        # Load the JSON configuration file
        with open(config_path, 'r') as file:
            config = json.load(file)
        
        # Set attributes from the JSON config
        self.DATASET_DIR = config['DATASET_DIR']
        self.LABELS = config['LABELS'] 
        self.EPOCHS = config['EPOCHS']
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.PATIENCE = config['PATIENCE']
        self.DURATION = config['DURATION']
        self.SAMPLE_RATE = config['SAMPLE_RATE']
        self.WINDOW_SIZE = config['WINDOW_SIZE']
        self.SEGMENT_DURATION = config['SEGMENT_DURATION']
        self.STEP_SIZE = config['STEP_SIZE']     
        self.FRAME_LENGTH = config['FRAME_LENGTH']
        self.HOP_LENGTH = config['HOP_LENGTH']         
        self.PEAK_HEIGHT = config['PEAK_HEIGHT']
        self.PEAK_DISTANCE = config['PEAK_DISTANCE']
        self.PEAK_THRESHOLD = config['PEAK_THRESHOLD']
        self.MAX_NUM_PEAKS = config['MAX_NUM_PEAKS']
        self.HIGH_THRESH_AMP = config['HIGH_THRESH_AMP']
        self.LOW_THRESH_AMP = config['LOW_THRESH_AMP']
        self.WEIGHTS = config['WEIGHTS']

config = Config()

def use_machine_learning(signal):    
    result = True      
    max_amp = np.max(np.abs(signal))
    num_peaks = get_num_amplitude_peaks(signal)   
    
    if (max_amp > config.HIGH_THRESH_AMP or max_amp < config.LOW_THRESH_AMP):
        result =  False
    
    if (num_peaks <= 0 or num_peaks >= config.MAX_NUM_PEAKS+1):
        result = False        
    
    return result
    
def get_num_amplitude_peaks(signal):
    peak_height = config.PEAK_HEIGHT   # Minimum height for peak detection in time domain
    peak_distance = config.PEAK_DISTANCE  # Minimum distance between peaks for peak detection
    peak_threshold = config.PEAK_THRESHOLD  # Minimum distance between peaks for peak detection

    # Detect peaks in the time-domain signal with specified height and distance constraints
    peaks_signal, _ = find_peaks(signal, height=peak_height, distance=peak_distance , threshold=peak_threshold)
    return len(peaks_signal)

def extract_peak_segment(signal, sr = config.SAMPLE_RATE, duration = config.SEGMENT_DURATION):
    # Calculate the envelope of the signal using the Hilbert Transform.
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    # Detect all peaks in the envelope.
    peaks, _ = find_peaks(envelope)    

    # Find the index of the highest peak.
    highest_peak_idx = peaks[np.argmax(envelope[peaks])]
    
    # Calculate the half-window size in samples.
    half_window_samples = int((duration * sr) / 2)
    
    # Get the segment around the peak
    start_index = max(0, highest_peak_idx - half_window_samples)
    end_index = min(len(signal), highest_peak_idx + half_window_samples)
    segment = signal[start_index:end_index]
    
    return segment      
    
def extract_amplitude_features(signal):    
    peak_amp = np.max(np.abs(signal))    
    mean_amp = np.mean(np.abs(signal))  
    min_amp = np.min(np.abs(signal))
    median_amp = np.median(np.abs(signal))
    
    return np.array([peak_amp, mean_amp, min_amp, median_amp])
    
def extract_fft_features(signal, sr=config.SAMPLE_RATE):
    window_size = len(signal)
    gaussian_window = windows.gaussian(window_size, std=window_size/6)
    windowed_signal = signal * gaussian_window
    fft_magnitude = np.abs(np.fft.fft(windowed_signal, n=sr)[:sr // 2])
    
    return fft_magnitude

def extract_max_energy_distribution_features(signal): 
    window_size = len(signal)
    gaussian_window = windows.gaussian(window_size, std=window_size/6)
    windowed_signal = signal * gaussian_window
    
    fft_result = np.fft.fft(windowed_signal)
    
    # Compute the magnitude of the FFT
    fft_magnitude = np.abs(fft_result)
    
    # Calculate energy by squaring the magnitudes
    energy_distribution = np.square(fft_magnitude)
    
    # Get the maximum energy distribution across all frequency bins
    max_energy_distribution = np.max(energy_distribution)  # Max energy across the entire frequency spectrum
    
    return max_energy_distribution

def extract_short_time_energy_features(signal):
    # Compute the short-term energy (STE) for each frame
    energy = np.array([sum(signal[i:i+config.FRAME_LENGTH]**2) for i in range(0, len(signal), config.HOP_LENGTH)])

    # Normalize energy (optional, but useful for comparisons)
    energy = energy / np.max(energy)

    # Calculate the statistical features
    mean_energy = np.mean(energy)
    std_energy = np.std(energy)
    median_energy = np.median(energy)
    max_energy = np.max(energy)
    min_energy = np.min(energy)
    variance_energy = np.var(energy)    
        
    return np.array([mean_energy, std_energy, median_energy, max_energy, min_energy, variance_energy])    

def extract_features(signal):        
    sr = config.SAMPLE_RATE    
        
    zero_crossing_rate = librosa.feature.zero_crossing_rate(signal)
    zero_crossing_rate_max = np.max(zero_crossing_rate, axis=1)    
    zero_crossing_rate_mean = np.mean(zero_crossing_rate, axis=1)    
    
    root_mean_square_energy = librosa.feature.rms(y=signal)
    root_mean_square_energy_max = np.max(root_mean_square_energy, axis=1)
    root_mean_square_energy_mean = np.mean(root_mean_square_energy, axis=1)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sr)
    spectral_centroid_max = np.max(spectral_centroid, axis=1)   
    spectral_centroid_mean  = np.mean(spectral_centroid, axis=1)   
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sr)
    spectral_bandwidth_max =  np.max(spectral_bandwidth, axis=1)   
    spectral_bandwidth_mean =  np.mean(spectral_bandwidth, axis=1)   
    
    short_time_energy = extract_short_time_energy_features(signal)            
    amplitude_features = extract_amplitude_features(signal)             
    max_energy_distribution = extract_max_energy_distribution_features(signal)        
    fft_magnitude = extract_fft_features(signal)    
    num_amplitude_peaks = np.array(get_num_amplitude_peaks(signal)) 

    features = np.hstack([
        config.WEIGHTS['NUM_AMPLITUDE_PEAKS'] * num_amplitude_peaks,
        config.WEIGHTS['ZERO_CROSSING_RATE'] * zero_crossing_rate_max,
        config.WEIGHTS['ZERO_CROSSING_RATE'] * zero_crossing_rate_mean,
        config.WEIGHTS['ROOT_MEAN_SQUARE_ENERGY'] * root_mean_square_energy_max,
        config.WEIGHTS['ROOT_MEAN_SQUARE_ENERGY'] * root_mean_square_energy_mean,
        config.WEIGHTS['SPECTRAL_CENTROID'] * spectral_centroid_max,
        config.WEIGHTS['SPECTRAL_CENTROID'] * spectral_centroid_mean,
        config.WEIGHTS['SPECTRAL_BANDWIDTH'] * spectral_bandwidth_max,
        config.WEIGHTS['SPECTRAL_BANDWIDTH'] * spectral_bandwidth_mean,
        config.WEIGHTS['AMPLITUDE_FEATURES'] * amplitude_features,
        config.WEIGHTS['SHORT_TIME_ENERGY'] * short_time_energy,
        config.WEIGHTS['FFT_MAGNITUDE'] * fft_magnitude,
        config.WEIGHTS['MAX_ENERGY_DISTRIBUTION'] * max_energy_distribution,
    ])

    return features

def load_dataset(data_dir, labels):
    # Initialize lists to store audio features and corresponding labels
    audio_data = []    # Will hold extracted features from audio files
    audio_labels = []  # Will hold labels corresponding to each audio file
    
    # Loop through each label (sub-directory) specified in labels
    for label in labels:
        # Construct the path to the label's directory
        folder_path = os.path.join(data_dir, label)
        count = 0

        # Loop through each file in the label's directory
        for file in os.listdir(folder_path):
            # Check if the file is a .wav audio file
            if file.endswith('.wav'):
                # Construct the full file path
                file_path = os.path.join(folder_path, file)
                
                # Load the audio file using librosa
                signal, _ = librosa.load(file_path)       
                
                segment = extract_peak_segment(signal)         
              
                # Extract features from the loaded audio
                features = extract_features(segment)                

                # Normalize the features to a range of [0, 1]
                # This prevents large feature values from skewing the data
                features = features / np.max(features, axis=0)       

                # Append the features and corresponding label index to their respective lists
                audio_data.append(features)
                audio_labels.append(labels.index(label))  # Store the index of the label in the list of labels   
    
    # Convert lists to NumPy arrays for easier handling and processing
    audio_data = np.array(audio_data)    # Array of feature vectors
    audio_labels = np.array(audio_labels)  # Array of label indices

    # Return the extracted features and corresponding labels
    return audio_data, audio_labels

def create_dense_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([ 
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(300, activation='relu', kernel_regularizer=regularizers.l2(0.003)), 
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=regularizers.l2(0.003)), 
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(100, activation='relu', kernel_regularizer=regularizers.l2(0.003)), 
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])    
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'])
    
    return model

def plot_training_history(history, model_index):
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
    
    plt.savefig(f'{os.path.join(source_directory, f'../training_history_{model_index}.png')}')
    plt.show()

def plot_confusion_matrix(model, x_val, y_val, labels, model_index):
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    plt.savefig(f'{os.path.join(source_directory, f'../confusion_matrix_{model_index}.png')}')
    plt.show()
    
def predict(signal, model, labels ):
    predicted_label = 'NG'
    try:  
            
        features = extract_features(signal)        

        features = features / np.max(features, axis=0)    

        # Reshape the features to match the input shape of the model
        input_data = features.reshape(1, -1)    
        
        # Predict using the trained model
        predictions = model.predict(input_data, verbose=0)
        
        # Return the predicted class
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        
        # Get the label name from the index
        predicted_label = labels[predicted_label_index]
        
    except :
        pass
    
    return predicted_label
    
def sliding_window_detection(models):
    window_size = config.WINDOW_SIZE
    step_size = config.STEP_SIZE
    
    buffer = np.array([])  # Initial empty buffer to hold incoming samples
    model_index = len(models) - 1
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
                
                segment = extract_peak_segment(current_window)                
                
                num_amplitude_peaks = get_num_amplitude_peaks(segment)
                max_amplitude = np.max(np.abs(segment))
                if (use_machine_learning(segment)):     
                    if(model_index == 0):
                        predicted_label = predict(segment, models[0], config.LABELS[0])
                        if predicted_label.startswith('OK'):
                            print(colored(f'OK: {num_amplitude_peaks} : {max_amplitude:.4f}', 'green'))
                            # timestamp = datetime.now().strftime('%d%m%y_%H%M%S_%f')[:-3]
                            # sf.write(os.path.join(source_directory, f'../unexpected/OK_{timestamp}.wav'),current_window,config.SAMPLE_RATE)
                            # exit()                            
                        if predicted_label.startswith('NG'):
                            # timestamp = datetime.now().strftime('%d%m%y_%H%M%S_%f')[:-3]
                            # sf.write(os.path.join(source_directory, f'../unexpected/NG_{timestamp}.wav'),current_window,config.SAMPLE_RATE)
                            print(colored(f'NG: {num_amplitude_peaks} : {max_amplitude:.4f}', 'red'))
                    if(model_index == 1):
                        predicted_label_1 = predict(segment, models[1], config.LABELS[1])                        
                        if predicted_label_1.startswith('OK'):
                            predicted_label_0 = predict(segment, models[0], config.LABELS[0])
                            if predicted_label_0.startswith('OK'):
                                print(colored(f'OK: {num_amplitude_peaks} : {max_amplitude:.4f}', 'green'))
                            if predicted_label_0.startswith('NG'):
                                print(colored(f'NG: {num_amplitude_peaks} : {max_amplitude:.4f}', 'red'))
                        if predicted_label_1.startswith('NG'):
                            print(colored(f'NG: {num_amplitude_peaks} : {max_amplitude:.4f}', 'red'))
                else:
                    # timestamp = datetime.now().strftime('%d%m%y_%H%M%S_%f')[:-3]
                    # sf.write(os.path.join(source_directory, f'../unexpected/NG_{timestamp}.wav'),current_window,config.SAMPLE_RATE)
                    print(colored(f'NG: {num_amplitude_peaks} : {max_amplitude:.4f}', 'red'))
         
                                    
                # Slide the window forward by step_size
                buffer = buffer[step_size:]

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

def main_train(model_index):    
    dataset_dir = config.DATASET_DIR
    epochs = config.EPOCHS
    batch_size = config.BATCH_SIZE
    patience = config.PATIENCE
    model_file_path = os.path.join(source_directory, f'../model_{model_index}.keras')
    labels = config.LABELS[model_index]
    
    # Load the audio data and extract frequency components using FFT
    X, y = load_dataset(dataset_dir, labels = labels)   
            
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.1, random_state=42)
    
    # Create the model
    input_shape = (X_train.shape[1],)  # Input shape is based on FFT size
    model = create_dense_model(input_shape, len(labels))
    
        
    early_stopping_accuracy = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy', 
        patience=patience,
        restore_best_weights=True)    

    early_stopping_loss = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=patience,
        restore_best_weights=True)
    
    history = model.fit(X_train, y_train, 
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping_accuracy, early_stopping_loss])
    
    # Plot training history
    plot_training_history(history, model_index)
    
    # Evaluate the model and plot confusion matrix
    plot_confusion_matrix(model, X_val, y_val, labels, model_index)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f'Test accuracy: {test_acc}')
    
    
    # Specify the current file name and the new name
    current_file = model_file_path
    new_file = model_file_path.replace("model_0","model_0_bk")
    try:
        os.rename(current_file, new_file)
        print(f"File renamed from {current_file} to {new_file}")
    except FileNotFoundError:
        print(f"The file {current_file} does not exist.")
    except PermissionError:
        print("You don't have permission to rename this file.")
    except Exception as e:
        print(f"An error occurred: {e}")
    # Save the model
    model.save(model_file_path)
    

def device_selection():
    return 1
    selected_device = None
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            if('Microphone (Sound Blaster Play!'in device['name'] and ')' not in device['name']  ):
                print(f'Device {i}: {device['name']}')
        
    while selected_device is None:
        try:
            user_input = int(input('Please enter the index of the microphone device you want to use: '))
            if 0 <= user_input < len(devices) and devices[user_input]['max_input_channels'] > 0:
                selected_device = user_input
            else:
                print('Invalid selection. Please select a valid microphone device index.')
        except ValueError:
            print('Please enter a valid number.')
    
    return selected_device
    
def main_test(max_idx_model):  
    selected_device = device_selection()
    models = []
    for idx in range(max_idx_model+1):        
        model_file_path = os.path.join(source_directory, f'../model_{idx}.keras')
        model = tf.keras.models.load_model(model_file_path) 
        models.append(model)
            
    # Create threads for recording and sliding window detection
    detect_thread = threading.Thread(target=sliding_window_detection, args=(models,))
    record_thread = threading.Thread(target=record_sound, args=(selected_device,))
    
    # Start the threads
    detect_thread.start()
    record_thread.start()

    # Wait for the threads to finish
    detect_thread.join()
    record_thread.join()
    
if __name__ == '__main__':                     
    if(args.function =='train'):
        main_train(args.model)        
    if(args.function =='test'):
        main_test(args.model)
