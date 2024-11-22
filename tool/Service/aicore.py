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

import sounddevice as sd
import soundfile as sf

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from scipy.signal import find_peaks, hilbert, windows

from termcolor import colored
from datetime import datetime


import tensorflow as tf    
from tensorflow.keras import regularizers
from tensorflow.keras.callbacks import EarlyStopping

import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='.*n_fft.*')
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Get the absolute path of the current script
current_file_path = os.path.abspath(__file__)
# Get the directory of the current script
source_directory = os.path.dirname(current_file_path)    

class Config:
    def __init__(self, config_path=os.path.join(source_directory, 'config.json')):
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


current_file_path = os.path.abspath(__file__)
# Get the directory of the current script
source_directory = os.path.dirname(current_file_path)    


def use_machine_learning(signal):    
    result = True      
    max_amp = np.max(np.abs(signal))
    num_peaks = get_num_amplitude_peaks(signal)   
    
    if (max_amp > config.HIGH_THRESH_AMP or max_amp < config.LOW_THRESH_AMP):
        result =  False
    
    if (num_peaks <= 0 or num_peaks >= config.MAX_NUM_PEAKS+1):
        result = False        
    
    return result

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
    
    plt.savefig(os.path.join(source_directory, 'training_history.png'))
    # plt.show()

def plot_confusion_matrix(model, x_val, y_val, labels):
    y_pred = model.predict(x_val)
    y_pred_classes = np.argmax(y_pred, axis=1)  # Convert probabilities to class labels
    cm = confusion_matrix(y_val, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    
    plt.savefig(os.path.join(source_directory, 'confusion_matrix.png'))
    # plt.show()

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

def predict(signal,model, labels):
    segment = extract_peak_segment(signal)                
                
    num_amplitude_peaks = get_num_amplitude_peaks(segment)
    max_amplitude = np.max(np.abs(segment))
    if (use_machine_learning(segment)):          
            predicted_label = predict(segment, model, labels)
            if predicted_label.startswith('OK'):
                return "OK"
            if predicted_label.startswith('NG'):
                return "NG"       
    else:
        return "NG"
    
def train(dataset_dir, epochs, batch_size, patience):    
    labels = config.LABELS[0]
    model_file_path = os.path.join(source_directory, 'model.keras')
    
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
                callbacks=[early_stopping_accuracy, early_stopping_loss], verbose=0)
    
    # Plot training history
    # plot_training_history(history)
    
    # # Evaluate the model and plot confusion matrix
    # plot_confusion_matrix(model, X_val, y_val, labels)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=0)
    # print(f'Test accuracy: {test_acc}')
    
    # Save the model
    model.save(model_file_path)
    
    return test_acc, test_loss
