import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

import win32pipe
import win32file
import win32con

WORKING_DIR = ''
MODEL_PATH = ''
PIPE_NAME = r'//./pipe/TensorflowService'     

model = None

import numpy as np
import matplotlib.pyplot as plt
import librosa
from scipy.signal import find_peaks, hilbert, windows

# Get the absolute path of the current script
current_file_path = os.path.abspath(__file__)
# Get the directory of the current script
source_directory = os.path.dirname(current_file_path)    

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



    
def predict(signal, model, labels ):
    
    predicted_label = 'NG'
    segment = extract_peak_segment(signal)                
                
    # num_amplitude_peaks = get_num_amplitude_peaks(segment)
    # max_amplitude = np.max(np.abs(segment))
    if (use_machine_learning(segment)):     
       
        features = extract_features(segment)        

        features = features / np.max(features, axis=0)    

        # Reshape the features to match the input shape of the model
        input_data = features.reshape(1, -1)    
        
        # Predict using the trained model
        predictions = model.predict(input_data, verbose=0)
        
        # Return the predicted class
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        
        # Get the label name from the index
        predicted_label = labels[predicted_label_index]
        if predicted_label.startswith('OK'):
            return "OK"
        if predicted_label.startswith('NG'):
            return "NG"
       

    return predicted_label

def main():
    global WORKING_DIR
    global MODEL_PATH
    global model
    
    # home_directory = os.path.expanduser("~")
    # WORKING_DIR = os.path.join(home_directory, ".tensorflow_service")
    
    # os.makedirs(WORKING_DIR, exist_ok=True)
    # os.makedirs(os.path.join(WORKING_DIR, "audio"), exist_ok=True)
    
    WORKING_DIR = 'C:/Users/ADMIN/Documents/main/working/Audio.Classification/Tool/main/.tensorflow_service'
    
    MODEL_PATH = os.path.join(WORKING_DIR, 'model_0.keras')
    

    model = tf.keras.models.load_model(MODEL_PATH)

    from tensorflow.python.client import device_lib
    print(device_lib.list_local_devices())

    
    while (True):        
        # Create a named pipe
        pipe = win32pipe.CreateNamedPipe(
            PIPE_NAME,
            win32pipe.PIPE_ACCESS_DUPLEX,
            win32pipe.PIPE_TYPE_MESSAGE | win32pipe.PIPE_READMODE_MESSAGE | win32pipe.PIPE_WAIT,
            1, 65536, 65536, 0, None
        )
        
        print("Waiting for next request...")
        win32pipe.ConnectNamedPipe(pipe, None)

        # Read from the pipe
        hr, data = win32file.ReadFile(pipe, 64*1024)
              
        if(data.decode().startswith("predict@")):
            
            wav_index = data.decode().split("@")[1]  
            
            file_name = f'{wav_index}.wav'   
            wave_path = os.path.join(WORKING_DIR, file_name)   
                        
            signal, _ = librosa.load(wave_path)   
            
            result = predict(signal,model, LABELS)
            
            response = f'response:{result}'            
            win32file.WriteFile(pipe, bytes(response, "utf-8"))
            print(response)
            

            
        # Clean up
        win32file.CloseHandle(pipe)

if __name__ == "__main__":
    main()