import os
import sys
import io
import logging
import queue
import threading
import time

import win32pipe
import win32file
import win32con

import librosa
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
import sounddevice as sd

import tensorflow as tf
from sklearn.model_selection import train_test_split

import cfg


WORKING_DIR = ''
MODEL_PATH = ''
HISTORY_IMG_PATH = ''

q = queue.Queue()
stop_flag = threading.Event()


def extract_features(audio, sr):    
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))   
    rms = np.mean(librosa.feature.rms(y=audio))    
    peak_amp = np.max(np.abs(audio))    
    mean_amp = np.mean(audio)
    
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    # Apply FFT to convert to the frequency domain
    fft = np.fft.fft(audio, n=sr)  # Compute the FFT
    fft_magnitude = np.abs(fft[:sr // 2])  # Take magnitude of the FFT and only positive frequencies       

    # return np.hstack([ np.array([zcr, rms, peak_amp, mean_amp]), fft_magnitude])
    return np.hstack([ np.array([zcr, rms, peak_amp, mean_amp]),  mfccs, spectral_centroid , spectral_bandwidth, fft_magnitude])


# Function to plot training and validation accuracy and loss
def plot_training_history(history, filename):
    
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
  
    plt.savefig(filename)

    # Optionally, close the plot to free memory
    plt.close()

def load_dataset(data_dir, labels):
    audio_data = []
    audio_labels = []
    
    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                # Load the audio file
                audio, _ = librosa.load(file_path, sr=cfg.SAMPLING_RATE)
               
                features = extract_features(audio, cfg.SAMPLING_RATE)
                 
                features = features /  np.max(features)               

                audio_data.append(features)
                audio_labels.append(labels.index(label))
    
    audio_data = np.array(audio_data)
    audio_labels = np.array(audio_labels)
    return audio_data, audio_labels

# 2. Create a fully connected neural network model
def create_dense_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([ 
        tf.keras.layers.InputLayer(input_shape=input_shape), 
        tf.keras.layers.Dense(500, activation='relu'),
        tf.keras.layers.Dropout(0.5),                  
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dropout(0.5),        
        tf.keras.layers.Dense(150, activation='relu'),
        tf.keras.layers.Dropout(0.5),        
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(60, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])    
    
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    
# 3. Main code to execute training
def train(dataset_path, epochs, batch_size):
    data_dir = dataset_path    
    
    # Load the audio data and extract frequency components using FFT
    X, y = load_dataset(data_dir, cfg.LABELS)    
   
    X = X / np.max(X)  # Scale the features between 0 and 1

    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    input_shape = (X_train.shape[1],)  # Input shape is based on FFT size
    num_classes = len(cfg.LABELS)
    model = create_dense_model(input_shape, num_classes)       
    
    # Train the model
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=epochs, batch_size=batch_size)
    
    # Plot training history
    plot_training_history(history, HISTORY_IMG_PATH)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f"Test accuracy: {test_acc}")
    print(f"Test loss: {test_loss}")

    # Save the model
    model.save(MODEL_PATH)
    
    return test_acc, test_loss

# 4. Load the trained model
def load_model(model_file):
    return tf.keras.models.load_model(model_file)

# 6. Predict the label of a new audio file
def predict_audio_file(file_path, model_file, labels):
    # Load the model
    model = load_model(model_file)       
    
    # _, audio_filter = wavfile.read(file_path)
    
    recorded_audio, _ = librosa.load(file_path, sr=cfg.SAMPLING_RATE)   
       
    for i in range(int((2*cfg.T-1)*cfg.N+1)):
        segment_audio = recorded_audio[i*len(recorded_audio)//(int(2*cfg.T*cfg.N)):(i+cfg.N)*len(recorded_audio)//(int(2*cfg.T*cfg.N))]
        segment_audio_filter = audio_filter[i*len(audio_filter)//(N*4):(i+4)*len(audio_filter)//(N*4)]
        
        features = extract_features(audio, cfg.SAMPLING_RATE)        
        
        features = features / np.max(features)
        
        # Reshape the features to match the input shape of the model
        input_data = features.reshape(1, -1)        
        
        # Predict using the trained model
        predictions = model.predict(input_data, verbose=0)
        
        # Return the predicted class
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        
        # Get the label name from the index
        predicted_label = labels[predicted_label_index]       
    
        if (predicted_label == "ok" ):
            return "ok"        
       
    return "ng"     

# 7. Example usage for prediction
def predict(wav_index , model_path, target_label_name):    
    file_name = f'{wav_index}.wav'   
    wave_path = os.path.join(os.path.join(WORKING_DIR, 'audio'), file_name)   

    predicted_label = predict_audio_file(wave_path, model_path, cfg.LABELS)
    return predicted_label == target_label_name

def predict_audio_data(audio_data ,model_file, labels):
    
    model = load_model(model_file)       

    features = extract_features(audio_data, cfg.SAMPLING_RATE)        
    
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
        
    buffer = np.array([])  # Initial empty buffer to hold incoming samples
    window_index = 0
    while not stop_flag.is_set() or not q.empty():
        if not q.empty():
            # Get the next chunk of samples from the queue
            new_samples = q.get()

            # Append the new samples to the buffer
            buffer = np.concatenate((buffer, new_samples.flatten()))

            # Process the buffer only if we have at least window_size samples
            while len(buffer) >= cfg.WINDOW_SIZE:
                # Extract the current window
                current_window = buffer[:cfg.WINDOW_SIZE]
                
                predicted_label = predict_audio_data(current_window, model, cfg.LABELS)                    
              
                predicted_result = predicted_label == "ok"                              
                    
                current_window_normalized = current_window / np.max(np.abs(current_window))     
                
                response = f'response:result:{predicted_result}:end'
                # response = f'response:result:{predicted_result}:sounddata:{",".join(map(str, current_window_normalized))}:end'
                win32file.WriteFile(pipe, bytes(response, "utf-8"))

                if (predicted_result):   
                    stop_flag.set()  # Stop the recording                                   
                    break
                                    
                # Slide the window forward by step_size
                buffer = buffer[cfg.STEP_SIZE:]
                window_index = window_index + 1

        else:
            time.sleep(0.1)  # Avoid busy-waiting if the queue is empty
            
    response = 'endingtest'
    win32file.WriteFile(pipe, bytes(response, "utf-8"))

            
def record_sound(selected_device):
    
    start_time = time.time()  # Record the start time
    max_duration = 30  # Maximum recording duration in seconds
    
    def callback(indata, frames, time, status):                
        q.put(indata.copy())

    with sd.InputStream(callback=callback, channels=1, samplerate=cfg.SAMPLING_RATE, device=selected_device):
        while not stop_flag.is_set() and (time.time() - start_time) < max_duration:
            time.sleep(0.1)  # Keep recording until stop_flag is set
    
    stop_flag.set() 
    
def get_device_by_name(name):
    devices = sd.query_devices()
    for device in devices:
        if name.lower() in device['name'].lower():  # Case-insensitive match
            return device
    return None  # Return None if no match found


def main():    
    if len(sys.argv) < 2:        
        exit()
    else:
        PIPE_NAME = rf'\\.\pipe\{sys.argv[1] }'               
        
    global WORKING_DIR
    global MODEL_PATH
    global HISTORY_IMG_PATH
    
    home_directory = os.path.expanduser("~")
    WORKING_DIR = os.path.join(home_directory, ".soundkit")
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORKING_DIR, "audio"), exist_ok=True)
    
    MODEL_PATH = os.path.join(WORKING_DIR, 'model.h5')    
    HISTORY_IMG_PATH = os.path.join(WORKING_DIR, 'history.png')
    
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
        
        if(data.decode().startswith("init@")):                       
            response = f'response:{WORKING_DIR}'
            win32file.WriteFile(pipe, bytes(response, "utf-8"))
            print(response)            
            # Example usage
            matched_device = get_device_by_name(device_name)
              
        if(data.decode().startswith("predict@")):
            
            wav_index = data.decode().split("@")[1]  
            model_path= data.decode().split("@")[2]              
            
            result = predict(int(wav_index), model_path,"ok") 
            response = f'response:{result}'            
            win32file.WriteFile(pipe, bytes(response, "utf-8"))
            print(response)
            
        if(data.decode().startswith("test@")):
            
            model_path= data.decode().split("@")[1]  
            selected_device= int(data.decode().split("@")[2])
 
            model = tf.keras.models.load_model(model_path)
            
         
            # selected_device = None
            
              # Create threads for recording and sliding window detection
            detect_thread = threading.Thread(target=sliding_window_detection, args=(model,))
            record_thread = threading.Thread(target=record_sound, args=(selected_device,))
            
            # Start the threads
            detect_thread.start()
            record_thread.start()

            # Wait for the threads to finish
            detect_thread.join()
            record_thread.join()    
            
          
            
        if(data.decode().startswith("train@")):
            
            dataset_path = data.decode().split("@")[1]      
            epochs = data.decode().split("@")[2]              
            batch_size = data.decode().split("@")[3]              

            accuracy, loss = train(dataset_path, int(epochs), int(batch_size))
            
            response = f'response:{accuracy}:{loss}'
            win32file.WriteFile(pipe, bytes(response, "utf-8"))
            print(response)
            
        # Clean up
        win32file.CloseHandle(pipe)

if __name__ == "__main__":
    main()