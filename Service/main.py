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
PIPE_NAME = r'\\.\pipe\TensorflowService'     
N = 4

# 1. Load and preprocess the data in the frequency domain using FFT
def load_audio_files_fft(data_dir, labels, sr=22050):
    audio_data = []
    audio_labels = []
    
    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                # Load the audio file
                audio, _ = librosa.load(file_path, sr=sr)
                # Apply FFT to convert to the frequency domain
                fft = np.fft.fft(audio, n=sr)  # Compute the FFT
                fft_magnitude = np.abs(fft[:sr // 2])  # Take magnitude of the FFT and only positive frequencies
                audio_data.append(fft_magnitude)
                audio_labels.append(labels.index(label))
    
    audio_data = np.array(audio_data)
    audio_labels = np.array(audio_labels)
    return audio_data, audio_labels

# 2. Create a fully connected neural network model
def create_dense_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(22050//10, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(22050//50, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(22050//150, activation='relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model
    
# 3. Main code to execute training
def train(dataset_path):
    data_dir = dataset_path
    labels = ["ng", "ok"]  # Replace with your actual labels
    
    # Load the audio data and extract frequency components using FFT
    X, y = load_audio_files_fft(data_dir, labels)
    
    # Normalize the data
    X = X / np.max(X)  # Scale the features between 0 and 1
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Create the model
    input_shape = (X_train.shape[1],)  # Input shape is based on FFT size
    num_classes = len(labels)
    model = create_dense_model(input_shape, num_classes)       
    
    # Train the model
    history = model.fit(X_train, y_train, 
                    validation_data=(X_val, y_val),
                    epochs=20,
                    batch_size=32)
    
    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f"Test accuracy: {test_acc}")

    # Save the model
    model.save(MODEL_PATH)
    
    return test_acc

# 4. Load the trained model
def load_model(model_file):
    return tf.keras.models.load_model(model_file)

# 5. Preprocess the audio using FFT
def preprocess_audio_fft(audio, sr=22050):
    # Load the audio file    
    # Apply FFT to convert to the frequency domain
    fft = np.fft.fft(audio, n=sr)  # Compute the FFT
    fft_magnitude = np.abs(fft[:sr // 2])  # Take magnitude of the FFT and only positive frequencies
    # Normalize the data (same normalization as in training)
    fft_magnitude = fft_magnitude / np.max(fft_magnitude)
    
    return fft_magnitude

# 6. Predict the label of a new audio file
def predict_audio_file(file_path, model_file, labels, sr=22050):
    # Load the model
    model = load_model(model_file)       
    
    audio, _ = librosa.load(file_path, sr=sr)   
    
    # for i in range((N*2)-1):
    #     segment_audio = audio[i*len(audio)//(N*2):(i+2)*len(audio)//(N*2)]
    for i in range((N*4)-1):
        segment_audio = audio[i*len(audio)//(N*4):(i+4)*len(audio)//(N*4)]
        
        fft_magnitude = preprocess_audio_fft(segment_audio, sr=sr)
    
        # Reshape the data to match the input shape of the model (1 sample, input_length)
        input_data = fft_magnitude.reshape(1, -1)
        
        # Predict the label
        predictions = model.predict(input_data)
        predicted_label_index = np.argmax(predictions, axis=1)[0]
        
        # Map the predicted index back to the corresponding label
        predicted_label = labels[predicted_label_index]

        if (predicted_label == "ok"):
            return "ok"           
   
    return "ng"       
        

# 7. Example usage for prediction
def predict(wav_index , model_path, target_label_name):
    labels = ["ng", "ok"]  # Replace with your actual labels
    
    file_name = f'{wav_index}.wav'   
    wave_path = os.path.join(os.path.join(WORKING_DIR, 'audio'), file_name)   

    predicted_label = predict_audio_file(wave_path, model_path, labels)
    return predicted_label == target_label_name

def main():
    global WORKING_DIR
    global MODEL_PATH
    
    home_directory = os.path.expanduser("~")
    WORKING_DIR = os.path.join(home_directory, ".tensorflow_service")
    os.makedirs(WORKING_DIR, exist_ok=True)
    os.makedirs(os.path.join(WORKING_DIR, "audio"), exist_ok=True)
    
    MODEL_PATH = os.path.join(WORKING_DIR, 'model.h5')
    
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
              
        if(data.decode().startswith("predict@")):
            
            wav_index = data.decode().split("@")[1]  
            model_path= data.decode().split("@")[2]              
            
            result = predict(int(wav_index), model_path,"ok") 
            response = f'response:{result}'            
            win32file.WriteFile(pipe, bytes(response, "utf-8"))
            print(response)
            
        if(data.decode().startswith("train@")):
            
            dataset_path = data.decode().split("@")[1]              
            accuracy = train(dataset_path)
            
            response = f'response:{accuracy}'
            win32file.WriteFile(pipe, bytes(response, "utf-8"))
            print(response)
            
        # Clean up
        win32file.CloseHandle(pipe)

if __name__ == "__main__":
    main()