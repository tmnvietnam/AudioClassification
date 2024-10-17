import os
import sys

import librosa
import sounddevice as sd

import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf    
from sklearn.model_selection import train_test_split
from termcolor import colored

import features
import cfg

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
               
                frequency_features = features.extract_frequency_domain_features(audio, cfg.SAMPLING_RATE)
                time_features = features.extract_time_domain_features(audio, cfg.SAMPLING_RATE)

                combined_features = np.hstack([time_features, frequency_features])                
                combined_features = combined_features /  np.max(combined_features)               


                audio_data.append(combined_features)
                audio_labels.append(labels.index(label))
    
    audio_data = np.array(audio_data)
    audio_labels = np.array(audio_labels)
    return audio_data, audio_labels

def create_dense_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dropout(0.5),        
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dropout(0.6),        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
   
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size)
    return history

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

    plt.show()

def predict(model, audio, labels ):

    # Preprocess the audio (FFT)
    frequency_features = features.extract_frequency_domain_features(audio, cfg.SAMPLING_RATE)
    time_features = features.extract_time_domain_features(audio, cfg.SAMPLING_RATE)
    
    # Combine features
    combined_features = np.hstack([time_features, frequency_features])
    
    combined_features = combined_features / np.max(combined_features)
    # Reshape the features to match the input shape of the model
    input_data = combined_features.reshape(1, -1)
    
    
    # Predict using the trained model
    predictions = model.predict(input_data, verbose=0)
    
    # Return the predicted class
    predicted_label_index = np.argmax(predictions, axis=1)[0]
    
    # Get the label name from the index
    predicted_label = labels[predicted_label_index]
    
    return predicted_label

def predict_scan(recorded_audio, model):
    for i in range(int((2*cfg.T-1)*cfg.N+1)):
        segment_audio = recorded_audio[i*len(recorded_audio)//(int(2*cfg.T*cfg.N)):(i+cfg.N)*len(recorded_audio)//(int(2*cfg.T*cfg.N))]
                        
        # Classify the recorded audio
        predicted_label = predict(model, segment_audio, cfg.LABELS)

        if (predicted_label == "ok"):         
            print(colored("OK", "green"))   
            return
    print(colored("NG", "red"))   
   
def record(duration):
    print(f"Start Recording")
    recorded_audio = sd.rec(int(duration * cfg.SAMPLING_RATE), samplerate=cfg.SAMPLING_RATE, channels=1)
    sd.wait()  # Wait for the recording to finish
    recorded_audio = recorded_audio.flatten()  # Convert 2D array to 1D
    print("Stop Recording")
    return recorded_audio

if __name__ == "__main__": 
    
    if len(sys.argv) < 2:        
        function_name = None
    else:
        function_name = sys.argv[1]    

    if(function_name=='train' or function_name==None):
        
        # Load the audio data and extract frequency components using FFT
        X, y = load_dataset(cfg.DATASET_DIR, labels = cfg.LABELS)   
               
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Create the model
        input_shape = (X_train.shape[1],)  # Input shape is based on FFT size
        model = create_dense_model(input_shape, len(cfg.LABELS))
        
        # Train the model
        history = train_model(model, X_train, y_train, X_val, y_val, cfg.EPOCHS, cfg.BATCH_SIZE)
        
        # Plot training history
        plot_training_history(history)

        # Evaluate the model
        test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
        print(f"Test accuracy: {test_acc}")

        # Save the model
        model.save(cfg.MODEL_FILE)
        
    if (function_name=='test' or function_name==None):

            # Load the trained model
        model = tf.keras.models.load_model(cfg.MODEL_FILE)
        
        while(True):            
            predict_scan(record(cfg.T), model)
           


