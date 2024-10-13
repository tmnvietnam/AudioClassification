import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

SAMPLING_RATE = 22050

# 1. Load and preprocess the data in the frequency domain using FFT
def load_audio_files_fft(data_dir, labels):
    audio_data = []
    audio_labels = []
    
    for label in labels:
        folder_path = os.path.join(data_dir, label)
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                file_path = os.path.join(folder_path, file)
                # Load the audio file
                audio, _ = librosa.load(file_path, sr=SAMPLING_RATE)
                # Apply FFT to convert to the frequency domain
                fft = np.fft.fft(audio, n=SAMPLING_RATE)  # Compute the FFT
                fft_magnitude = np.abs(fft[:SAMPLING_RATE // 2])  # Take magnitude of the FFT and only positive frequencies
                audio_data.append(fft_magnitude)
                audio_labels.append(labels.index(label))
    
    audio_data = np.array(audio_data)
    audio_labels = np.array(audio_labels)
    return audio_data, audio_labels

# 2. Create a fully connected neural network model
def create_dense_model(input_shape, num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(600, activation='relu'),
        tf.keras.layers.Dropout(0.5),        
        tf.keras.layers.Dense(300, activation='relu'),
        tf.keras.layers.Dropout(0.6),        
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
   
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# 3. Train the model
def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size):
    history = model.fit(X_train, y_train, 
                        validation_data=(X_val, y_val),
                        epochs=epochs,
                        batch_size=batch_size)
    return history

# Function to plot training and validation accuracy and loss
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

# 4. Main code to execute
if __name__ == "__main__":
    data_dir = "C:/Users/ADMIN/Documents/main/working/Audio.Classification/dataset/new"
    labels = ["ng", "ok"]  # Replace with your actual labels
    
    # Load the audio data and extract frequency components using FFT
    X, y = load_audio_files_fft(data_dir, labels)
    
    # Normalize the data
    X = X / np.max(X)  # Scale the features between 0 and 1
    
    # Split into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create the model
    input_shape = (X_train.shape[1],)  # Input shape is based on FFT size
    num_classes = len(labels)
    model = create_dense_model(input_shape, num_classes)
    
    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val, 64, 16)
    
    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_val, y_val, verbose=2)
    print(f"Test accuracy: {test_acc}")

    # Save the model
    model.save("model.h5")
