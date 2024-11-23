
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
from scipy.signal import hilbert

# name = "NG_CON"
for name in ['OK','NG']:
    input_folder = f"C:/Users/ADMIN/Documents/main/working/Audio.Classification/Tool/main/data/{name}"
    output_folder = f"output/hilbert.transform/{name}"
    cutoff_freq = 160

    os.makedirs(output_folder, exist_ok=True)
    
    max_envelopes = []

    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)

            signal, sr = librosa.load(filepath)
            
            analytic_signal = hilbert(signal)

            # Compute the amplitude envelope (magnitude of the analytic signal)
            envelope = np.abs(analytic_signal)
            
            
            # print(f'{name}:{filename}:{max_envelope}')

            # Plot the signal and its amplitude envelope
            plt.figure(figsize=(12, 6))
            plt.subplot(2, 1, 1)
            librosa.display.waveshow(signal, sr=sr)
            plt.title("Original Signal")

            plt.subplot(2, 1, 2)
            plt.plot(envelope, label='Amplitude Envelope', color='r')
            plt.title("Amplitude Envelope")
            plt.xlabel('Time (samples)')
            plt.ylabel('Amplitude')
            plt.tight_layout()
            
  
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(output_path)
            plt.close()  # Close the figure to save memor
    