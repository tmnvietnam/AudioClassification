import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display

def extract_fft_energy_distribution_features(signal, sr, duration=0.5):      
    # Find the peak of the signal
    peak_index = np.argmax(np.abs(signal))
    peak_amplitude = np.max(np.abs(signal))
    peak_time = peak_index / sr

    # Calculate the number of samples for the desired duration
    half_window_samples = int((duration * sr) / 2)
    
    # Get the segment around the peak
    start_index = max(0, peak_index - half_window_samples)
    end_index = min(len(signal), peak_index + half_window_samples)
    segment = signal[start_index:end_index]
    
    # Compute the FFT of the segment
    fft_result = np.fft.fft(segment)
    
    # Compute the energy (square of the magnitude of the FFT)
    energy_fft = np.abs(fft_result)**2

    # Compute frequencies corresponding to the FFT
    frequencies = np.fft.fftfreq(len(segment), d=1/sr)

    # Only take the positive frequencies
    positive_frequencies = frequencies[:len(frequencies)//2]
    positive_energy_fft = energy_fft[:len(energy_fft)//2]  # Slice for positive frequencies

    return positive_frequencies, positive_energy_fft

# name = "NG_CON"
for name in ['OK_','NG_']:
    input_folder = f"C:/Users/ADMIN/Documents/main/working/Audio.Classification/Tool/audio_classification/data/{name}"
    output_folder = f"output/energy/{name}"

    os.makedirs(output_folder, exist_ok=True)

    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)
            
            signal, sr = librosa.load(filepath)

            positive_frequencies, positive_energy_frequency= extract_fft_energy_distribution_features(signal,sr)

            # Plot the energy distribution up to the cutoff frequency
            plt.figure(figsize=(10, 6))
            plt.plot(positive_frequencies, positive_energy_frequency)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Energy')
            plt.title('Frequency-domain Energy Distribution')
            plt.xlim(left=100,right=4000)
            plt.ylim(bottom=0,top=3000)
           
            
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(output_path)
            plt.close()  # Close the figure to save memor