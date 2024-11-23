import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, find_peaks
import librosa


height= 0.3
distance = 250
threshold = 0.02
names = ["NG", "OK"]


def extract_peak_segment(signal, sr, duration):
    # Sử dụng Hilbert Transform để tìm phong bì (envelope)
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    # Tìm vị trí của các đỉnh
    peaks, _ = find_peaks(envelope)    

    # Tìm đỉnh cao nhất
    highest_peak_idx = peaks[np.argmax(envelope[peaks])]
    
    half_window_samples = int((duration * sr) / 2)
    
    # Get the segment around the peak
    start_index = max(0, highest_peak_idx - half_window_samples)
    end_index = min(len(signal), highest_peak_idx + half_window_samples)
    segment = signal[start_index:end_index]
    
    return segment 

for name in names:
    input_folder = f"C:/Users/ADMIN/Documents/main/working/Audio.Classification/Tool/main/data/{name}"
    peaks_folder = f"output/find.peeks/{name}"  # New folder for peaks

    os.makedirs(peaks_folder, exist_ok=True)  # Create peaks folder

    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)
            
            # Load the .wav file
            signal, sample_rate = librosa.load(filepath)
            
            segment = extract_peak_segment(signal, sample_rate, 0.15)

            # Time-domain plot of the filtered signal
            time = np.linspace(0, len(segment), num=len(segment))
            # Find peaks in the time-domain signal
            peaks, _ = find_peaks(segment, height=height, distance=distance, threshold=threshold)  # Adjust height and distance as needed
            
            print(f'{name}::{filename}:{len(peaks)}:{np.max(np.abs(segment)):.4f}')

            # Plot the filtered signal with detected peaks
            plt.figure(figsize=(10, 4))
            plt.plot(time, segment, label='Filtered Signal')
            plt.plot(time[peaks], segment[peaks], "x", label='Peaks', color='red')
            plt.title(f'Time Domain of Filtered {filename} with Peaks')
            plt.ylim(-1, 1)

            plt.xlabel('Time (s)')
            plt.ylabel('Amplitude')
            plt.legend()

            # Save the time-domain plot with peaks as PNG in the peaks folder
            peak_time_domain_output_path = os.path.join(peaks_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(peak_time_domain_output_path)
            plt.close()  # Close the figure to save memory
