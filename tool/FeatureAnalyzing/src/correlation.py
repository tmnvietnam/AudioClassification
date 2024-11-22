import os
import numpy as np
import librosa
from scipy.signal import correlate, hilbert, find_peaks, windows

import matplotlib.pyplot as plt

# Define the folder paths
ng_folder = "C:/Users/ADMIN/Documents/main/working/Audio.Classification/Tool/main/data/NG"
ok_folder = "C:/Users/ADMIN/Documents/main/working/Audio.Classification/Tool/main/data/OK"
output_folder = "output/correlation"
os.makedirs(output_folder, exist_ok=True)

# Get list of files in each folder
files1 = [os.path.join(ng_folder, f) for f in os.listdir(ng_folder) if f.endswith('.wav')]
files2 = [os.path.join(ok_folder, f) for f in os.listdir(ok_folder) if f.endswith('.wav')]

# Initialize a matrix to store correlation values
correlation_matrix = np.zeros((len(files1), len(files2)))


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

# Function to compute correlation between two signals
def compute_correlation(signal1, signal2 ,sr1, sr2, duration = 0.15):
    
    segment1 = extract_peak_segment(signal1 , sr1, duration)
    segment2 = extract_peak_segment(signal2 , sr2, duration)
    # Make sure both signals are the same length
    # Compute cross-correlation
    window_size1 = len(segment1)
    gaussian_window1 = windows.gaussian(window_size1, std=window_size1/6)
    windowed_segment1= segment1 * gaussian_window1
    fft_magnitude1 = np.abs(np.fft.fft(windowed_segment1, n=sr1)[:sr1 // 2])
    
    window_size2 = len(segment2)
    gaussian_window2 = windows.gaussian(window_size2, std=window_size2/6)
    windowed_segment2= segment2 * gaussian_window2
    fft_magnitude2 = np.abs(np.fft.fft(windowed_segment2, n=sr2)[:sr2 // 2])

    correlation = correlate(fft_magnitude1, fft_magnitude2, mode='full')
    # Return the maximum correlation value
    return np.max(correlation)

for i, file1 in enumerate(files1):
    signal1, sr1 = librosa.load(file1, sr=None)
    
    for j, file2 in enumerate(files2):
        signal2, sr2 = librosa.load(file2, sr=None)
        
        # Compute correlation and store in matrix
        correlation_matrix[i, j] = compute_correlation(signal1, signal2 ,sr1, sr2)
# Normalize correlation values to get percentage
max_corr_value = np.abs(correlation_matrix).max()  # Find absolute max across the matrix
correlation_percentage_matrix = (correlation_matrix / max_corr_value) * 100

# Create a binary mask for the second graph
binary_threshold = 60  # Threshold in percentage
binary_matrix = np.where(correlation_percentage_matrix >= binary_threshold, 1, 0)

# Plot the two heatmaps
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(56, 24))

# First heatmap: Full percentage correlation matrix
cax1 = ax1.imshow(correlation_percentage_matrix, aspect='auto', cmap='viridis', vmin=0, vmax=100)
fig.colorbar(cax1, ax=ax1, label='Correlation (%)')
ax1.set_title('Correlation Percentage Matrix')
ax2.set_xlabel('Files in OK folder')
ax2.set_ylabel('Files in NG folder')
ax1.set_xticks(np.arange(len(files2)))
ax1.set_yticks(np.arange(len(files1)))
ax1.set_xticklabels([os.path.basename(f) for f in files2], rotation=90, fontsize=10)
ax1.set_yticklabels([os.path.basename(f) for f in files1], fontsize=10)

cax2 = ax2.imshow(binary_matrix, aspect='auto', cmap='gray', vmin=0, vmax=1)
ax2.set_title(f'Binary Correlation Matrix (Black < {binary_threshold}%, White >= {binary_threshold}%)')
ax2.set_xlabel('Files in OK folder')
ax2.set_ylabel('Files in NG folder')
ax2.set_xticks(np.arange(len(files2)))
ax2.set_yticks(np.arange(len(files1)))
ax2.set_xticklabels([os.path.basename(f) for f in files2], rotation=90, fontsize=10)
ax2.set_yticklabels([os.path.basename(f) for f in files1], fontsize=10)

plt.tight_layout()
# plt.show()


output_path = os.path.join(output_folder, "plot.png")
plt.savefig(output_path)