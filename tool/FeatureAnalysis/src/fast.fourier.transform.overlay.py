
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
from scipy.signal.windows import gaussian  # Corrected import
from scipy.signal import hilbert, find_peaks


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

# name = "NG_CON"
for name in ['OK','NG', 'NG.PCB']:
    input_folder = f"C:/Users/ADMIN/Documents/main/working/Audio.Classification/Dataset/{name}"
    output_folder = f"output/fast.fourier.transform.overlay"

    os.makedirs(output_folder, exist_ok=True)
    
    plt.figure(figsize=(36, 18))


    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)

            signal, sr = librosa.load(filepath)
            
            segment = extract_peak_segment(signal, sr, 0.15)
            window_size = len(segment)
            gaussian_window = gaussian(window_size, std=window_size/6)
            windowed_segment = segment * gaussian_window
            fft_magnitude = np.abs(np.fft.fft(windowed_segment, n=sr)[:sr // 2])
            
            # print(f'{name}:{filename}:{max_envelope}')

            # Plot the signal and its amplitude envelope
            plt.plot(np.linspace(0, sr / 2, len(fft_magnitude)), fft_magnitude, label=f"{filename}")

    plt.title(f"Overlay of FFT for {name} category")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(left=0, right=sr / 2)  # Set frequency range from 0 to Nyquist
    plt.ylim(bottom=0, top=60)  # Adjust the magnitude axis if needed
    # plt.legend(loc='upper right')  # Show legend with filenames

    # Save the figure as a PNG
    output_path = os.path.join(output_folder, f"{name}.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the figure to save memory
    
