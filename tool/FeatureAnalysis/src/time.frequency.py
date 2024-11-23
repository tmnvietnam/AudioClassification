import numpy as np
import librosa
import librosa.display
import pywt
import matplotlib.pyplot as plt
import os
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
 

for name in ['OK','NG']:
    input_folder = f"C:/Users/ADMIN/Documents/main/working/Audio.Classification/Tool/audio_classification/data/{name}"
    ogram_folder = f"output/time.frequency/{name}" 

    os.makedirs(ogram_folder, exist_ok=True) 

    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)
                        
            # Load the audio file
            signal, sr = librosa.load(filepath)
            
            segment = extract_peak_segment(signal, sr, 0.15)

            # Compute the spectrogram
            D = librosa.amplitude_to_db(np.abs(librosa.stft(segment)), ref=np.max)

            # Define scales for the scaleogram
            scales = np.arange(1, 128)
            coefficients, frequencies = pywt.cwt(segment, scales, 'cmor')  # Limit length for demonstration

            # Plot both spectrogram and scaleogram side by side
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Plot spectrogram
            img1 = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log', ax=ax1, cmap='viridis')
            ax1.set(title='Spectrogram')
            fig.colorbar(img1, ax=ax1, label='Amplitude (dB)', format='%+2.0f dB')

            # Plot scaleogram
            img2 = ax2.imshow(np.abs(coefficients), extent=[0, 1, 1, 128], cmap='jet', aspect='auto',
                            vmax=abs(coefficients).max())
            ax2.set(title='Scaleogram', xlabel='Time', ylabel='Scale')
            fig.colorbar(img2, ax=ax2, label='Magnitude')

            plt.tight_layout()
            # plt.show()
            
            output_path = os.path.join(ogram_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(output_path)
            plt.close()  # Close the figure to save memory