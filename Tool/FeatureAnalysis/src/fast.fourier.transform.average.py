import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
from scipy.signal.windows import gaussian
from scipy.signal import hilbert, find_peaks


def extract_peak_segment(signal, sr, duration):
    # Calculate the envelope using Hilbert Transform
    analytic_signal = hilbert(signal)
    envelope = np.abs(analytic_signal)

    # Find peaks in the envelope
    peaks, _ = find_peaks(envelope)

    # Find the index of the highest peak
    highest_peak_idx = peaks[np.argmax(envelope[peaks])]

    # Calculate the half-window size in samples
    half_window_samples = int((duration * sr) / 2)

    # Extract the segment around the peak
    start_index = max(0, highest_peak_idx - half_window_samples)
    end_index = min(len(signal), highest_peak_idx + half_window_samples)
    segment = signal[start_index:end_index]

    return segment


# Process categories
for name in ['OK', 'NG', 'NG.PCB']:
    input_folder = f"C:/Users/ADMIN/Documents/main/working/Audio.Classification/Dataset/{name}"
    output_folder = f"output/fast.fourier.transform.average"

    os.makedirs(output_folder, exist_ok=True)

    plt.figure(figsize=(18, 9))

    fft_sum = None  # To accumulate FFT magnitudes
    file_count = 0  # To count the number of files processed

    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)

            signal, sr = librosa.load(filepath)

            # Extract peak segment
            segment = extract_peak_segment(signal, sr, 0.15)

            # Apply Gaussian window
            window_size = len(segment)
            gaussian_window = gaussian(window_size, std=window_size / 6)
            windowed_segment = segment * gaussian_window

            # Compute FFT magnitude
            fft_magnitude = np.abs(np.fft.fft(windowed_segment, n=sr)[:sr // 2])

            # Accumulate FFT magnitudes
            if fft_sum is None:
                fft_sum = fft_magnitude
            else:
                fft_sum += fft_magnitude

            file_count += 1

    # Calculate the average FFT magnitude
    if file_count > 0:
        fft_average = fft_sum / file_count
        frequencies = np.linspace(0, sr / 2, len(fft_average))

        # Plot the average FFT
        plt.plot(frequencies, fft_average, label=f"Average FFT for {name}")

    plt.title(f"Average FFT for {name} category")
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(left=0, right=sr / 2)  # Set frequency range from 0 to Nyquist
    plt.ylim(bottom=0)  # Adjust magnitude axis as needed

    # Save the figure as a PNG
    output_path = os.path.join(output_folder, f"{name}.png")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()  # Close the figure to save memory
