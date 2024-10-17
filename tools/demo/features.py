import numpy as np
import librosa

def extract_frequency_domain_features(audio, sr):
    # Apply FFT to convert to the frequency domain
    fft = np.fft.fft(audio, n=sr)  # Compute the FFT
    fft_magnitude = np.abs(fft[:sr // 2])  # Take magnitude of the FFT and only positive frequencies       
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio, sr=sr))
    # spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    # mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sr), axis=1)
    return np.hstack([ spectral_bandwidth, fft_magnitude])


def extract_time_domain_features(audio, sr):
    zcr = np.mean(librosa.feature.zero_crossing_rate(audio))   
    rms = np.mean(librosa.feature.rms(y=audio))    
    peak_amp = np.max(np.abs(audio))    
    mean_amp = np.mean(audio)
        
    return np.array([zcr, rms, peak_amp, mean_amp])
