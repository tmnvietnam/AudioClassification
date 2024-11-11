import os
import numpy as np
import librosa
import soundfile as sf

import numpy as np

def time_shift(audio, shift_max):
    shift = np.random.randint(-shift_max, shift_max)
    return np.roll(audio, shift)

def add_noise(audio, noise_factor=0.0000005):
    """Add random noise to the audio."""
    noise = np.random.randn(len(audio))
    return audio + noise_factor * noise

path = '..\\audio_classification_wavelets\\data\\OK_'
path_output = "output_folder"


# List audio files in both folders
files = [f for f in os.listdir(path) if f.endswith('.wav')]


for file_name in files:
    try:
        # Load Sound A and Sound B
        y, sr = librosa.load(os.path.join(path, f'{file_name}'), sr=None)
        audio_  = time_shift(y,int(sr * 0.05))
        audio_ = add_noise(audio_)
        
        # Save the combined sound
        output_file = os.path.join(path_output, f'New_{file_name}')
        sf.write(output_file, audio_, sr)
        print(f'Saved combined sound: {output_file}')

    except Exception as e:
        print(f"Error processing file {file_name}: {e}")
