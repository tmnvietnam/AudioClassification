import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display


for name in ['OK_','NG_']:
        
    input_folder = f"C:\\Users\\ADMIN\\Documents\\main\\working\\Audio.Classification\\Tool\\audio_classification\\data\\{name}"
    output_folder = f"output/short.time.energy/{name}"

    os.makedirs(output_folder, exist_ok=True)

    # Process each .wav file in the input folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".wav"):
            filepath = os.path.join(input_folder, filename)
            
            signal, sr = librosa.load(filepath)
            # Load the audio file

            # Set frame length and hop length (in samples)
            frame_length = 512
            hop_length = 128

            # Compute the short-term energy
            energy = np.array([sum(y[i:i+frame_length]**2) for i in range(0, len(signal), hop_length)])

            # Normalize energy if needed
            energy = energy / np.max(energy)

            # You can also visualize it using librosa.display or matplotlib
            import matplotlib.pyplot as plt
            plt.plot(energy)
            plt.title('Short-Term Energy')
            plt.xlabel('Time (frames)')
            plt.ylabel('Energy')
            
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.png")
            plt.savefig(output_path)
            plt.close()  # Close the figure to save memor