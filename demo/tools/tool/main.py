import os
import random
from pydub import AudioSegment

# Path to folder A and folder B
folder_a = "ng"
folder_b = "ok"
output_folder = "c"

# Create output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# List all files in both folders
files_a = sorted(os.listdir(folder_a))
files_b = sorted(os.listdir(folder_b))

# Shuffle the files to get random order
random.shuffle(files_a)
random.shuffle(files_b)

# Ensure both folders have the same number of files
min_files = min(len(files_a), len(files_b))

# Loop through and randomly combine files
for i in range(min_files):
    file_a = os.path.join(folder_a, files_a[i])
    file_b = os.path.join(folder_b, files_b[i])
    
    # Load both audio files
    sound_a = AudioSegment.from_file(file_a)
    
    # sound_a = sound_a - 5  # Giảm âm lượng 3 dB

    sound_b = AudioSegment.from_file(file_b)
    
    # Combine the audio files
    combined = sound_a.overlay(sound_b)
    
    # Export combined file to the output folder
    output_file = os.path.join(output_folder, f"0{i+1+80}.wav")
    combined.export(output_file, format="wav")
    
    print(f"Randomly combined file saved as: {output_file}")
