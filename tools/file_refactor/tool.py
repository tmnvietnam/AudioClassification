import os
import shutil

# Path to the folder containing the files
input_path = "C:\\Users\\ADMIN\\Documents\\main\\working\\Audio.Classification\\dataset\\win\\ok"
output_path = "C:\\Users\\ADMIN\\Documents\\main\\working\\Audio.Classification\\dataset\\ok"
os.mkdir(output_path)

# List all files in the folder
files = os.listdir(input_path)

# Sort files if needed (e.g., alphabetically)
files.sort()

# Loop over each file and rename
for i, filename in enumerate(files):
    # Get the file extension (if needed)
    file_extension = os.path.splitext(filename)[1]
    
    # Create the new file name (xxxx followed by a number padded with 4 zeros)
    new_name = f"{i+1:04d}{file_extension}"
    
    # Full path for old and new file names
    old_file = os.path.join(input_path, filename)
    
    new_file = os.path.join(output_path, new_name)
    shutil.copy(old_file,new_file)
    

print("Renaming completed!")
