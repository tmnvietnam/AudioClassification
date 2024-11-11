import os
import shutil

# Path to the folder containing the files
src_path = "in"
start_index = 1

os.makedirs("out", exist_ok=True)

# List all files in the folder
files = os.listdir(src_path)

# Sort files if needed (e.g., alphabetically)
files.sort()

# Loop over each file and rename
for i, filename in enumerate(files):
    # Get the file extension (if needed)
    file_extension = os.path.splitext(filename)[1]
    
    # Create the new file name (xxxx followed by a number padded with 4 zeros)
    new_name = f"{i+start_index:04d}{file_extension}"
    
    # Full path for old and new file names
    old_file = os.path.join(src_path, filename)
    
    new_file = os.path.join("out", new_name)
    shutil.copy(old_file,new_file)
    

