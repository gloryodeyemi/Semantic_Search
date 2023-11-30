import os
import shutil

# Source directory containing subfolders with text files
source_folder = 'data/2_short'

# Destination directory to collect all text files
destination_folder = 'data/data_new'

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Iterate through all subfolders and move text files to the destination folder
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file.endswith('.txt'):
            file_path = os.path.join(root, file)
            shutil.copy(file_path, destination_folder)

print("All text files moved to the destination folder.")
