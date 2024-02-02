import os
import shutil

from config import *

def copy_files(source_folder, target_folder, file_extensions):
    for root, _, files in os.walk(source_folder):
        for file in files:
            target_folder = os.path.join(target_folder, file.split('.')[-1].upper())
            if not os.path.exists(target_folder):
                os.makedirs(target_folder)

            if any(file.lower().endswith(ext) for ext in file_extensions):
                source_file_path = os.path.join(root, file)
                target_file_path = os.path.join(target_folder, file)
                
                shutil.copy(source_file_path, target_file_path)
                print(f"Copied: {source_file_path} to {target_file_path}")

copy_files(source_black_path, target_black_path, file_extensions)
copy_files(source_white_path, target_white_path, file_extensions)
