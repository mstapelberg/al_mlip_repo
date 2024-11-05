import os
import shutil
import math
import argparse
from collections import defaultdict

def split_folder(source_dir, num_splits, destination_base):
    # Collect all files from source directory
    all_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            all_files.append(os.path.join(root, file))
    
    # Calculate files per split
    files_per_split = math.ceil(len(all_files) / num_splits)
    
    # Create destination directories
    dest_dirs = [os.path.join(destination_base, f'split_{i+1}') for i in range(num_splits)]
    for dir in dest_dirs:
        os.makedirs(dir, exist_ok=True)
    
    # Distribute files
    for i, file in enumerate(all_files):
        dest_index = i // files_per_split
        if dest_index >= num_splits:
            dest_index = num_splits - 1
        
        rel_path = os.path.relpath(file, source_dir)
        dest_file = os.path.join(dest_dirs[dest_index], rel_path)
        
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(file, dest_file)
    
    # Count files in each split
    file_counts = defaultdict(int)
    for dir in dest_dirs:
        for root, _, files in os.walk(dir):
            file_counts[dir] += len(files)
    
    print("File distribution:")
    for dir, count in file_counts.items():
        print(f"{dir}: {count} files")

def main():
    parser = argparse.ArgumentParser(description="Split a folder into N evenly distributed directories.")
    parser.add_argument("source_dir", help="Path to the source directory")
    parser.add_argument("num_splits", type=int, help="Number of splits to create")
    parser.add_argument("destination_base", help="Base path for destination directories")
    
    args = parser.parse_args()
    
    split_folder(args.source_dir, args.num_splits, args.destination_base)

if __name__ == "__main__":
    main()
