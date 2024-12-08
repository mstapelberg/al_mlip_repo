import os
import shutil
from glob import glob
import re

def move_frame_folders(source_dir, target_dir, frame_numbers=[0, 4, 9, 14, 19, 24]):
    """
    Move folders for specific frame numbers to a new location.
    
    Args:
        source_dir (str): Directory containing the VASP calculation folders
        target_dir (str): Directory where selected folders will be moved
        frame_numbers (list): List of frame numbers to move
    """
    # Create target directory if it doesn't exist
    os.makedirs(target_dir, exist_ok=True)
    
    # Pattern to match directories like "structure_20_vac_adversarial_1563"
    pattern = r'structure_(\d+)_(.+)'
    
    # Get all directories in the source folder
    all_dirs = [d for d in glob(os.path.join(source_dir, '*')) if os.path.isdir(d)]
    
    moved_count = 0
    for dir_path in all_dirs:
        dir_name = os.path.basename(dir_path)
        match = re.match(pattern, dir_name)
        
        if match:
            frame_num = int(match.group(1))
            if frame_num in frame_numbers:
                # Create target path
                target_path = os.path.join(target_dir, dir_name)
                
                # Move the directory
                shutil.move(dir_path, target_path)
                print(f"Moved {dir_name} to {target_dir}")
                moved_count += 1
    
    print(f"\nMoved {moved_count} folders in total")

def main():
    # Define source and target directories
    source_dir = '../vasp_jobs/zr-w-v-ti-cr/job_gen_2-2024-11-10'  # Adjust this path
    target_dir = '../vasp_jobs/zr-w-v-ti-cr/job_gen_2-2024-11-10_selected_frames_t2'       # Adjust this path

    # Define which frames to move
    frames_to_move = [1, 5, 10, 15, 20, 23]
    
    # Move the folders
    move_frame_folders(source_dir, target_dir, frames_to_move)

if __name__ == "__main__":
    main()
