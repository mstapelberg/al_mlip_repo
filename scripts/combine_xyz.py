import os
import glob
import sys

def combine_xyz_files(directory, output_file):
    # Get all XYZ files in the directory
    xyz_files = glob.glob(os.path.join(directory, '**', '*.xyz'), recursive=True)
    
    if not xyz_files:
        print(f"No XYZ files found in {directory}")
        return
    
    with open(output_file, 'w') as outfile:
        for i, xyz_file in enumerate(xyz_files):
            with open(xyz_file, 'r') as infile:
                # Copy the entire content of each file
                content = infile.read()
                outfile.write(content)
                
                # Add a newline if it's not the last file and doesn't end with a newline
                if i < len(xyz_files) - 1 and not content.endswith('\n'):
                    outfile.write('\n')
    
    print(f"Combined XYZ file created: {output_file}")

# Usage
#directory = '/path/to/your/directory'  # Replace with your directory path
directory = sys.argv[1]
output_file = sys.argv[2]
#output_file = 'combined_output.xyz'  # Name of the output file

combine_xyz_files(directory, output_file)
