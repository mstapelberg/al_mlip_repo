#!/usr/bin/env python3

import sys
from ase.io import read, write
import numpy as np

def split_xyz(input_file, n_parts):
    # Read all structures
    atoms_list = read(input_file, index=':')
    n_structs = len(atoms_list)
    
    # Calculate structures per part
    structs_per_part = int(np.ceil(n_structs / n_parts))
    
    # Split and write
    for i in range(n_parts):
        start_idx = i * structs_per_part
        end_idx = min((i + 1) * structs_per_part, n_structs)
        
        if start_idx >= n_structs:
            break
            
        output_file = f"{input_file.rsplit('.', 1)[0]}_part{i}.xyz"
        write(output_file, atoms_list[start_idx:end_idx])
        print(f"Written {end_idx - start_idx} structures to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python split_xyz.py <input_xyz> <n_parts>")
        sys.exit(1)
        
    input_file = sys.argv[1]
    n_parts = int(sys.argv[2])
    split_xyz(input_file, n_parts)
