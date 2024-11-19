import json
import os
import sys 
from ase.io import read, write 
from typing import Dict 
import glob

def analyze_xyz_file(xyz_path: str) -> Dict[str, float]:
    """
    Analyze single XYZ file and extract compositions.
    Returns: Dict mapping composition string to frequency
    """
    atoms_list = read(xyz_path, ':')
    compositions = {}
    for atoms in atoms_list:
        comp = str(atoms.get_chemical_formula()).replace(" ", "")
        compositions[comp] = compositions.get(comp, 0) + 1
    return compositions

def main():
    input_dir = sys.argv[1]  # Directory containing split XYZ files
    output_json = sys.argv[2] # Output JSON for composition analysis
    
    # Modified glob pattern to match only files containing "_part" in their name
    xyz_files = glob.glob(os.path.join(input_dir, "*_part*.xyz"))
    
    if not xyz_files:
        print(f"Warning: No part XYZ files found in {input_dir}")
        print(f"Contents of directory:")
        os.system(f"ls -l {input_dir}")
        return
        
    print(f"Found {len(xyz_files)} part XYZ files in {input_dir}:")
    for f in xyz_files:
        print(f"  - {os.path.basename(f)}")
    
    all_compositions = {}
    
    # Process each XYZ file
    for xyz_file in xyz_files:
        try:
            comps = analyze_xyz_file(xyz_file)
            all_compositions.update(comps)
            print(f"Successfully processed: {os.path.basename(xyz_file)}")
        except Exception as e:
            print(f"Error processing {xyz_file}: {str(e)}")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Save results
    with open(output_json, 'w') as f:
        json.dump(all_compositions, f, indent=2)
    print(f"\nResults saved to: {output_json}")

if __name__ == "__main__":
    main()