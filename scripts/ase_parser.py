import os
import argparse
from ase.io import read, write

def parse_arguments():
    parser = argparse.ArgumentParser(description='Process OUTCAR files and apply selection scheme.')
    parser.add_argument('root_dir', type=str, help='Root directory containing subdirectories with OUTCAR files')
    parser.add_argument('--selection_scheme', type=str, choices=['all','gnome'], default='all', help='Selection scheme to apply')
    parser.add_argument('--output_file', type=str, default='combined_selected_steps.extxyz', help='Output filename for the combined extxyz file')
    return parser.parse_args()

def find_outcar_files(root_dir):
    outcar_files = []
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.startswith('OUTCAR') or filename.endswith('_OUTCAR'):
                outcar_files.append(os.path.join(dirpath, filename))
    return outcar_files

def apply_selection_scheme(all_steps, scheme):
    selected_steps = []
    num_steps = len(all_steps)

    if scheme == 'all':
        selected_steps = all_steps  # Use all steps when the 'all' scheme is selected

    elif scheme == 'gnome':
        first_half = all_steps[:num_steps // 2]
        second_half = all_steps[num_steps // 2:]

        # Take every other step from the first half
        selected_steps.extend(first_half[::2])

        # Take a quarter of the second half
        selected_steps.extend(second_half[::4])

    return selected_steps

def process_outcar_files(outcar_files, output_file, selection_scheme):
    open(output_file, 'w').close()

    for outcar in outcar_files:
        try:
            all_steps = read(outcar,format='vasp-out', index=':')
        except Exception as e:
            print(f'Error occurred reading {outcar} : Error = {e}')
            continue

        selected_steps = apply_selection_scheme(all_steps, selection_scheme)

        for atoms in selected_steps:
            if 'energy' in atoms.info:
                atoms.info['REF_energy'] = atoms.info.pop('energy')
            if 'forces' in atoms.arrays:
                atoms.arrays['REF_force'] = atoms.arrays.pop('forces')
            if 'stress' in atoms.info:
                atoms.info['REF_stress'] = atoms.info.pop('stress')

        write(output_file, selected_steps, format='extxyz', append=True)

def main():
    args = parse_arguments()
    outcar_files = find_outcar_files(args.root_dir)
    process_outcar_files(outcar_files, args.output_file, args.selection_scheme)

if __name__ == '__main__':
    main()
