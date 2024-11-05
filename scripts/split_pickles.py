import pickle
import os
import sys
import argparse
from math import ceil

def split_pickle(input_file, num_parts):
    # Create output directory
    output_dir = os.path.splitext(input_file)[0]
    os.makedirs(output_dir, exist_ok=True)

    # Load the pickle file
    with open(input_file, 'rb') as f:
        data = pickle.load(f)

    # Determine the type of data and split accordingly
    if isinstance(data, list):
        chunk_size = ceil(len(data) / num_parts)
        for i in range(num_parts):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(data))
            chunk = data[start:end]
            
            output_file = os.path.join(output_dir, f'part_{i}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(chunk, f)
    elif isinstance(data, dict):
        keys = list(data.keys())
        chunk_size = ceil(len(keys) / num_parts)
        for i in range(num_parts):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, len(keys))
            chunk = {k: data[k] for k in keys[start:end]}
            
            output_file = os.path.join(output_dir, f'part_{i}.pkl')
            with open(output_file, 'wb') as f:
                pickle.dump(chunk, f)
    else:
        raise ValueError("Unsupported data type in pickle file. Expected list or dict.")

    print(f"Split {input_file} into {num_parts} parts in the '{output_dir}' directory.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split a pickle file into N parts.")
    parser.add_argument("input_file", help="Path to the input pickle file")
    parser.add_argument("num_parts", type=int, help="Number of parts to split the file into")
    args = parser.parse_args()

    split_pickle(args.input_file, args.num_parts)
