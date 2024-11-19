#!/usr/bin/env python
# Import necessary libraries
from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    MaxwellBoltzmannDistribution
)
from mace.calculators import MACECalculator
import numpy as np
import os
import time
import argparse
from pathlib import Path
import glob
from typing import List

def simpleMD(init_conf, temp, calc, fname, s, T):
    """
    Perform a simple molecular dynamics (MD) simulation using Langevin dynamics.

    Parameters:
    init_conf : Atoms object
        Initial atomic configuration.
    temp : float
        Desired temperature for the simulation in Kelvin.
    calc : Calculator object
        Calculator to be used for energy and force calculations.
    fname : str
        Filename to store the trajectory.
    s : int
        Interval for writing frames to the trajectory file.
    T : int
        Total number of steps for the MD simulation.
    """
    # Set the calculator
    init_conf.calc = calc

    # Initialize the temperature
    MaxwellBoltzmannDistribution(init_conf, temperature_K=300)
    Stationary(init_conf)
    ZeroRotation(init_conf)

    # Setup Langevin dynamics
    dyn = Langevin(init_conf, 1.0*units.fs, temperature_K=temp, friction=0.5)

    # Remove previously stored trajectory with the same name
    os.system('rm -rfv '+fname)

    def write_frame():
        # Store energy and forces in atoms object
        dyn.atoms.info['energy_mace'] = dyn.atoms.get_potential_energy()
        dyn.atoms.arrays['force_mace'] = dyn.atoms.calc.get_forces()
        # Write frame to trajectory file
        dyn.atoms.write(fname, append=True, write_results=False)

    # Attach frame writing to dynamics
    dyn.attach(write_frame, interval=s)

    # Run MD
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print(f"MD finished in {(t1-t0)/60:.2f} minutes!")

def get_model_paths(model_dir: str) -> List[str]:
    """Get all model paths from directory."""
    pattern = f"{model_dir}/gen_4_model_*-11-14_b4_stagetwo_compiled.model"
    return sorted(glob.glob(pattern))

def main():
    parser = argparse.ArgumentParser(description='Run parallel MD simulations')
    parser.add_argument('--input', type=str, required=True,
                      help='Input structure file')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Output directory for MD trajectories')
    parser.add_argument('--model_dir', type=str, required=True,
                      help='Directory containing MACE models')
    parser.add_argument('--task_id', type=int, required=True,
                      help='Slurm task ID')
    
    args = parser.parse_args()
    
    # Get model paths
    model_paths = get_model_paths(args.model_dir)
    if not model_paths:
        raise ValueError(f"No model files found in {args.model_dir}")
    
    print(f"Task {args.task_id}: Found models: {model_paths}")
    
    # Create output directory if needed
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Read structure
    atoms = read(args.input)
    struct_name = Path(args.input).stem
    
    # Setup calculator
    mace_calc = MACECalculator(
        model_paths=model_paths,
        device='cuda',
        default_dtype='float32'
    )
    
    # Run MD at different temperatures
    temperatures = [1000, 2000, 3000]
    for temp in temperatures:
        filename = os.path.join(args.output_dir, f"{struct_name}_T{temp}_md.xyz")
        
        if os.path.exists(filename):
            print(f"Task {args.task_id}: Skipping existing file {filename}")
            continue
            
        try:
            print(f"Task {args.task_id}: Running MD at {temp}K for {struct_name}")
            simpleMD(atoms, temp=temp, calc=mace_calc,
                    fname=filename, s=100, T=1000)
                    
        except Exception as e:
            print(f"Task {args.task_id}: Error in MD for {struct_name} at {temp}K: {str(e)}")
            continue

if __name__ == "__main__":
    main()