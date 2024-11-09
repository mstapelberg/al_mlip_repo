#!/usr/bin/env python3
import os
import sys
import json
import random
import numpy as np
from ase.io import read, write
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    Stationary, 
    ZeroRotation, 
    MaxwellBoltzmannDistribution
)
from mace.calculators import MACECalculator

# Import all your functions here
from ase.io import read, write
import numpy as np
from ase import Atoms
import json
import random
from typing import List, Dict
from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
#from aseMolec import extAtoms as ea
import matplotlib
import os
import time
import numpy as np
import pylab as pl
from IPython import display

def load_historical_compositions(json_file: str) -> Dict[int, List[Dict[str, float]]]:
    """Load compositions from generations 0-3 from JSON file"""
    with open(json_file, 'r') as f:
        all_comps = json.load(f)
    # Filter for generations 0-3
    gen_comps = {}
    for comp in all_comps:
        gen = comp['Generation']
        if gen <= 3:
            if gen not in gen_comps:
                gen_comps[gen] = []
            gen_comps[gen].append(comp)
    return gen_comps

def generate_nary_composition(n_elements: int, elements: List[str] = ['V', 'Cr', 'Ti', 'Zr', 'W']) -> Dict[str, float]:
    """Generate n-ary composition with V-rich preference"""
    # Sample V content from normal distribution
    v_content = np.random.normal(0.75, 0.125)
    v_content = np.clip(v_content, 0.0, 1.0)
    
    # Select n-1 other elements randomly (V is always included)
    other_elements = random.sample([e for e in elements if e != 'V'], n_elements - 1)
    
    # Distribute remaining fraction among other elements
    remaining = 1.0 - v_content
    other_fractions = np.random.dirichlet(np.ones(n_elements - 1)) * remaining
    
    # Create composition dictionary
    composition = {'V': v_content}
    for elem, frac in zip(other_elements, other_fractions):
        composition[elem] = frac
    
    # Fill remaining elements with 0
    for elem in elements:
        if elem not in composition:
            composition[elem] = 0.0
            
    return composition

def generate_equimolar_composition(elements: List[str] = ['V', 'Cr', 'Ti', 'Zr', 'W']) -> Dict[str, float]:
    """Generate equimolar composition"""
    fraction = 1.0 / len(elements)
    return {elem: fraction for elem in elements}

def generate_target_composition(historical_comps: Dict[int, List[Dict[str, float]]]) -> Dict[str, float]:
    """Generate a target composition based on the defined probabilities"""
    rand = random.random()
    
    if rand < 0.70:  # Historical composition
        # Randomly select generation and composition
        gen = random.choice(list(historical_comps.keys()))
        comp = random.choice(historical_comps[gen])
        return {k: v for k, v in comp.items() if k != 'Generation'}
    
    elif rand < 0.95:  # N-ary composition
        n_elements = random.randint(1, 4)  # Unary to quaternary
        return generate_nary_composition(n_elements)
    
    else:  # Equimolar
        return generate_equimolar_composition()

def modify_atoms_composition(atoms: Atoms, target_comp: Dict[str, float]) -> Atoms:
    """Modify atoms object to match target composition as closely as possible"""
    # Get initial composition
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    
    # Count total atoms
    n_atoms = len(atoms)
    print(f"Initial composition: {dict(zip(*np.unique(symbols, return_counts=True)))}")
    print(f"Target composition: {target_comp}")
    
    # Calculate target counts
    target_counts = {elem: int(round(frac * n_atoms)) 
                    for elem, frac in target_comp.items()}
    
    # Ensure total matches original atom count
    total_target = sum(target_counts.values())
    if total_target != n_atoms:
        # Find element with highest count to adjust
        max_elem = max(target_counts.items(), key=lambda x: x[1])[0]
        adjustment = n_atoms - total_target
        target_counts[max_elem] += adjustment
        print(f"Adjusted {max_elem} count by {adjustment}")
        print(f"New target counts: {target_counts}")
    
    # Create new symbol list
    new_symbols = []
    for elem, count in target_counts.items():
        new_symbols.extend([elem] * count)
    
    # Randomly shuffle the new symbols
    np.random.shuffle(new_symbols)
    
    print(f"Length of new_symbols: {len(new_symbols)}")
    print(f"Length of filtered_positions: {len(positions)}")
    print(f"New composition: {dict(zip(*np.unique(new_symbols, return_counts=True)))}")
    
    # Create new atoms object with modified composition
    new_atoms = Atoms(symbols=new_symbols,
                     positions=positions,
                     cell=atoms.cell,
                     pbc=atoms.pbc)
    
    return new_atoms

def process_structures(atoms_list: List[Atoms], historical_comps: Dict[int, List[Dict[str, float]]]) -> List[Atoms]:
    """Process list of atoms objects and create modified versions"""
    new_structures = []
    
    for atoms in atoms_list:
        # Generate target composition
        target_comp = generate_target_composition(historical_comps)
        
        # Modify atoms to match target composition
        new_atoms = modify_atoms_composition(atoms, target_comp)
        
        # Add to list of new structures
        new_structures.append(new_atoms)
    
    return new_structures

def _simpleMD(init_conf, temp, calc, fname, s, T):
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

    Returns:
    None
    """
    
    # Set the calculator
    init_conf.calc = calc

    #initialize the temperature

    MaxwellBoltzmannDistribution(init_conf, temperature_K=300) #initialize temperature at 300
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(init_conf, 1.0*units.fs, temperature_K=temp, friction=0.5) #drive system to desired temperature


    time_fs = []
    temperature = []
    energies = []

    #remove previously stored trajectory with the same name
    os.system('rm -rfv '+fname)

    fig, ax = pl.subplots(2, 1, figsize=(6,6), sharex='all', gridspec_kw={'hspace': 0, 'wspace': 0})

    def write_frame():
            dyn.atoms.info['energy_mace'] = dyn.atoms.get_potential_energy()
            dyn.atoms.arrays['force_mace'] = dyn.atoms.calc.get_forces()
            dyn.atoms.write(fname, append=True, write_results=False)
            time_fs.append(dyn.get_time()/units.fs)
            temperature.append(dyn.atoms.get_temperature())
            energies.append(dyn.atoms.get_potential_energy()/len(dyn.atoms))

            ax[0].plot(np.array(time_fs), np.array(energies), color="b")
            ax[0].set_ylabel('E (eV/atom)')

            # plot the temperature of the system as subplots
            ax[1].plot(np.array(time_fs), temperature, color="r")
            ax[1].set_ylabel('T (K)')
            ax[1].set_xlabel('Time (fs)')

            display.clear_output(wait=True)
            display.display(pl.gcf())
            time.sleep(0.01)

    dyn.attach(write_frame, interval=s)
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))


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

    Returns:
    None
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

def main():
    # Check command line arguments
    if len(sys.argv) != 2:
        print("Usage: python run_md_gen.py <input_xyz_file>")
        sys.exit(1)

    input_xyz = sys.argv[1]
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '../Models/zr-w-v-ti-cr/gen_0_2024-11-06')
    output_dir = os.path.join(base_dir, '../data/zr-w-v-ti-cr/gen_0_2024-11-06/md_frames')
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load atomic structures from input xyz
    atoms_list = read(input_xyz, index=':')
    print(f"Loaded {len(atoms_list)} structures from {input_xyz}")
    
    # Setup MACE models
    model_paths = [
        f'{model_dir}/gen_0_model_{i}-11-06-fixedtest_stagetwo_compiled.model'
        for i in range(5)
    ]
    
    # Run MD simulations
    temperatures = [2000, 3000]
    for t in temperatures:
        for i, atoms in enumerate(atoms_list):
            mace_calc = MACECalculator(
                model_paths=model_paths, 
                device='cuda', 
                default_dtype='float32'
            )
            
            comp = str(atoms.get_chemical_formula()).replace(' ', '')
            filename = os.path.join(output_dir, f'gen_0_idx-{i}_comp-{comp}_temp-{t}_md.xyz')
            
            if not os.path.exists(filename):
                try:
                    print(f"Running MD for structure {i} at {t}K")
                    simpleMD(atoms, temp=t, calc=mace_calc, 
                            fname=filename, s=100, T=1000)
                except Exception as e:
                    print(f"Error in MD for structure {i} at {t}K: {str(e)}")
            else:
                print(f"Skipping {filename} - file already exists")

if __name__ == "__main__":
    main()