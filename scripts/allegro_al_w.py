import numpy as np
from ase import Atoms
from ase.io import write
from nequip.ase import NequIPCalculator

def perturb_atoms(atoms, temperature, scale=0.1):
    """
    Apply random perturbations to atom positions based on temperature.
    
    :param atoms: ASE Atoms object
    :param temperature: Temperature in Kelvin
    :param scale: Scaling factor for perturbation magnitude
    :return: New Atoms object with perturbed positions
    """
    kB = 8.617333262e-5  # Boltzmann constant in eV/K
    positions = atoms.get_positions()
    masses = atoms.get_masses()
    
    thermal_energy = kB * temperature
    perturbation_magnitudes = np.sqrt(thermal_energy / masses) * scale
    perturbations = np.random.randn(*positions.shape) * perturbation_magnitudes[:, np.newaxis]
    
    perturbed_atoms = atoms.copy()
    perturbed_atoms.set_positions(positions + perturbations)
    
    return perturbed_atoms

def calc_mean_squared_displacement(atoms1, atoms2):
    """
    Calculate the displacement between two sets of atoms.
    
    :param atoms1: First ASE Atoms object
    :param atoms2: Second ASE Atoms object
    :return: Displacement vector
    """
    positions1 = atoms1.get_positions()
    positions2 = atoms2.get_positions()
    
    displacement = positions2 - positions1
    return np.mean(np.sum(displacement ** 2, axis=1))

def calculate_normalized_force_variance(forces):
    # forces shape: (5, N, 3)
    # Normalize the force vectors
    force_magnitudes = np.linalg.norm(forces, axis=2, keepdims=True)
    normalized_forces = np.where(force_magnitudes != 0, forces / force_magnitudes, 0)
    
    # Calculate the variance across models for each atom
    atom_variances = np.var(normalized_forces, axis=0)
    
    # Sum the variances across x, y, z components
    total_atom_variances = np.sum(atom_variances, axis=1)
    
    return total_atom_variances

def calculate_structure_disagreement(forces, atom_disagreement):
    # forces shape: (5, N, 3)
    # atom_disagreement shape: (N,)
    
    force_magnitudes = np.mean(np.linalg.norm(forces, axis=2), axis=0)
    
    mean_disagreement = np.mean(atom_disagreement)
    max_disagreement = np.max(atom_disagreement)
    weighted_disagreement = np.sum(atom_disagreement * force_magnitudes) / np.sum(force_magnitudes)
    
    return {
        "mean_disagreement": mean_disagreement,
        "max_disagreement": max_disagreement,
        "weighted_disagreement": weighted_disagreement
    }

def iterative_force_variance_maximization(atoms, calculator_ensemble, max_steps=10, temperature=300, scale=0.1, disagreement_type='weighted_disagreement'):
    """
    Iteratively perturb atoms to maximize force variance.
    
    :param atoms: Initial ASE Atoms object
    :param calculator_ensemble: A callable that takes an Atoms object and returns a list of force predictions
    :param max_steps: Maximum number of perturbation steps
    :param temperature: Temperature for perturbation in Kelvin
    :param scale: Scaling factor for perturbation magnitude
    :param disagreement_type: Type of disagreement to maximize ('mean_disagreement', 'max_disagreement', 'weighted_disagreement')
    :return: List of Atoms objects with increasing force variance
    """
    selected_atoms = []
    current_atoms = atoms.copy()
    
    for step in range(max_steps):
        # Calculate current force structure disagreement
        current_forces_comm = calculator_ensemble(current_atoms)
        current_force_disagreement = calculate_normalized_force_variance(current_forces_comm)
        current_structure_disagreement = calculate_structure_disagreement(current_forces_comm, current_force_disagreement)
        
        # Perturb atoms
        perturbed_atoms = perturb_atoms(current_atoms, temperature, scale)
        
        # Calculate force structure disagreement for perturbed atoms
        perturbed_forces_comm = calculator_ensemble(perturbed_atoms)
        perturbed_force_disagreement = calculate_normalized_force_variance(perturbed_forces_comm)
        perturbed_structure_disagreement = calculate_structure_disagreement(perturbed_forces_comm, perturbed_force_disagreement)
        
        # Compare disagreements
        if perturbed_structure_disagreement[disagreement_type] > current_structure_disagreement[disagreement_type]:
            selected_atoms.append(perturbed_atoms)
            print(f"Step {step + 1}: Force variance increased. New disagreement: {perturbed_structure_disagreement[disagreement_type]:.6f}")
            print(f"Mean squared displacement: {calc_mean_squared_displacement(current_atoms, perturbed_atoms):.6f}")
            current_atoms = perturbed_atoms  # Use this as the new starting point
        else:
            print(f"Step {step + 1}: No improvement. Current disagreement: {current_structure_disagreement[disagreement_type]:.6f}")
    
    return selected_atoms

def mace_ensemble_calculator(mace_calc):
    """
    Create a calculator ensemble function for MACE.
    
    :param mace_calc: MACE calculator
    :return: Function that takes an Atoms object and returns a list of force predictions
    """
    def calculator(atoms):
        atoms.calc = mace_calc
        atoms.get_potential_energy()  # Initialize the calculator
        return atoms.calc.results['forces_comm']
    return calculator

def allegro_ensemble_calculator(allegro_calcs):
    """
    Create a calculator ensemble function for Allegro.
    
    :param allegro_calcs: List of Allegro calculators
    :return: Function that takes an Atoms object and returns a list of force predictions
    """
    def calculator(atoms):
        forces_comm = []
        for i,calc in enumerate(allegro_calcs):
            atoms_copy = atoms.copy()
            atoms_copy.calc = calc
            forces = atoms_copy.get_forces()
            #print(f"Allegro forces for model {i}: {forces}")
            forces_comm.append(forces)
        return np.array(forces_comm)
    return calculator

def process_atoms_list(atoms_list, calculator_ensemble, max_steps=10, temperature=300, scale=0.1):
    """
    Process a list of atoms, performing iterative force variance maximization on each.
    
    :param atoms_list: List of initial ASE Atoms objects
    :param calculator_ensemble: A callable that takes an Atoms object and returns a list of force predictions
    :param max_steps: Maximum number of perturbation steps for each atom
    :param temperature: Temperature for perturbation in Kelvin
    :param scale: Scaling factor for perturbation magnitude
    :return: List of all selected Atoms objects with increased force variance
    """
    all_selected_atoms = []
    
    for i, atoms in enumerate(atoms_list):
        print(f"\nProcessing atom {i + 1}/{len(atoms_list)}")
        selected = iterative_force_variance_maximization(atoms, calculator_ensemble, max_steps, temperature, scale)
        all_selected_atoms.extend(selected)
        print(f"Selected {len(selected)} configurations for atom {i + 1}")
    
    return all_selected_atoms


def allegro_ensemble(model_paths, device='cpu', species=None, default_dtype='float32'):
    """
    Create an ensemble of ALLEGRO models.
    
    :param model_paths: List of paths to ALLEGRO model files
    :param device: Device to use for calculations ('cpu' or 'cuda')
    :param species: List of species for the ensemble
    :param default_dtype: Default data type for calculations ('float32' or 'float64')
    :return: MACECalculator object
    """
    calcs = []
    for model_path in model_paths:
        calc = NequIPCalculator.from_deployed_model(model_path=model_path, species_to_type_name = species, device=device)
        calcs.append(calc)
    return calcs



# load the data from the db file 
from ase.io import read

atoms = read('data/w/w-14.xyz', index=':')

# create the initial atoms object
initial_atoms = [atom.copy() for atom in atoms if np.mean(atom.arrays['force']) != 0]

# define the model paths
model_paths = ['models/w/gen_0/gen_0_model_0.pth', 'models/w/gen_0/gen_0_model_1.pth', 'models/w/gen_0/gen_0_model_2.pth', 'models/w/gen_0/gen_0_model_3.pth', 'models/w/gen_0/gen_0_model_4.pth']

# create the ensemble 
allegro_calcs = allegro_ensemble(model_paths, device='cpu', species={'W' : 'W'}, default_dtype='float32') 
calculator_ensemble = allegro_ensemble_calculator(allegro_calcs)
#iterative_force_variance_maximization(atoms, calculator_ensemble, max_steps=10, temperature=1200, scale=0.1)
process_atoms_list(initial_atoms, calculator_ensemble, max_steps=10, temperature=1200, scale=1)