import numpy as np
from ase import Atoms 
from ase.io import read
from nequip.ase import NequIPCalculator
from mace.ase import MACECalculator 

class AdversarialCalculator:
    def __init__(self, model_paths, calculator_type='mace', device='cpu', species=None, default_dtype='float32'):
        """
        Initialize the calculator with either single or ensemble of models.
        
        Args:
            model_paths (str or list): Path(s) to model file(s)
            calculator_type (str): Type of calculator ('mace' or 'allegro')
            device (str): Device to use ('cpu' or 'cuda')
            species (dict): Dictionary mapping species to type names (for Allegro)
            default_dtype (str): Default data type for calculations
        """
        self.calculator_type = calculator_type
        self.device = device
        self.default_dtype = default_dtype
        
        if isinstance(model_paths, str):
            self.is_ensemble = False
            self.model_paths = [model_paths]
        else:
            self.is_ensemble = True
            self.model_paths = model_paths
            
        self.calculators = self._initialize_calculators(species)
        
    def _initialize_calculators(self, species):
        """Initialize single calculator or ensemble of calculators."""
        if self.calculator_type == 'mace':
            return MACECalculator(model_paths=self.model_paths, 
                                device=self.device, 
                                default_dtype=self.default_dtype)
        elif self.calculator_type == 'allegro':
            return [NequIPCalculator.from_deployed_model(
                model_path=model_path,
                species_to_type_name=species,
                device=self.device
            ) for model_path in self.model_paths]
        else:
            raise ValueError(f"Unknown calculator type: {self.calculator_type}")
            
    def calculate_forces(self, atoms):
        """Calculate forces using single calculator or ensemble."""
        if self.calculator_type == 'mace':
            atoms.calc = self.calculators
            atoms.get_potential_energy()  # Initialize the calculator
            return atoms.calc.results['forces_comm']
        else:
            forces_comm = []
            for calc in self.calculators:
                atoms_copy = atoms.copy()
                atoms_copy.calc = calc
                forces = atoms_copy.get_forces()
                forces_comm.append(forces)
            return np.array(forces_comm)
            
    def calculate_normalized_force_variance(self, forces):
        """Calculate normalized force variance across ensemble predictions."""
        force_magnitudes = np.linalg.norm(forces, axis=2, keepdims=True)
        normalized_forces = np.where(force_magnitudes != 0, 
                                   forces / force_magnitudes, 
                                   0)
        atom_variances = np.var(normalized_forces, axis=0)
        total_atom_variances = np.sum(atom_variances, axis=1)
        return total_atom_variances
        
    def calculate_structure_disagreement(self, forces, atom_disagreement):
        """Calculate structure disagreement metrics."""
        force_magnitudes = np.mean(np.linalg.norm(forces, axis=2), axis=0)
        mean_disagreement = np.mean(atom_disagreement)
        max_disagreement = np.max(atom_disagreement)
        weighted_disagreement = np.sum(atom_disagreement * force_magnitudes) / np.sum(force_magnitudes)
        
        return {
            "mean_disagreement": mean_disagreement,
            "max_disagreement": max_disagreement,
            "weighted_disagreement": weighted_disagreement
        }
            
