#!/usr/bin/env python3
import sys 
from typing import Dict, List
sys.path.append('../../Modules')
from al_functions import AdversarialCalculator
from ase.io import read
from ase import Atoms 
import os
import sys
import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path
from ase import Atoms
from ase.io import read
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
from monty.json import MontyEncoder, MontyDecoder
from monty.serialization import loadfn, dumpfn

class CompositionDatabase:
    def __init__(self, database_file: Optional[str] = None):
        self.database_file = database_file
        self.data = []
        
        if database_file:
            if Path(database_file).exists() and Path(database_file).stat().st_size > 0:
                self.load_database()
            else:
                self._save_database()
    
    def composition_exists(self, composition: Dict[str, float], tolerance: float = 1e-3) -> bool:
        """Check if composition already exists in database."""
        for entry in self.data:
            existing_comp = entry['composition']
            if all(abs(existing_comp.get(elem, 0) - composition.get(elem, 0)) < tolerance 
                   for elem in set(existing_comp) | set(composition)):
                return True
        return False
        
    def add_entry(self, composition: Dict[str, float], variance_metrics: Dict, 
                 atoms_info: Optional[Dict] = None):
        entry = {
            'composition': composition,
            'variance_metrics': variance_metrics,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        if atoms_info:
            entry.update(atoms_info)
            
        self.data.append(entry)
        self._save_database()
        
    def load_database(self):
        try:
            self.data = loadfn(self.database_file).get('compositions', [])
        except (json.JSONDecodeError, FileNotFoundError):
            self.data = []
            
    def _save_database(self):
        if self.database_file:
            data_to_save = {'compositions': self.data}
            dumpfn(data_to_save, self.database_file)

def calculate_force_variances(compositions: List[str], 
                            template_atoms: Atoms,
                            model_paths: List[str],
                            database_file: str,
                            device: str = 'cuda') -> Dict[str, Dict]:
    """Calculate force variance for compositions not in database."""
    calculator = AdversarialCalculator(model_paths=model_paths, device=device)
    database = CompositionDatabase(database_file)
    variances = {}
    
    print(f"Processing {len(compositions)} compositions...")
    
    for comp_str in tqdm(compositions):
        try:
            comp_dict = process_composition_string(comp_str)
            
            if not database.composition_exists(comp_dict):
                atoms = modify_atoms_composition(template_atoms, comp_dict)
                forces = calculator.calculate_forces(atoms)
                
                atom_disagreement = calculator.calculate_normalized_force_variance(forces)
                structure_metrics = calculator.calculate_structure_disagreement(forces, atom_disagreement)
                
                # Add per-atom variances
                structure_metrics['per_atom_variances'] = atom_disagreement.tolist()
                
                # Save to database
                database.add_entry(
                    composition=comp_dict,
                    variance_metrics=structure_metrics,
                    atoms_info={'formula': comp_str}
                )
                
                variances[comp_str] = structure_metrics
                
                print(f"\nProcessed {comp_str}:")
                print(f"  Mean disagreement: {structure_metrics['mean_disagreement']:.6f}")
                print(f"  Max disagreement: {structure_metrics['max_disagreement']:.6f}")
                print(f"  Weighted disagreement: {structure_metrics['weighted_disagreement']:.6f}")
                
        except Exception as e:
            print(f"\nError processing {comp_str}: {str(e)}")
            continue
    
    return variances

def process_composition_string(comp_str: str) -> Dict[str, float]:
    """Convert composition string to dictionary of fractions."""
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    comp_dict = defaultdict(float)
    current_element = ''
    current_number = ''
    
    i = 0
    while i < len(comp_str):
        if comp_str[i].isalpha():
            # If we have a completed element and number, add it
            if current_element and current_number:
                comp_dict[current_element] = float(current_number)
                current_number = ''
            
            # Start new element
            if comp_str[i].isupper():
                current_element = comp_str[i]
                # Check for second letter
                if i + 1 < len(comp_str) and comp_str[i + 1].islower():
                    current_element += comp_str[i + 1]
                    i += 1
            i += 1
        else:
            current_number += comp_str[i]
            i += 1
    
    # Add the last element-number pair
    if current_element and current_number:
        comp_dict[current_element] = float(current_number)
    
    # Convert counts to fractions
    total = sum(comp_dict.values())
    comp_dict = {k: v/total for k, v in comp_dict.items()}
    
    # Ensure all elements are present
    return {elem: comp_dict.get(elem, 0.0) for elem in elements}

def modify_atoms_composition(atoms: Atoms, target_comp: Dict[str, float]) -> Atoms:
    """Modify atoms object to match target composition."""
    symbols = atoms.get_chemical_symbols()
    positions = atoms.get_positions()
    n_atoms = len(atoms)
    
    target_counts = {elem: int(round(frac * n_atoms)) 
                    for elem, frac in target_comp.items()}
    
    total_target = sum(target_counts.values())
    if total_target != n_atoms:
        max_elem = max(target_counts.items(), key=lambda x: x[1])[0]
        adjustment = n_atoms - total_target
        target_counts[max_elem] += adjustment
    
    new_symbols = []
    for elem, count in target_counts.items():
        new_symbols.extend([elem] * count)
    
    np.random.shuffle(new_symbols)
    
    return Atoms(symbols=new_symbols,
                positions=positions,
                cell=atoms.cell,
                pbc=atoms.pbc)


def main():
    if len(sys.argv) != 4:
        print("Usage: python calculate_variance.py <compositions_json> <template_xyz> <output_json>")
        sys.exit(1)
        
    compositions_file = sys.argv[1]
    template_file = sys.argv[2]
    output_file = sys.argv[3]
    
    # Load compositions using monty
    compositions = loadfn(compositions_file)
    
    # Load template structure
    template_atoms = read(template_file)
    print(f"Loaded template structure with {len(template_atoms)} atoms")
    
    # Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '../../Models/zr-w-v-ti-cr/gen_4_2024-11-15')
    database_file = os.path.join(base_dir, '../../data/composition_database.json')
    
    # Setup model paths
    model_paths = [
        f'{model_dir}/gen_4_model_{i}-11-14_b4_stagetwo_compiled.model'
        for i in range(3)
    ]
    
    # Calculate variances
    variances = calculate_force_variances(
        compositions=list(compositions.keys()),
        template_atoms=template_atoms,
        model_paths=model_paths,
        database_file=database_file,
        device='cuda'
    )
    
    print(variances)

    # Save results using monty
    dumpfn(variances, output_file)

if __name__ == "__main__":
    main()