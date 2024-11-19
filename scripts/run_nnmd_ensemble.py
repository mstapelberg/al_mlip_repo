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
import os
import sys
import json
import random
import numpy as np
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # For headless environments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from dataclasses import dataclass
import pandas as pd
from collections import defaultdict
from ase.io import read, write
from ase import Atoms, units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import (
    Stationary,
    ZeroRotation,
    MaxwellBoltzmannDistribution
)
from mace.calculators import MACECalculator

import sys
sys.path.append('../Modules')
from al_functions import AdversarialCalculator, DisplacementGenerator, AdversarialOptimizer

@dataclass
class CompositionStats:
    mean_variance: float
    median_variance: float
    q95_variance: float
    max_variance: float
    min_variance: float
    clusters_stats: Dict[int, Dict[str, float]]

class CompositionVisualizer:
    def __init__(self, output_dir: str):
        """Initialize visualizer with output directory."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def plot_tsne_clusters(self, 
                          embedding: np.ndarray, 
                          clusters: np.ndarray,
                          compositions: np.ndarray,
                          selected_points: Optional[np.ndarray] = None):
        """
        Plot t-SNE embedding with cluster assignments and compositions.
        
        Args:
            embedding: t-SNE embedding array (n_samples, 2)
            clusters: Cluster assignments
            compositions: Original composition array
            selected_points: Indices of selected compositions
        """
        plt.figure(figsize=(15, 10))
        
        # Create main scatter plot
        scatter = plt.scatter(embedding[:, 0], embedding[:, 1], 
                            c=clusters, cmap='viridis',
                            alpha=0.6)
        
        # Add colorbar
        plt.colorbar(scatter, label='Cluster')
        
        # Highlight selected points if provided
        if selected_points is not None:
            plt.scatter(embedding[selected_points, 0], 
                       embedding[selected_points, 1],
                       color='red', marker='*', s=200,
                       label='Selected Compositions')
        
        plt.title('t-SNE Visualization of Composition Space')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.legend()
        
        # Save plot
        plt.savefig(self.output_dir / 'tsne_clusters.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create composition heatmap
        self._plot_composition_heatmap(compositions, clusters)
        
    def _plot_composition_heatmap(self, compositions: np.ndarray, clusters: np.ndarray):
        """Create heatmap of average compositions per cluster."""
        cluster_compositions = defaultdict(list)
        for comp, cluster in zip(compositions, clusters):
            cluster_compositions[cluster].append(comp)
            
        # Calculate mean compositions per cluster
        mean_compositions = {
            cluster: np.mean(comps, axis=0) 
            for cluster, comps in cluster_compositions.items()
        }
        
        # Create DataFrame for heatmap
        df_data = pd.DataFrame(
            mean_compositions.values(),
            index=mean_compositions.keys(),
            columns=['V', 'Cr', 'Ti', 'W', 'Zr']
        )
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_data, annot=True, fmt='.3f', cmap='YlOrRd')
        plt.title('Average Compositions per Cluster')
        plt.xlabel('Elements')
        plt.ylabel('Cluster')
        
        # Save plot
        plt.savefig(self.output_dir / 'cluster_compositions.png', dpi=300, bbox_inches='tight')
        plt.close()

class CompositionDatabase:
    def __init__(self, database_file: Optional[str] = None):
        """
        Initialize database for storing composition-variance data.
        
        Args:
            database_file: Path to existing database file (optional)
        """
        self.database_file = database_file
        self.data = []
        
        if database_file and Path(database_file).exists():
            self.load_database()
            
    def add_entry(self, composition: Dict[str, float], variance: float, 
                 atoms_info: Optional[Dict] = None):
        """Add new composition entry with variance."""
        entry = {
            'composition': composition,
            'variance': variance,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        if atoms_info:
            entry.update(atoms_info)
            
        self.data.append(entry)
        self._save_database()
        
    def load_database(self):
        """Load existing database."""
        with open(self.database_file, 'r') as f:
            self.data = json.load(f)
            
    def _save_database(self):
        """Save database to file in chunks if necessary."""
        if self.database_file:
            # Convert to list of dictionaries for JSON serialization
            data_to_save = {'compositions': self.data}
            with open(self.database_file, 'w') as f:
                json.dump(data_to_save, f)

class CompositionExplorer:
    def __init__(self, 
                 model_paths: List[str], 
                 database_file: str,
                 output_dir: str,
                 device: str = 'cpu',
                 grid_size: int = 125,
                 n_clusters: int = 10):
        self.calculator = AdversarialCalculator(
            model_paths=model_paths,
            device=device
        )
        
        self.database = CompositionDatabase(database_file)
        self.grid_size = grid_size
        self.n_clusters = n_clusters
        self.visualizer = CompositionVisualizer(output_dir)
        
        self.constraints = {
            'V': (0.696, 1.0),
            'Cr': (0.008, 0.304),
            'Ti': (0.008, 0.304),
            'W': (0.008, 0.304),
            'Zr': (0.008, 0.304)
        }
    
    def _generate_grid_compositions(self) -> np.ndarray:
        """Generate discretized composition grid."""
        step = 1.0 / self.grid_size
        
        v_range = np.arange(self.constraints['V'][0], 
                           self.constraints['V'][1] + step, 
                           step)
        
        compositions = []
        for v in v_range:
            remaining = 1.0 - v
            base_step = step * (remaining / 4)  # 4 other elements
            
            for cr in np.arange(self.constraints['Cr'][0], 
                              min(self.constraints['Cr'][1], remaining), 
                              base_step):
                for ti in np.arange(self.constraints['Ti'][0], 
                                  min(self.constraints['Ti'][1], remaining - cr), 
                                  base_step):
                    for w in np.arange(self.constraints['W'][0], 
                                     min(self.constraints['W'][1], remaining - cr - ti), 
                                     base_step):
                        zr = remaining - cr - ti - w
                        if self.constraints['Zr'][0] <= zr <= self.constraints['Zr'][1]:
                            compositions.append([v, cr, ti, w, zr])
        
        return np.array(compositions)
    
    def process_predefined_compositions(self, compositions: List[str]) -> List[Dict[str, float]]:
        """Process predefined compositions from chemical formulas."""
        processed_comps = []
        elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
        
        for comp_str in compositions:
            comp_dict = defaultdict(float)
            current_element = ''
            current_number = ''
            
            for char in comp_str:
                if char.isalpha():
                    if current_element and current_number:
                        comp_dict[current_element] = float(current_number)
                    current_element = char
                    current_number = ''
                else:
                    current_number += char
            
            if current_element and current_number:
                comp_dict[current_element] = float(current_number)
            
            total = sum(comp_dict.values())
            comp_dict = {k: v/total for k, v in comp_dict.items()}
            
            for elem in elements:
                if elem not in comp_dict:
                    comp_dict[elem] = 0.0
            
            processed_comps.append(dict(comp_dict))
        
        return processed_comps

    def analyze_composition_space(self, 
                                template_atoms: Atoms,
                                predefined_compositions: Optional[List[str]] = None
                                ) -> Tuple[List[Dict[str, float]], np.ndarray, np.ndarray]:
        """Analyze composition space with optional predefined compositions."""
        grid_comps = self._generate_grid_compositions()
        
        if predefined_compositions:
            pred_comps = self.process_predefined_compositions(predefined_compositions)
            pred_array = np.array([[comp[el] for el in ['V', 'Cr', 'Ti', 'W', 'Zr']]
                                 for comp in pred_comps])
            grid_comps = np.vstack([grid_comps, pred_array])
        
        tsne = TSNE(n_components=2, random_state=42)
        embedding = tsne.fit_transform(grid_comps)
        
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(embedding)
        
        # Select compositions from each cluster
        selected_indices = []
        for i in range(self.n_clusters):
            cluster_mask = clusters == i
            cluster_points = grid_comps[cluster_mask]
            center_idx = np.where(cluster_mask)[0][0]  # Take first point in cluster
            selected_indices.append(center_idx)
        
        self.visualizer.plot_tsne_clusters(embedding, clusters, grid_comps, selected_indices)
        
        selected_compositions = [{
            'V': grid_comps[idx][0],
            'Cr': grid_comps[idx][1],
            'Ti': grid_comps[idx][2],
            'W': grid_comps[idx][3],
            'Zr': grid_comps[idx][4]
        } for idx in selected_indices]
        
        return selected_compositions, embedding, clusters

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
    if len(sys.argv) != 2:
        print("Usage: python run_md_gen.py <input_xyz_file>")
        sys.exit(1)

    input_xyz = sys.argv[1]
    
    # Set random seed
    random.seed(42)
    np.random.seed(42)

    # Define paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(base_dir, '../Models/zr-w-v-ti-cr/gen_4_2024-11-15')
    output_dir = os.path.join(base_dir, '../data/zr-w-v-ti-cr/gen_5_2024-11-17/md_frames')
    database_file = os.path.join(base_dir, '../data/composition_database.json')
    vis_dir = os.path.join(output_dir, 'visualization')

    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    # Load atomic structures
    atoms_list = read(input_xyz, index=':')
    print(f"Loaded {len(atoms_list)} structures from {input_xyz}")

    # gen_4_model_0-11-14_b4_stagetwo_compiled.model
    # Setup MACE models
    model_paths = [
        f'{model_dir}/gen_4_model_{i}-11-14_b4_stagetwo_compiled.model'
        for i in range(3)
    ]
    
    # Get predefined compositions from input structures
    predefined_comps = [
        str(atoms.get_chemical_formula()).replace(" ", "")
        for atoms in atoms_list
    ]
    
    # Initialize explorer
    explorer = CompositionExplorer(
        model_paths=model_paths,
        database_file=database_file,
        output_dir=vis_dir,
        device='cuda',
        grid_size=125,
        n_clusters=10
    )
    
    # Run composition space analysis
    selected_compositions, _, _ = explorer.analyze_composition_space(
        template_atoms=atoms_list[0],
        predefined_compositions=predefined_comps
    )
    
    # Run MD simulations
    temperatures = [1000, 2000, 3000]
    for comp_idx, composition in enumerate(selected_compositions):
        # Create atoms object with this composition
        atoms = modify_atoms_composition(atoms_list[0], composition)
        
        for t in temperatures:
            # Create descriptive filename
            comp_str = '_'.join([f"{k}{v:.3f}" for k, v in composition.items()])
            filename = os.path.join(output_dir, f'comp_{comp_idx}_{comp_str}_T{t}_md.xyz')
            
            if not os.path.exists(filename):
                try:
                    print(f"Running MD for composition {comp_idx} at {t}K")
                    mace_calc = MACECalculator(
                        model_paths=model_paths,
                        device='cuda',
                        default_dtype='float32'
                    )
                    simpleMD(atoms, temp=t, calc=mace_calc,
                            fname=filename, s=100, T=1000)
                except Exception as e:
                    print(f"Error in MD for composition {comp_idx} at {t}K: {str(e)}")
            else:
                print(f"Skipping {filename} - file already exists")

if __name__ == "__main__":
    main()
