import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from collections import defaultdict
from monty.serialization import loadfn, dumpfn
from ase import Atoms
from ase.io import write, read
import random
import time
import pandas as pd

@dataclass
class CompositionAnalysisResult:
    """Container for analysis results"""
    selected_compositions: List[Dict[str, float]]
    embeddings: np.ndarray
    clusters: np.ndarray
    variance_scores: np.ndarray
    cluster_stats: Dict[int, Dict[str, float]]

def process_composition(formula: str) -> Dict[str, float]:
    """
    Convert a chemical formula string into a normalized composition dictionary.
    
    Args:
        formula: Chemical formula string (e.g., 'V50Cr20Ti10W10Zr10')
        
    Returns:
        Dict[str, float]: Normalized composition dictionary
    """
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    comp_dict = defaultdict(float)
    
    # Parse formula
    current_element = ''
    current_number = ''
    
    for char in formula:
        if char.isalpha():
            if current_element and current_number:
                comp_dict[current_element] = float(current_number)
            current_element = char
            if char not in elements:
                continue
            current_number = ''
        else:
            current_number += char
            
    if current_element and current_number:
        comp_dict[current_element] = float(current_number)
    
    # Ensure all elements are present
    for elem in elements:
        if elem not in comp_dict:
            comp_dict[elem] = 0.0
            
    # Normalize
    total = sum(comp_dict.values())
    if total > 0:
        for elem in elements:
            comp_dict[elem] /= total
            
    return comp_dict

def analyze_composition_space(
    compositions: List[Dict[str, float]],
    variances: np.ndarray,
    n_select: int = 150,
    random_state: int = 42
) -> CompositionAnalysisResult:
    """
    Analyze composition space using t-SNE and clustering.
    """
    # Convert compositions to array format
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    comp_array = np.array([[comp[elem] for elem in elements] for comp in compositions])
    
    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=random_state)
    embeddings = tsne.fit_transform(comp_array)
    
    # Perform clustering
    n_clusters = n_select // 2
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(comp_array)  # Cluster on original compositions
    
    # Calculate cluster statistics
    cluster_stats = {}
    selected_indices = []
    
    for cluster_idx in range(n_clusters):
        mask = clusters == cluster_idx
        cluster_comps = comp_array[mask]
        cluster_vars = variances[mask]
        
        # Calculate cluster statistics
        mean_comp = np.mean(cluster_comps, axis=0)
        mean_comp_dict = dict(zip(elements, mean_comp))
        
        cluster_stats[cluster_idx] = {
            'mean_variance': float(np.mean(cluster_vars)),
            'max_variance': float(np.max(cluster_vars)),
            'size': int(np.sum(mask)),
            'mean_composition': mean_comp_dict
        }
        
        # Select highest variance point from cluster
        if np.any(mask):
            cluster_max_var_idx = np.where(mask)[0][np.argmax(cluster_vars)]
            selected_indices.append(cluster_max_var_idx)
    
    # Create selected compositions list ensuring proper normalization
    selected_compositions = []
    for idx in selected_indices:
        comp = dict(zip(elements, comp_array[idx]))
        # Double-check normalization
        total = sum(comp.values())
        if total > 0:
            comp = {k: v/total for k, v in comp.items()}
        selected_compositions.append(comp)
    
    return CompositionAnalysisResult(
        selected_compositions=selected_compositions,
        embeddings=embeddings,
        clusters=clusters,
        variance_scores=variances,
        cluster_stats=cluster_stats
    )

def load_and_process_database(database_path: str) -> Tuple[List[Dict[str, float]], np.ndarray, List[str]]:
    """
    Load and process the composition database using monty.
    
    Args:
        database_path: Path to the JSON database
        
    Returns:
        Tuple containing:
        - List of composition dictionaries
        - Variance array
        - List of original formulas
    """
    data = loadfn(database_path)
    compositions_data = data['compositions']
    
    compositions = []
    variances = []
    formulas = []
    
    for entry in compositions_data:
        # Process composition
        comp_dict = process_composition(entry['formula'])
        compositions.append(comp_dict)
        
        # Get variance
        if isinstance(entry['variance_metrics'], dict):
            variance = entry['variance_metrics'].get('mean_variance', 0.0)
        else:
            variance = float(entry['variance_metrics'])
        
        variances.append(variance)
        formulas.append(entry['formula'])
    
    return compositions, np.array(variances), formulas

def modify_atoms_composition(template: Atoms, target_comp: Dict[str, float]) -> Atoms:
    """
    Modify template structure to match target composition.
    """
    print("\nModifying Composition:")
    atoms = template.copy()
    n_atoms = len(atoms)
    print(f"Template has {n_atoms} atoms")
    
    # Verify composition is valid
    constraints = {
        'V': (0.704, 1.0),
        'Cr': (0.008, 0.296),
        'Ti': (0.008, 0.296),
        'W': (0.008, 0.296),
        'Zr': (0.008, 0.296)
    }
    
    if not is_valid_composition(target_comp, constraints):
        raise ValueError(f"Invalid composition: {target_comp}")
    
    # Verify composition sums to 1
    total = sum(target_comp.values())
    if abs(total - 1.0) > 1e-6:
        normalized_comp = {k: v/total for k, v in target_comp.items()}
    else:
        normalized_comp = target_comp
    
    print(f"Target composition: {target_comp}")
    print(f"Normalized composition: {normalized_comp}")
    
    # First pass: calculate ideal atom counts
    ideal_counts = {
        elem: normalized_comp[elem] * n_atoms
        for elem in normalized_comp
    }
    print(f"Ideal atom counts: {ideal_counts}")
    
    # Initialize with floor values
    target_counts = {
        elem: max(1, int(count))
        for elem, count in ideal_counts.items()
    }
    
    # Calculate remaining atoms to distribute
    current_total = sum(target_counts.values())
    remaining = n_atoms - current_total
    print(f"Initial distribution: {target_counts}, remaining: {remaining}")
    
    if remaining > 0:
        # Sort elements by fractional part of ideal counts
        fractional_parts = [
            (elem, ideal_counts[elem] - target_counts[elem])
            for elem in ideal_counts
        ]
        fractional_parts.sort(key=lambda x: x[1], reverse=True)
        
        # Distribute remaining atoms
        for i in range(remaining):
            elem = fractional_parts[i % len(fractional_parts)][0]
            target_counts[elem] += 1
    elif remaining < 0:
        # Need to remove some atoms
        # Sort elements by count, remove from those with most
        while remaining < 0:
            elem = max(target_counts.items(), key=lambda x: x[1])[0]
            if target_counts[elem] > 1:  # Ensure at least 1 atom remains
                target_counts[elem] -= 1
                remaining += 1
    
    print(f"Final target counts: {target_counts}")
    total_atoms = sum(target_counts.values())
    print(f"Total atoms after distribution: {total_atoms}")
    
    if total_atoms != n_atoms:
        raise ValueError(f"Failed to match atom count. Target: {n_atoms}, Got: {total_atoms}")
    
    # Create new chemical symbols list
    new_symbols = []
    for elem, count in target_counts.items():
        new_symbols.extend([elem] * count)
    
    if len(new_symbols) != n_atoms:
        raise ValueError(f"Symbol list length ({len(new_symbols)}) != number of atoms ({n_atoms})")
    
    # Randomly shuffle
    random.shuffle(new_symbols)
    
    # Update atomic symbols
    atoms.symbols = new_symbols
    
    # Verify final composition
    final_comp = {}
    for symbol in atoms.get_chemical_symbols():
        final_comp[symbol] = final_comp.get(symbol, 0) + 1
    print(f"Final composition counts: {final_comp}")
    
    return atoms

def generate_structures(
    result: CompositionAnalysisResult,
    template_atoms: Atoms,
    output_dir: str
) -> None:
    """
    Generate structures for selected compositions and save as XYZ files.
    
    Args:
        result: CompositionAnalysisResult object
        template_atoms: Template structure to modify
        output_dir: Directory to save structures
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    structures_dir = output_dir / 'structures'
    structures_dir.mkdir(exist_ok=True)
    
    # Save structures and create info dictionary
    structures_info = []
    n_atoms = len(template_atoms)
    print(f"\nTemplate structure has {n_atoms} atoms")
    
    for idx, comp in enumerate(result.selected_compositions):
        print(f"\nGenerating structure {idx+1}/{len(result.selected_compositions)}")
        
        try:
            # Generate structure
            modified_atoms = modify_atoms_composition(template_atoms, comp)
            
            # Verify atom count
            if len(modified_atoms) != n_atoms:
                raise ValueError(f"Generated structure has {len(modified_atoms)} atoms, expected {n_atoms}")
            
            # Create descriptive composition string
            comp_str = '_'.join([f"{k}{v:.3f}" for k, v in comp.items()])
            filename = f"structure_{idx:03d}_{comp_str}.xyz"
            
            # Save structure
            write(str(structures_dir / filename), modified_atoms)
            
            # Store structure info
            structures_info.append({
                'filename': filename,
                'composition': comp,
                'n_atoms': len(modified_atoms)
            })
            
        except Exception as e:
            print(f"Error generating structure {idx}: {str(e)}")
            continue
    
    # Save structures info using monty
    dumpfn(structures_info, output_dir / 'structures_info.json')
    print(f"\nGenerated {len(structures_info)} structures successfully")

def plot_analysis_results(
    result: CompositionAnalysisResult,
    output_dir: str
):
    """Create visualization plots for the analysis results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # t-SNE plot
    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        result.embeddings[:, 0],
        result.embeddings[:, 1],
        c=result.clusters,
        cmap='viridis',
        alpha=0.6
    )
    plt.colorbar(scatter, label='Cluster')
    plt.title('t-SNE Visualization of Composition Space')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.savefig(output_dir / 'tsne_clusters.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Cluster composition heatmap
    cluster_compositions = pd.DataFrame([
        stats['mean_composition']
        for stats in result.cluster_stats.values()
    ])
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cluster_compositions, annot=True, fmt='.3f', cmap='YlOrRd')
    plt.title('Average Compositions per Cluster')
    plt.xlabel('Elements')
    plt.ylabel('Cluster')
    plt.savefig(output_dir / 'cluster_compositions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save cluster statistics using monty
    dumpfn(result.cluster_stats, output_dir / 'cluster_statistics.json')

def plot_composition_analysis(compositions: List[Dict[str, float]], 
                            selected_compositions: List[Dict[str, float]], 
                            output_dir: str):
    """
    Create detailed visualizations of composition distributions.
    """
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    output_dir = Path(output_dir)
    
    # Convert to arrays for easier plotting
    comp_array = np.array([[comp[elem] for elem in elements] for comp in compositions])
    selected_array = np.array([[comp[elem] for elem in elements] 
                              for comp in selected_compositions])
    
    # 1. Pairwise composition plots
    fig, axes = plt.subplots(5, 5, figsize=(20, 20))
    for i, elem1 in enumerate(elements):
        for j, elem2 in enumerate(elements):
            ax = axes[i, j]
            if i != j:
                # Plot all compositions
                ax.scatter(comp_array[:, j], comp_array[:, i], 
                         alpha=0.3, c='gray', s=20)
                # Highlight selected compositions
                ax.scatter(selected_array[:, j], selected_array[:, i],
                         alpha=0.8, c='red', s=50)
                ax.set_xlabel(elem2)
                ax.set_ylabel(elem1)
            else:
                # Histogram on diagonal
                ax.hist(comp_array[:, i], bins=20, alpha=0.3, color='gray')
                ax.hist(selected_array[:, i], bins=20, alpha=0.8, color='red')
                ax.set_xlabel(elem1)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'pairwise_compositions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Composition space coverage plot
    # Using parallel coordinates
    fig, ax = plt.subplots(figsize=(15, 8))
    
    # Plot all compositions in gray
    pd.DataFrame(comp_array, columns=elements).plot(
        ax=ax, color='gray', alpha=0.1, legend=False
    )
    
    # Plot selected compositions in red
    pd.DataFrame(selected_array, columns=elements).plot(
        ax=ax, color='red', alpha=0.8, legend=False
    )
    
    plt.xticks(range(len(elements)), elements)
    plt.ylabel('Composition Fraction')
    plt.title('Composition Space Coverage')
    plt.savefig(output_dir / 'composition_coverage.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save composition statistics
    stats = {
        'overall': {
            'mean': dict(zip(elements, np.mean(comp_array, axis=0))),
            'std': dict(zip(elements, np.std(comp_array, axis=0))),
            'min': dict(zip(elements, np.min(comp_array, axis=0))),
            'max': dict(zip(elements, np.max(comp_array, axis=0)))
        },
        'selected': {
            'mean': dict(zip(elements, np.mean(selected_array, axis=0))),
            'std': dict(zip(elements, np.std(selected_array, axis=0))),
            'min': dict(zip(elements, np.min(selected_array, axis=0))),
            'max': dict(zip(elements, np.max(selected_array, axis=0)))
        }
    }
    
    dumpfn(stats, output_dir / 'composition_statistics.json')

def iterative_composition_selection(
    compositions: List[Dict[str, float]],
    variances: np.ndarray,
    n_select: int = 150,
    batch_size: int = 5,
    random_state: int = 42
) -> List[Dict[str, float]]:
    """
    Iteratively select compositions using dynamic clustering.
    
    Args:
        compositions: List of composition dictionaries
        variances: Array of variance values
        n_select: Total number of compositions to select
        batch_size: Number of compositions to select before reclustering
        random_state: Random seed
        
    Returns:
        List of selected compositions
    """
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    comp_array = np.array([[comp[elem] for elem in elements] for comp in compositions])
    selected_compositions = []
    selected_indices = set()
    
    # Create mask for available compositions
    available_mask = np.ones(len(compositions), dtype=bool)
    
    while len(selected_compositions) < n_select and np.any(available_mask):
        # Get remaining compositions
        remaining_comps = comp_array[available_mask]
        remaining_vars = variances[available_mask]
        remaining_indices = np.where(available_mask)[0]
        
        # Calculate number of compositions to select in this iteration
        n_to_select = min(batch_size, n_select - len(selected_compositions))
        
        # Make sure we don't try to create more clusters than remaining points
        n_clusters = min(max(n_to_select, 5), len(remaining_comps))
        
        if n_clusters < 2:
            # If too few points remain, just select them all
            for idx in remaining_indices:
                if len(selected_compositions) >= n_select:
                    break
                comp = dict(zip(elements, comp_array[idx]))
                selected_compositions.append(comp)
                available_mask[idx] = False
            continue
            
        # Perform clustering on remaining compositions
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        clusters = kmeans.fit_predict(remaining_comps)
        
        # Select compositions from each cluster
        for cluster_idx in range(n_clusters):
            if len(selected_compositions) >= n_select:
                break
                
            # Get compositions in this cluster
            cluster_mask = clusters == cluster_idx
            if not np.any(cluster_mask):
                continue
                
            # Select highest variance composition from cluster
            cluster_vars = remaining_vars[cluster_mask]
            max_var_idx = np.argmax(cluster_vars)
            
            # Get the actual index in the original array
            original_idx = remaining_indices[np.where(cluster_mask)[0][max_var_idx]]
            
            # Add to selected compositions
            comp = dict(zip(elements, comp_array[original_idx]))
            selected_compositions.append(comp)
            available_mask[original_idx] = False
        
        # Print progress
        print(f"Selected {len(selected_compositions)}/{n_select} compositions "
              f"({np.sum(available_mask)} compositions remaining)")
    
    return selected_compositions

def analyze_composition_space(
    compositions: List[Dict[str, float]],
    variances: np.ndarray,
    n_generate: int = 150,
    batch_size: int = 5,
    random_state: int = 42
) -> CompositionAnalysisResult:
    """
    Analyze composition space and generate new compositions.
    """
    # Generate new compositions
    new_compositions = generate_new_compositions(
        existing_compositions=compositions,
        n_generate=n_generate,
        batch_size=batch_size,
        random_state=random_state
    )
    
    # Combine existing and new compositions for visualization
    all_compositions = compositions + new_compositions
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    comp_array = np.array([[comp[elem] for elem in elements] for comp in all_compositions])
    
    # Perform t-SNE on all compositions
    tsne = TSNE(n_components=2, random_state=random_state)
    embeddings = tsne.fit_transform(comp_array)
    
    # Cluster all compositions for visualization
    n_clusters = min(50, len(all_compositions) // 2)
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    clusters = kmeans.fit_predict(comp_array)
    
    # Calculate cluster statistics
    cluster_stats = {}
    for cluster_idx in range(n_clusters):
        mask = clusters == cluster_idx
        cluster_comps = comp_array[mask]
        
        cluster_stats[cluster_idx] = {
            'size': int(np.sum(mask)),
            'mean_composition': dict(zip(elements, np.mean(cluster_comps, axis=0))),
            'contains_new': bool(np.any(mask[len(compositions):]))
        }
    
    return CompositionAnalysisResult(
        selected_compositions=new_compositions,  # These are the new compositions to generate
        embeddings=embeddings,
        clusters=clusters,
        variance_scores=np.zeros(len(all_compositions)),  # Placeholder for viz
        cluster_stats=cluster_stats
    )

def is_valid_composition(comp: Dict[str, float], 
                        constraints: Dict[str, Tuple[float, float]] = None) -> bool:
    """
    Check if a composition satisfies the constraints.
    
    Args:
        comp: Composition dictionary
        constraints: Dictionary of (min, max) ranges for each element
        
    Returns:
        bool: Whether composition is valid
    """
    if constraints is None:
        constraints = {
            'V': (0.704, 1.0),
            'Cr': (0.008, 0.296),
            'Ti': (0.008, 0.296),
            'W': (0.008, 0.296),
            'Zr': (0.008, 0.296)
        }
    
    # Check if all elements are within their ranges
    for elem, (min_val, max_val) in constraints.items():
        if comp[elem] < min_val or comp[elem] > max_val:
            return False
            
    # Verify sum is approximately 1
    if abs(sum(comp.values()) - 1.0) > 1e-6:
        return False
        
    return True

def generate_valid_composition(constraints: Dict[str, Tuple[float, float]] = None) -> Dict[str, float]:
    """
    Generate a single valid composition satisfying constraints.
    """
    if constraints is None:
        constraints = {
            'V': (0.704, 1.0),
            'Cr': (0.008, 0.296),
            'Ti': (0.008, 0.296),
            'W': (0.008, 0.296),
            'Zr': (0.008, 0.296)
        }
    
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']  # Fixed order
    max_attempts = 1000
    
    for _ in range(max_attempts):
        # Initialize all elements with zeros
        comp = {elem: 0.0 for elem in elements}
        
        # Start with V (primary component)
        comp['V'] = np.random.uniform(constraints['V'][0], constraints['V'][1])
        remaining = 1.0 - comp['V']
        
        # Get other elements
        other_elements = [e for e in elements if e != 'V']
        
        # First ensure minimum values
        min_required = sum(constraints[e][0] for e in other_elements)
        if remaining < min_required:
            continue
            
        # Randomly distribute remaining fraction among other elements
        remaining_after_min = remaining - min_required
        
        # Set minimum values first
        for elem in other_elements:
            comp[elem] = constraints[elem][0]
        
        # Try to distribute remaining amount
        for i in range(len(other_elements) - 1):
            elem = other_elements[i]
            max_additional = min(
                constraints[elem][1] - comp[elem],
                remaining_after_min * (1.0 / (len(other_elements) - i))
            )
            if max_additional > 0:
                additional = np.random.uniform(0, max_additional)
                comp[elem] += additional
                remaining_after_min -= additional
        
        # Last element gets the remainder
        last_elem = other_elements[-1]
        comp[last_elem] += remaining_after_min
        
        # Verify composition is valid
        if is_valid_composition(comp, constraints):
            return comp
            
    raise ValueError("Could not generate valid composition after maximum attempts")

def generate_new_compositions(
    existing_compositions: List[Dict[str, float]],
    n_generate: int = 150,
    batch_size: int = 5,
    random_state: int = 42,
    constraints: Dict[str, Tuple[float, float]] = None
) -> List[Dict[str, float]]:
    """
    Generate new compositions by analyzing holes in composition space.
    """
    if constraints is None:
        constraints = {
            'V': (0.704, 1.0),
            'Cr': (0.008, 0.296),
            'Ti': (0.008, 0.296),
            'W': (0.008, 0.296),
            'Zr': (0.008, 0.296)
        }
    
    elements = ['V', 'Cr', 'Ti', 'W', 'Zr']
    comp_array = np.array([[comp[elem] for elem in elements] for comp in existing_compositions])
    
    # Perform t-SNE on existing compositions
    tsne = TSNE(n_components=2, random_state=random_state)
    embeddings = tsne.fit_transform(comp_array)
    
    # Initialize list for new compositions
    new_compositions = []
    
    print("\nGenerating new compositions...")
    attempts = 0
    max_attempts = n_generate * 10
    
    while len(new_compositions) < n_generate and attempts < max_attempts:
        try:
            # Generate a valid composition
            new_comp = generate_valid_composition(constraints)
            
            # Debug output
            print(f"\nAttempt {attempts+1}:")
            print(f"Generated composition: {new_comp}")
            print(f"Sum: {sum(new_comp.values()):.6f}")
            
            # Convert to array for distance calculation
            new_comp_array = np.array([new_comp[elem] for elem in elements])
            
            # Check distance from existing compositions
            if len(comp_array) > 0:
                min_distance = np.min(np.linalg.norm(comp_array - new_comp_array, axis=1))
            else:
                min_distance = float('inf')
            
            # Accept if sufficiently different from existing compositions
            if min_distance > 0.05:  # Threshold for minimum diversity
                new_compositions.append(new_comp)
                if len(new_compositions) % 10 == 0:
                    print(f"\nProgress: Generated {len(new_compositions)}/{n_generate} compositions")
        
        except ValueError as e:
            print(f"Warning: {str(e)}")
        
        attempts += 1
        if attempts % 100 == 0:
            print(f"\nAttempted {attempts} compositions...")
    
    if len(new_compositions) < n_generate:
        print(f"\nWarning: Could only generate {len(new_compositions)} valid compositions")
    else:
        print(f"\nSuccessfully generated {len(new_compositions)} compositions")
    
    return new_compositions


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze composition space and generate structures')
    parser.add_argument('database', type=str, help='Path to composition database JSON')
    parser.add_argument('template', type=str, help='Path to template structure file')
    parser.add_argument('--output', type=str, default='analysis_output',
                      help='Output directory for analysis results')
    parser.add_argument('--n_generate', type=int, default=150,
                      help='Number of new compositions to generate')
    parser.add_argument('--batch_size', type=int, default=5,
                      help='Number of compositions to select before reclustering')
    args = parser.parse_args()
    
    print("Loading database...")
    compositions, variances, formulas = load_and_process_database(args.database)
    print(f"Found {len(compositions)} existing compositions")

    # Load template structure
    template_atoms = read(args.template)
    
    print("\nAnalyzing composition space and generating new compositions...")
    result = analyze_composition_space(
        compositions=compositions,
        variances=variances,
        n_generate=args.n_generate,
        batch_size=args.batch_size
    )
    
    print(f"\nGenerated {len(result.selected_compositions)} new compositions")
    
    # Create visualizations
    print("\nCreating visualizations...")
    plot_analysis_results(result, args.output)
    
    # Generate structures
    print("\nGenerating structures...")
    generate_structures(result, template_atoms, args.output)

if __name__ == "__main__":
    main()