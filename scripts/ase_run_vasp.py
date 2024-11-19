import os
from ase import Atoms
from ase.io import read, write
from ase.calculators.vasp import Vasp
from ase.stress import full_3x3_to_voigt_6_stress
from datetime import datetime

def setup_environment():
    # Set up environment variables
    os.environ['LD_LIBRARY_PATH'] = ":".join([
        "/home/myless/intel/oneapi/compiler/2021.3.0/lib",
        "/home/myless/intel/oneapi/mkl/2021.3.0/lib/intel64",
        "/home/myless/hpc_sdk/Linux_x86_64/22.5/compilers/extras/qd/lib",
        os.environ.get('LD_LIBRARY_PATH', '')
    ])

    os.environ['NVHPC_CUDA_HOME'] = "/usr/local/cuda-11.7/"
    os.environ['PATH'] = f"{os.environ['NVHPC_CUDA_HOME']}/bin:{os.environ['PATH']}"
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['NVHPC_CUDA_HOME']}/lib64:{os.environ['LD_LIBRARY_PATH']}"

    # NVIDIA HPC SDK
    os.environ['NVHPC_ROOT'] = "/home/myless/Packages/hpc_sdk/Linux_x86_64/22.5"
    os.environ['PATH'] = f"{os.environ['NVHPC_ROOT']}/compilers/bin:{os.environ['PATH']}"
    os.environ['MANPATH'] = f"{os.environ.get('MANPATH', '')}:{os.environ['NVHPC_ROOT']}/comm_libs/mpi/man:{os.environ['NVHPC_ROOT']}/compilers/man"
    os.environ['PATH'] = f"{os.environ['NVHPC_ROOT']}/comm_libs/mpi/bin:{os.environ['NVHPC_ROOT']}/compilers/extras:{os.environ['PATH']}"
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['NVHPC_ROOT']}/compilers/extras/qd/lib:{os.environ['LD_LIBRARY_PATH']}"
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['NVHPC_ROOT']}/cuda/11.7/targets/x86_64-linux/lib:{os.environ['LD_LIBRARY_PATH']}"
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['NVHPC_ROOT']}/math_libs/lib64:{os.environ['LD_LIBRARY_PATH']}"

    # OpenMPI
    os.environ['OMPI_ROOT'] = f"{os.environ['NVHPC_ROOT']}/comm_libs/openmpi4/openmpi-4.0.5"
    os.environ['LD_LIBRARY_PATH'] = f"{os.environ['OMPI_ROOT']}/lib:{os.environ['LD_LIBRARY_PATH']}"
    os.environ['PATH'] = f"{os.environ['OMPI_ROOT']}/bin:{os.environ['PATH']}"
    os.environ['OPAL_PREFIX'] = os.environ['OMPI_ROOT']

    os.environ['VASP_PP_PATH'] = "/home/myless/VASP/POTCAR_64_PBE"
    
def is_job_complete(output_dir):
    """
    Check if a job is complete by looking for the OUTCAR file and checking its content.
    """
    outcar_path = os.path.join(output_dir, 'OUTCAR')
    if not os.path.exists(outcar_path):
        return False
    
    with open(outcar_path, 'r') as f:
        content = f.read()
        # Check for phrases that typically indicate job completion
        if 'reached required accuracy' in content and 'General timing and accounting informations' in content:
            return True
    return False
    
def check_pseudopotentials(elements):
    pp_path = os.environ.get('VASP_PP_PATH')
    if not pp_path:
        raise EnvironmentError("VASP_PP_PATH is not set. Please set it to the path of your VASP pseudopotentials.")
    
    pbe_path = os.path.join(pp_path)
    if not os.path.exists(pbe_path):
        raise FileNotFoundError(f"PBE pseudopotential directory not found at {pbe_path}")
    
    for element in elements:
        element_path = os.path.join(pbe_path, 'potpaw_PBE',element)
        element_pv_path = os.path.join(pbe_path, 'potpaw_PBE',f"{element}_pv")
        element_sv_path = os.path.join(pbe_path, 'potpaw_PBE',f"{element}_sv")
        if not (os.path.exists(element_path) or os.path.exists(element_sv_path) or os.path.exists(element_pv_path)):
            raise FileNotFoundError(f"Pseudopotential for {element} not found in {pbe_path}")

def get_vasp_potential(symbol):
    pp_path = os.environ.get('VASP_PP_PATH')
    if not pp_path:
        raise EnvironmentError("VASP_PP_PATH is not set. Please set it to the path of your VASP pseudopotentials.")
    
    pbe_path = os.path.join(pp_path, 'potpaw_PBE')
    
    # Check for _pv version first
    if os.path.exists(os.path.join(pbe_path, f"{symbol}_pv")):
        return "_pv"
    elif os.path.exists(os.path.join(pbe_path, f"{symbol}_sv")):
        return "_sv"
    # Then check for standard version
    #elif os.path.exists(os.path.join(pbe_path, symbol)):
        #return symbol
    else:
        raise FileNotFoundError(f"No pseudopotential found for {symbol} in {pbe_path}")

def run_static_vasp(atoms, output_dir, all_results_file, create_inputs_only=False):
    if os.path.exists(os.path.join(output_dir,'result.extxyz')) and not create_inputs_only:
        print(f"Job in {output_dir} is already complete. Skipping.")
        return

    # Check pseudopotentials before setting up the calculator
    unique_elements = set(atoms.get_chemical_symbols())
    check_pseudopotentials(unique_elements)
    
    # Determine appropriate setups for each element
    setups = {symbol: get_vasp_potential(symbol) for symbol in unique_elements}
    
    # VASP calculator settings
    calc = Vasp(command="/home/myless/Packages/hpc_sdk/Linux_x86_64/22.5/comm_libs/mpi/bin/mpirun /home/myless/VASP/vasp.6.4.2/bin/vasp_std",
        prec='Accurate',
        encut=520,
        ediff=1e-6,
        ediffg=-0.01,
        nelm=100,
        nsw=0,
        ibrion=-1,
        ismear=1,
        sigma=0.2,
        lcharg=False,
        lwave=False,
        lreal=False,
        lorbit=11,
        xc='PBE',
        kpts=(4, 4, 4),
        gamma=True,
        setups=setups,
        directory=output_dir,
        # GPU-specific settings
        algo='Normal',
    )
    
    atoms.calc = calc

    if create_inputs_only:
        # Create input files without running the calculation
        calc.initialize(atoms)
        calc.write_input(atoms)
        print(f"Input files created in {output_dir}")
        return

    # Run the calculation
    try:
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        stress = atoms.get_stress(voigt=False)
        
        # Add results to the Atoms object
        atoms.info['ENGRAD_energy'] = energy
        atoms.arrays['ENGRAD_forces'] = forces
        atoms.info['ENGRAD_stress'] = stress
        
        # Save results as extXYZ in the job directory
        job_results_file = os.path.join(output_dir, 'result.extxyz')
        write(job_results_file, atoms, format='extxyz')

        # Append results to the all_results file
        write(all_results_file, atoms, format='extxyz', append=True)
        print(f"Calculation completed for {output_dir}")
    except Exception as e:
        print(f"Error in calculation for {output_dir}: {str(e)}")

from datetime import datetime
import re
from ase.io import read
from glob import glob

def parse_xyz_filename(filename):
    """
    Parse different types of XYZ filenames:
    1. MD simulation files: structure_021_V0.861_Cr0.032_Ti0.012_W0.047_Zr0.048_T3000_md.xyz
    2. Adversarial files: 
       - neb_unfinished_fin_2_structure_1768_step_0_adversarial.xyz
       - vac_structure_924_step_0_adversarial.xyz
    
    Returns:
        tuple: (idx, comp, temp, file_type, source_folder)
        - For MD files: (idx number, composition, temperature, "md", None)
        - For adversarial files: (structure number, None, None, "adversarial", source_folder)
    """
    basename = os.path.basename(filename)
    
    # Pattern for new MD simulation files
    md_pattern = r'structure_(\d+)_([V0-9._CrTiWZ]+)_T(\d+)_md\.xyz'
    md_match = re.match(md_pattern, basename)
    
    # Pattern for adversarial files
    adv_pattern = r'(.+?)_structure_(\d+)_step_\d+_adversarial\.xyz'
    adv_match = re.match(adv_pattern, basename)
    
    if md_match:
        idx = md_match.group(1)          # Structure index (e.g., '021')
        comp = md_match.group(2)         # Full composition string
        temp = md_match.group(3)         # Temperature value
        return idx, comp, temp, "md", None
    elif adv_match:
        source_folder = adv_match.group(1)  # Everything before "structure"
        structure_num = adv_match.group(2)   # Structure number
        return structure_num, None, None, "adversarial", source_folder
    
    return None, None, None, None, None

def main():
    setup_environment()
    base_output_dir = '../vasp_jobs/zr-w-v-ti-cr/'
    job_generation = 5
    current_date = datetime.now().strftime('%Y-%m-%d')
    job_directory = os.path.join(base_output_dir, f'job_gen_{job_generation}-{current_date}')

    # Create directories
    os.makedirs(base_output_dir, exist_ok=True)
    os.makedirs(job_directory, exist_ok=True)

    all_results_file = os.path.join(base_output_dir, f'job_gen_{job_generation}','all_results.extxyz')
    #xyz_directory = '/home/myless/Packages/al_mlip_repo/data/zr-w-v-ti-cr/gen_1_2024-11-09/aa_out/'
    xyz_directory = '/home/myless/Packages/al_mlip_repo/data/zr-w-v-ti-cr/gen_5_2024-11-17/md_output/'
    
    # Get all xyz files in the directory
    xyz_files = glob(os.path.join(xyz_directory, '*.xyz'))

    # Process each xyz file
    for xyz_file in xyz_files:
        # Parse filename to get metadata
        idx, comp, temp, file_type, source_folder = parse_xyz_filename(xyz_file)
        
        if idx is None:
            print(f"Warning: Could not parse filename {xyz_file}, skipping...")
            continue

        # Read all frames from the xyz file
        atoms_list = read(xyz_file, index='1:', format='extxyz')

        # Process each frame
        for i, atoms in enumerate(atoms_list):
            # Create output directory name based on file type
            if file_type == "md":
                output_dir = os.path.join(
                    job_directory,
                    f'structure_{i}_idx_{idx}_comp_{comp}_temp_{temp}'
                )
            else:  # adversarial
                output_dir = os.path.join(
                    job_directory,
                    f'structure_{i}_{source_folder}_adversarial_{idx}'
                )
                
            os.makedirs(output_dir, exist_ok=True)
            run_static_vasp(atoms, output_dir, all_results_file, create_inputs_only=True)

if __name__ == "__main__":
    main()
