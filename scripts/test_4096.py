import modal 
import pathlib

modal_data_path = "/Users/myless/Packages/active_learning_mlips/modal_data_test"
app = modal.App("test_inference_4096_vcrti")
vol = modal.Volume.from_name("vcrti_4096_volume")
VOL_MOUNT_PATH = pathlib.Path("/vol")


image = modal.Image.debian_slim().pip_install("torch==2.4.0", 
                                              "torchvision==0.19.0",
                                              "torchaudo==2.4.0",
                                              extra_options="--index-url https://download.pytorch.org/whl/cu124")
image = image.pip_install()
image = (modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04", add_python="3.11"
    )
    .pip_install("torch==2.4.0", "torchvision==0.19.0", "torchaudio==2.4.0", extra_options="--index-url https://download.pytorch.org/whl/cu124"
    )
    .pip_install("mace-torch==0.3.6")
)
#gpu=modal.gpu.A100(size="80GB")
@app.function(cpu=4, 
              memory=32768, 
              #gpu=modal.gpu.A100(size="40GB"),
              gpu="A10G",
              image=image, 
              mounts=[modal.Mount.from_local_dir(modal_data_path, remote_path="/root/data")],
              volumes={VOL_MOUNT_PATH: vol},
              timeout=12*3600)
def run_test(test_atoms, calculator):

    from ase.io import read
    from mace.calculators.mace import MACECalculator
    from ase.optimize.precon import PreconLBFGS
    from ase.filters import FrechetCellFilter
    
    atoms = read(test_atoms)
    calc = MACECalculator(model_paths=[calculator], device='cuda', default_dtype='float32')
    atoms.calc = calc
    #ucf = FrechetCellFilter(atoms)
    optimizer = PreconLBFGS(atoms,use_armijo=True)
    optimizer.run(fmax=0.01, steps=100)
    print(atoms.get_potential_energy())


@app.local_entrypoint()
def main():
    # read in the data
    #train_dataset = '/root/data/fep+vac+neb_train.xyz'
    #test_dataset = '/root/data/fep+vac+neb_test.xyz'
    #train_dataset = '/root/data/fep_vac_neb_perf_train.xyz'
    #train_dataset = '/root/data/fep_vac_neb_perf_iso_train.xyz'
    #test_dataset = '/root/data/fep_vac_neb_perf_test.xyz'
    #test_atoms = read('../Visualization/Job_Structures/Pre_VASP/Old/VCrTi_Fixed_4096/V0_90625-Cr0_046875-Ti0_046875_middle.cif')
    #test_atoms = '/root/data/V0_90625-Cr0_046875-Ti0_046875_middle.cif'
    #'./modal_data_test/V0_901-Cr0_023-Ti0_076_middle_2661.cif'
    #test_atoms = '/root/data/V0_90625-Cr0_046875-Ti0_046875_middle.cif'
    test_atoms = '/root/data/V0_604-Cr0_301-Ti0_095_middle_2623.cif'
    calc = '/root/data/vcrti_vac_stress_e1_f50_s25_converted_complex_stagetwo_compiled.model'

    #calc = MACECalculator(model_paths=['../Potentials/vcrti_vac_stress_e1_f50_s25_converted_complex_stagetwo_compiled.model'], device='cpu', default_dtype='float32')
    run_test.remote(test_atoms, calc)

