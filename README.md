# Overview
This repository contains a tutorial on how to use the MACE potential to train a machine learning interatomic potential, and how to use the trained model to select new configurations for future DFT calculations. 

# Training a MACE potential 

For this, we use the w-14 dataset, you can go through the Notebooks/train_mlip.ipynb notebook to see how we train the model. 

# Running molecular dynamics 

We will use inspiration from the MACE tutorial notebook to run molecular dynamics, but this will serve as an example of how we can use the trained model to select new configurations for future DFT calculations. 

# Environment Setup 

You will first need to create a conda environment. I recommend doing this from scratch, but you can try to use the environment.yaml file to create the environment. 

python 3.10 is the recommended version. I then installed the following packages:
- pytorch 2.4.0 
- mace-torch 0.3.7
- jupyterlab 4.3.0
- ipykernel 6.29.5
- ase 3.22.1
- matplotlib 3.8.1
- numpy 1.26.4
- pandas 2.1.4
- scikit-learn 1.4.2