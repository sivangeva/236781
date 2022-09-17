# Universal adversarial perturbations on visual odometry systems
> This is a final project submission for [**CS236781 — Deep Learning on Computational Accelerators**](https://vistalab-technion.github.io/cs236781)
>
> Date: September, 2022. 
## Authors

* Sivan Schwartz @ [Technion](mailto:sivan.s@campus.technion.ac.il)
* Sivan Geva @ [Technion](mailto:sivangeva@campus.technion.ac.il)

## Environment setup

This project is built with `PyTorch v1.10` and `Cupy v10.6`. All dependencies are listed in 
`src\environment.yml`, and below are methods to setup your local development 
environment.

### Anaconda/Miniconda

Run the following from the project's root folder:
```bash
conda env create -f environment.yml
conda activate pytorch-cupy-3
```

## Datasets

The synthetic data produced by blender used in Nemcovsky et al.,2022 work was provided to us and was used for running the experiments in this project. For a complete description of using the data see here.

## Structure

### Code

All code is implemented in the `src/` directory, which includes:
* `models/`: includes the previous weights of TertanVO model implementation used for running the experiments: `tartanvo_1914.pkl`. 

* `data/`: contains dataset used in this project.
    
    -`VO_adv_project_train_dataset_8_frames_processed/`: preproccesed data for faster loading the next time.
* `Datasets/`: include methods for preparing the preprocessed datasets, as well as
backing them up. \

* [TartanVO.py](src/TartanVO.py): ITertanVO model implementation used for running the experiments
* [run_attacks.py](src/run_attacks.py): Contains high level optimization implementation using CrossValidation and data split to train, evaluation and test sets. 
* [run_attacks.py](src/run_attacks.py): Contains high level optimization implementation using CrossValidation and data split to train, evaluation and test sets.


### Notebooks

This project includes 2 notebooks with several experiments on utilities:

1. [Input.ipynb](src/Input.ipynb): Contains several examples on using the data and Wavelet decomposition implemented 
in this project
2. [Model.ipynb](src/Model.ipynb): Contains data loading and train/test experiments for baseline and advanced models.

## Experiments

The experiments were run on a single-GPU architecture (Nvidia GPU with CUDA).
 
To reproduce the results, start by running `jupyter-lab` connected to the GPU device.

#### Slurm
For slurm, you can do this with the following script (`jupyter-lab.sh`):
```bash
#!/bin/bash
unset XDG_RUNTIME_DIR
jupyter notebook --no-browser --ip=$(hostname -I) --port-retries=100
```

Then, summon a job for the notebook: `srun -c2 --gres=gpu:1 --pty ./jupyter-lab.sh`.

Experiments with the hyperparameters are available via the `Model` notebook.
