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
* `models/`: includes the model implementations used for running the experiments: 
[Baseline (baseline.py)](src/model/baseline.py) for the baseline model, [TCNNet (tcnnet.py)](src/model/tcnnet.py) for
the Temporal Convolutional model. [Blocks (blocks.py)](src/model/blocks.py) contains useful building blocks
used inside the model. See the paper for more details on implementation.
* `data/`: contains dataset used in this project.
  -`VO_adv_project_train_dataset_8_frames_processed/`: preproccesed data for faster loading the next time.
* `Datasets/`: include methods for preparing the preprocessed datasets, as well as
backing them up. \
  All datasets include a `load()` function for preparing the preprocessed datasets, as well as
backing them up for faster loading the next time. \
See code for parameters and usage documentation
* [dsp.py](src/dsp.py): Includes some utility functions used by the `WaveletTransform` module to perform
signal processing using *scipy* and *pycwt*.
* [training.py](src/training.py): Contains train/test functions for running the experiments.

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
