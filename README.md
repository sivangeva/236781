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
backing them up. 

* `attacks/`: include different attack methods for the experiments.


    -[attack.py](src/attack.py): Contains all functions needed to run the attacks.

    -[PGD.py](src/PGD.py): Inhetired class, contains the PGD attack.
    
    -[APGD.py](src/APGD.py): Inhetired class, contains the APGD attack.

* [TartanVO.py](src/TartanVO.py): TertanVO model implementation used for running the experiments
* [run_attacks.py](src/run_attacks.py): Contains high level optimization implementation using CrossValidation and data split to train, evaluation and test sets. 
* [loss.py](src/loss.py): Contains loss implementationfor running the experiments.
* [utils.py](src/utils.py): Contains all arguments options for running the experiments.


## Experiments

The experiments were run on a single-GPU architecture (Nvidia GPU with CUDA).
 
#### Slurm
For slurm, you can do this with the following command:

`srun -c 2 --gres=gpu:1 --pty python src/run_attacks.py --seed 42 --model-name tartanvo_1914.pkl --test-dir "VO_adv_project_train_dataset_8_frames"  --max_traj_len 8 --batch-size 1 --worker-num 1 --save_csv --attack pgd --attack_k 100 --attack_oos`.

