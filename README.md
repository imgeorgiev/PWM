# PWM: Policy Learning with Large World Models

[Ignat Georgiev](https://www.imgeorgiev.com/), [Varun Giridhar](https://www.linkedin.com/in/varun-giridhar-463947146/), [Nicklas Hansen](https://www.nicklashansen.com/), [Animesh Garg](https://animesh.garg.tech/)

[Project website](https://policy-world-model.github.io/)  [Paper](TODO)  [Models & Datasets](https://huggingface.co/imgeorgiev/pwm)

This repository is a soft fork of [FoRL](https://github.com/pairlab/FoRL).

## Overview

![](https://policy-world-model.github.io/media/wm-animation.mp4)

Instead of building world models into algorithms, we propose using large-scale multi-task world models as
differentiable simulators for policy learning. When well-regularized, these models enable efficient policy
learning with first-order gradient optimization. This allows PWM to learn to solve 80 tasks in < 10 minutes
each without the need for expensive online planning.


## Installation

Tested only on Ubuntu 22.04. Requires Python, conda and an Nvidia GPU with >24GB VRAM.

1. `git clone --recursive git@github.com:imgeorgiev/PWM.git`
2. `cd PWM`
3. `conda env create -f environment.yaml`
4. `ln -s $CONDA_PREFIX/lib $CONDA_PREFIX/lib64` (hack to get CUDA to work inside conda)
5. `pip install -e .`

## Single environment tasks

The first option for running PWM is on complex single-tasks with up to 152 action dimensions in the Dflex simulator. These runs used pre-trained world models which can be [downloaded from hugging face](https://huggingface.co/imgeorgiev/pwm/tree/main/dflex/pretrained).

```
cd scripts
conda activate forl
python train_dflex.py env=dflex_ant alg=pwm general.checkpoint=path/to/model
```

> Due to the nature of GPU acceleration, it is impossible to currently impossible to guarantee deterministic experiments. You can make them "less random" by using `seeding(seed, True)` but that slows down GPUs.

## Multitask setting

### Training large world model

We evaluate on the MT30 and MT80 task settings proposed by [TDMPC2](https://www.tdmpc2.com/).

1. Download the data for each task following the [TDMPC2 instructions](https://www.tdmpc2.com/dataset)
2. Train a world model from the TDMPC2 repository using the settings below. Note that `horizon=15` and `rho=0.99` are crucial. Note that training takes ~2 weeks on an RTX 3900. Alternatively, you can also use some of the pre-trained [multi-task world models we provide](https://huggingface.co/imgeorgiev/pwm/tree/main/multitask).
```
cd external/tdmpc2/tdmpc2
python -u train.py task=mt30 model_size=48 horizon=15 batch_size=1024 rho=0.99 mpc=false disable_wandb=False data_dir=path/to/data
```


### Evaluate multi-task

Train a policy for a specific task using the pre-trained world model

```
cd scripts
python train_multitask.py -cn config_mt30 general.run_wandb=True wandb.group=mt30 alg=pwm_48M task=pendulum-swingup
```

We also provide scripts which launch slurm tasks across all tasks. `scripts/mt30.bash` and `scripts/mt80.bash`

### Configs
```
cfg
├── alg
│   ├── pwm_19M.yaml - different sized PWM models which the main models that should be used. Paired with train_multitask.py
│   ├── pwm_317M.yaml - to be used with train_multitask.py
│   ├── pwm_48M.yaml 
│   ├── pwm_5M.yaml
│   ├── pwm.yaml - redunant but provided for reproducability; to be run with train_dflex.py
│   └── shac.yaml - works only with train_dflex.py
├── config_mt30.yaml - to be used with train_multitask.py
├── config_mt80.yaml - to be used with train_multitask.py
├── config.yaml  - to be used with train_dflex.py
└── env - dflex env config files
    ├── dflex_ant.yaml
    ├── dflex_anymal.yaml
    ├── dflex_cartpole.yaml
    ├── dflex_doublependulum.yaml
    ├── dflex_hopper.yaml
    ├── dflex_humanoid.yaml
    └── dflex_snu_humanoid.yaml
```

## Citation


```
@misc{georgiev2024pwm,
    title={PWM: Policy Learning with Large World Models},
    author={Ignat Georgiev, Varun Giridha, Nicklas Hansen, and Animesh Garg},
    eprint={2405.18418},  # TODO update
    archivePrefix={arXiv},
    primaryClass={cs.LG},
    year={2024}
}
```

# TODOs

- [x] upload pedagogical models
- [x] figure out which version of PWM I ran
- [x] find pretrained models for each env
- [x] upload data for each env
- [x] Upload multi-task models
- [x] merge PWM online and offline into the same model
- [x] figure out a way to integrate world model with and without terminate into one
- [x] train new dflex models and confirm that they work
- [x] update and commit results
- [x] write up readme
- [x] clean up and commit changes
- [ ] replicate results from readme

