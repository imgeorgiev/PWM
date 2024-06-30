#!/bin/bash

# List of parameters
params=(
        # 19 original dmcontrol tasks
        # "walker-stand" \
        # "walker-walk" \
        # "walker-run" \
        # "cheetah-run" \
        # "reacher-easy" \
        # "reacher-hard" \
        # "acrobot-swingup" \
        # "pendulum-swingup" \
        # "cartpole-balance" \
        # "cartpole-balance-sparse" \
        # "cartpole-swingup" \
        # "cartpole-swingup-sparse" \
        # "cup-catch" \
        # "finger-spin" \
        # "finger-turn-easy" \
        # "finger-turn-hard" \
        # "fish-swim" \
        # "hopper-stand" \
        # "hopper-hop" \
        # 11 custom dmcontrol tasks
        # "walker-walk-backwards" \
        # "walker-run-backwards" \
        "cheetah-run-backwards" \
        "cheetah-run-front" \
        "cheetah-run-back" \
        "cheetah-jump" \
        "hopper-hop-backwards" \
        "reacher-three-easy" \
        "reacher-three-hard" \
        "cup-spin" \
        "pendulum-spin" \
        )

# Command to run for each parameter
for param in "${params[@]}"
do
    echo "Running command with parameter: $param"
    # Replace 'your_command' with the actual command you want to run
    # your_command "$param"
    sbatch run.sh python train_multitask.py --multirun -cn config_mt30 general.run_wandb=True wandb.group=mt30 alg=pwm_48M task="$param"
    sleep 30 # to ensure that the command has finished before running the next one
done
