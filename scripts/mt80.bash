#!/bin/bash

# List of parameters
params=(
        # 19 original dmcontrol tasks
        "walker-stand" \
        "walker-walk" \
        "walker-run" \
        "cheetah-run" \
        "reacher-easy" \
        "reacher-hard" \
        "acrobot-swingup" \
        "pendulum-swingup" \
        "cartpole-balance" \
        "cartpole-balance-sparse" \
        "cartpole-swingup" \
        "cartpole-swingup-sparse" \
        "cup-catch" \
        "finger-spin" \
        "finger-turn-easy" \
        "finger-turn-hard" \
        "fish-swim" \
        "hopper-stand" \
        "hopper-hop" \
        # 11 custom dmcontrol tasks
        "walker-walk-backwards" \
        "walker-run-backwards" \
        "cheetah-run-backwards" \
        "cheetah-run-front" \
        "cheetah-run-back" \
        "cheetah-jump" \
        "hopper-hop-backwards" \
        "reacher-three-easy" \
        "reacher-three-hard" \
        "cup-spin" \
        "pendulum-spin" \
        # meta-world mt50
        "mw-assembly" \
        "mw-basketball" \
        "mw-button-press-topdown" \
        "mw-button-press-topdown-wall" \
        "mw-button-press" \
        "mw-button-press-wall" \
        "mw-coffee-button" \
        "mw-coffee-pull" \
        "mw-coffee-push" \
        "mw-dial-turn" \
        "mw-disassemble" \
        "mw-door-open" \
        "mw-door-close" \
        "mw-drawer-close" \
        "mw-drawer-open" \
        "mw-faucet-open" \
        "mw-faucet-close" \
        "mw-hammer" \
        "mw-handle-press-side" \
        "mw-handle-press" \
        "mw-handle-pull-side" \
        "mw-handle-pull" \
        "mw-lever-pull" \
        "mw-peg-insert-side" \
        "mw-peg-unplug-side" \
        "mw-pick-out-of-hole" \
        "mw-pick-place" \
        "mw-pick-place-wall" \
        "mw-plate-slide" \
        "mw-plate-slide-side" \
        "mw-plate-slide-back" \
        "mw-plate-slide-back-side" \
        "mw-push-back" \
        "mw-push" \
        "mw-push-wall" \
        "mw-reach" \
        "mw-reach-wall" \
        "mw-shelf-place" \
        "mw-soccer" \
        "mw-stick-push" \
        "mw-stick-pull" \
        "mw-sweep-into" \
        "mw-sweep" \
        "mw-window-open" \
        "mw-window-close" \
        "mw-bin-picking" \
        "mw-box-close" \
        "mw-door-lock" \
        "mw-door-unlock" \
        "mw-hand-insert"
        )

# Command to run for each parameter
for param in "${params[@]}"
do
    echo "Running command with parameter: $param"
    # Replace 'your_command' with the actual command you want to run
    # your_command "$param"
    sbatch run.sh python train_multitask.py --multirun -cn config_mt80 general.run_wandb=True wandb.group=mt80 alg=pwm_48M task="$param"
    sleep 30 # to ensure that the command has finished before running the next one
done
