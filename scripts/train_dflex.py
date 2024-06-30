import hydra, os, wandb, yaml
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra.core.hydra_config import HydraConfig
from pwm.utils import hydra_utils
from pwm.utils.common import seeding
from hydra.utils import instantiate

from IPython.core import ultratb
import sys

# For debugging
sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


def create_wandb_run(wandb_cfg, job_config, run_id=None):
    env_name = job_config["env"]["config"]["_target_"].split(".")[-1]
    try:
        alg_name = job_config["alg"]["_target_"].split(".")[-1]
    except:
        alg_name = job_config["alg"]["name"].upper()
    try:
        # Multirun config
        job_id = HydraConfig().get().job.num
        name = f"{alg_name}_{env_name}_sweep_{job_config['general']['seed']}"
        notes = wandb_cfg.get("notes", None)
    except:
        # Normal (singular) run config
        name = f"{alg_name}_{env_name}"
        notes = wandb_cfg["notes"]  # force user to make notes
    return wandb.init(
        project=wandb_cfg.project,
        config=job_config,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        name=name,
        notes=notes,
        id=run_id,
        resume=run_id is not None,
    )


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    cfg_full = OmegaConf.to_container(cfg, resolve=True)

    if cfg.general.run_wandb:
        create_wandb_run(cfg.wandb, cfg_full)

    # patch code to make jobs log in the correct directory when doing multirun
    logdir = HydraConfig.get()["runtime"]["output_dir"]
    logdir = os.path.join(logdir, cfg.general.logdir)

    seeding(cfg.general.seed, False)

    if "SHAC" in cfg.alg._target_ or "AHAC" in cfg.alg._target_:
        cfg.env.config.no_grad = False
    else:
        cfg.env.config.no_grad = True
    print(f"Running sim with no_grad={cfg.env.config.no_grad}")

    env = instantiate(cfg.env.config, logdir=logdir)
    print("num_envs = ", env.num_envs)
    print("num_actions = ", env.num_actions)
    print("num_obs = ", env.num_obs)

    agent = instantiate(
        cfg.alg,
        env=env,
        obs_dim=env.num_obs,
        act_dim=env.num_actions,
        logdir=logdir,
        log=cfg.general.run_wandb,
    )

    if cfg.general.checkpoint:
        agent.load(cfg.general.checkpoint, cfg.general.checkpoint_with_buffer)
        agent.wm_bootstrapped = True

    if cfg.general.pretrain:
        actually_train = True if not cfg.general.checkpoint else False
        agent.pretrain_wm(
            cfg.general.pretrain, cfg.general.pretrain_steps, actually_train
        )

    if cfg.general.train:
        agent.train()

    # evaluate the final policy's performance
    loss, discounted_loss, ep_len = agent.eval(cfg.general.eval_runs)
    print(
        f"mean episode loss = {loss:.2f}, mean discounted loss = {discounted_loss:.2f}, mean episode length = {ep_len:.2f}"
    )

    if cfg.general.run_wandb:
        wandb.finish()


if __name__ == "__main__":
    train()
