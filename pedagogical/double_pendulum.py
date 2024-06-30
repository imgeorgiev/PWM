import hydra
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import random
import os

sns.set()


from dflex.envs.double_pendulum import DoublePendulumEnv

# TDMPC libraries
from tdmpc2 import TDMPC2
from tdmpc2_simple import TDMPC2 as TDMPC2_simple
from tdmpc2_no_simnorm import TDMPC2 as TDMPC2_no_simnorm
from common.parser import parse_cfg
from common.math import two_hot_inv
from common.init import weight_init


from IPython.core import ultratb
import sys

# For debugging
sys.excepthook = ultratb.FormattedTB(
    mode="Plain", color_scheme="Neutral", call_pdb=True
)


def seeding(seed=0):
    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    return seed


@hydra.main(config_name="config", config_path=".", version_base="1.1")
def main(cfg: dict):
    device = "cpu"
    seeding(0)
    env = DoublePendulumEnv(
        render=False,
        num_envs=1,
        episode_length=240,
        no_grad=False,
        stochastic_init=False,
        # start_state=[np.pi / 2, np.pi / 2, 0.0, 0.0],
        device=device,
    )
    H = 50
    samples = 100

    model_paths = {
        3: "models/tdmpc_h3.pt",
        # 8: "models/tdmpc_h8.pt",
        16: "models/tdmpc_h16.pt",
        # 32: "models/tdmpc_h32.pt",
    }

    tdmpcs = {}
    for key in model_paths:
        print(f"Loading {key}")
        cfg = parse_cfg(cfg)
        cfg.obs_shape = {"state": env.obs_space.shape}
        cfg.action_dim = env.act_space.shape[0]
        cfg.episode_length = env.episode_length
        clas = TDMPC2_simple
        # if key == "TDMPC_no_sn":
        #     clas = TDMPC2_no_simnorm
        # elif key == "TDMPC_simple":
        #     clas = TDMPC2_simple
        tdmpc = clas(cfg)

        tdmpc.load(model_paths[key])
        seeding(0)  # keep model weights the same
        tdmpc.model._pi.apply(weight_init)  # reset actor weight
        tdmpc.model.eval()
        tdmpcs.update({key: tdmpc})

    f, ax = plt.subplots(1, 2, figsize=(8, 3))
    ax = ax.flatten()

    print("Computing policy variance for dflex")
    all_grads = []
    seeding(0)
    for h in tqdm(range(1, H + 1)):
        grads = []
        for _ in range(samples):
            obs = env.reset(grads=True)
            tdmpc.pi_optim.zero_grad()
            rew = torch.zeros(env.num_envs).to(device)
            for t in range(h):
                z = tdmpc.model.encode(obs, None)
                a = tdmpc.model.pi(z.detach(), None)[1]
                obs, reward, done, info = env.step(a)
                # do reward transformation to be comparable with tdmpc
                reward /= 21554.2305
                reward += 1.0
                rew += reward

            (-rew).mean().backward()
            grads.append(get_grads(tdmpc.model._pi))
        grads = torch.vstack(grads)
        all_grads.append(grads)

    # Plotting!
    all_grads = torch.stack(all_grads)  # H x samples x n_params
    torch.save(all_grads, "dflex_grads.pt")
    variance = all_grads.var(dim=1).mean(dim=1)
    variance_std = all_grads.var(dim=1).std(dim=1)
    policy_snr = (all_grads.mean(dim=1) ** 2 / (all_grads.var(dim=1) + 1e-9)).mean(
        dim=1
    )
    ax[0].plot(range(H), variance, label="GT")
    ax[1].plot(range(H), policy_snr, label="GT")

    for key, tdmpc in tdmpcs.items():
        print(f"Computing policy variance for {key}")
        seeding(0)
        all_grads = []
        for h in tqdm(range(1, H + 1)):
            grads = []
            for _ in range(samples):
                obs = env.reset(grads=True)
                z = tdmpc.model.encode(obs, None)
                tdmpc.pi_optim.zero_grad()
                rew = torch.zeros(env.num_envs).to(device)
                for _ in range(h):
                    a = tdmpc.model.pi(z.detach(), None)[1]
                    r = tdmpc.model.reward(z, a, None)[0]
                    rew += r
                    z = tdmpc.model.next(z, a, None)

                (-rew).mean().backward()
                grads.append(get_grads(tdmpc.model._pi))
            grads = torch.vstack(grads)
            all_grads.append(grads)

        # Plotting!
        all_grads = torch.stack(all_grads)  # H x samples x n_params
        torch.save(all_grads, f"H={key}.pt")
        variance = all_grads.var(dim=1).mean(dim=1)
        variance_std = all_grads.var(dim=1).std(dim=1)
        policy_snr = (all_grads.mean(dim=1) ** 2 / (all_grads.var(dim=1) + 1e-9)).mean(
            dim=1
        )
        ax[0].plot(range(H), variance, label=f"H={key}")
        ax[1].plot(range(H), policy_snr, label=f"H={key}")

    print("Saving figure")
    ax[0].set_xlabel(r"$H$")
    ax[0].set_ylabel(r"Policy variance")
    ax[0].set_yscale("log")
    ax[0].legend()
    ax[1].set_xlabel(r"$H$")
    ax[1].set_ylabel(r"Policy ESNR")
    ax[1].set_yscale("log")
    ax[1].legend()
    plt.tight_layout()
    plt.savefig("sensitivity.pdf", bbox_inches="tight")


def get_grads(model):
    grads = []
    for param in model.parameters():
        grads.append(param.grad.view(-1))
    grads = torch.cat(grads)
    return grads


if __name__ == "__main__":
    main()
