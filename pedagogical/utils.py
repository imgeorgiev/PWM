from dflex.envs import AntEnv
from pwm.models.actor import ActorDeterministicMLP, ActorStochasticMLP
from pwm.utils.common import seeding
import torch
import sys
from IPython.core import ultratb
import seaborn as sns
import hydra
from time import time
from torch.autograd.functional import jacobian as jac2


sns.set()
colors = sns.color_palette()

sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


def jacobian(output: torch.Tensor, input: torch.Tensor) -> torch.Tensor:
    """
    Computes the jacobian of `output` tensor with respect to the `input`

    returns a tensor of shape [B, out_dim, in_dim] where B is the batch size
    """
    assert input.dim() == 2, "Input must have 2 dimensions [batch, in_dim]"
    B, in_dim = input.shape
    out_dim = output.shape[-1]
    jacobians = torch.zeros((B, out_dim, in_dim), dtype=input.dtype)
    for out_idx in range(
        min(out_dim, 11)
    ):  # 11 is a dflex limitation here TODO need to fix in dflex
        select_index = torch.zeros(output.shape[1], device=input.device)
        select_index[out_idx] = 1.0
        e = torch.tile(select_index, (B, 1))
        (grad,) = torch.autograd.grad(
            outputs=output, inputs=input, grad_outputs=e, retain_graph=True
        )
        jacobians[:, out_idx, :] = grad.view(B, in_dim)

    return jacobians


# def dflex_jacobian(env, _, act):
#     act = env.unscale_act(act)
#     inputs = torch.cat((env.obs_buf.clone(), act.clone()), dim=1)
#     inputs.requires_grad_(True)
#     last_obs = inputs[:, : env.num_obs]
#     act = inputs[:, env.num_obs :]
#     # env.set_state_act(last_obs, act)
#     output = env.integrator.forward(
#         env.model,
#         env.state,
#         env.sim_dt,
#         env.sim_substeps,
#         env.MM_caching_frequency,
#         False,
#     )
#     outputs = env.observation_from_state(output)
#     breakpoint()
#     jac = jacobian(outputs, inputs)
#     return jac


# for double pendulum in particular
def dflex_jacobian(env, _, act):
    act = env.unscale_act(act)
    num_envs = env.num_envs
    inputs = torch.cat(
        (
            env.state.joint_q.view(num_envs, -1),
            env.state.joint_qd.view(num_envs, -1),
            act.clone(),
        ),
        dim=1,
    )
    inputs.requires_grad_(True)
    # last_obs = inputs[:, : env.num_obs]
    # act = inputs[:, env.num_obs :]
    # env.set_state_act(last_obs, act)

    env.state.joint_q.view(num_envs, -1)[:, 0] = inputs[:, 0]
    env.state.joint_q.view(num_envs, -1)[:, 1] = inputs[:, 1]
    env.state.joint_qd.view(num_envs, -1)[:, 0] = inputs[:, 2]
    env.state.joint_qd.view(num_envs, -1)[:, 1] = inputs[:, 3]
    env.state.joint_act.view(num_envs, -1)[:, 1] = inputs[:, 4]

    output = env.integrator.forward(
        env.model,
        env.state,
        env.sim_dt,
        env.sim_substeps,
        env.MM_caching_frequency,
        False,
    )
    outputs = env.observation_from_state(output)
    # breakpoint()
    jac = jacobian(outputs, inputs)
    return jac


@hydra.main(config_name="config", config_path=".")
def main(cfg: dict):

    # This here is just for testing

    device = "cuda"

    seeding(42, True)
    env = AntEnv(num_envs=1, no_grad=False, early_termination=False, device=device)
    actor = ActorStochasticMLP(
        env.num_obs, env.num_acts, [400, 200, 100], torch.nn.Mish
    ).to(device)
    chkpt = torch.load(
        "/storage/home/hcoda1/7/igeorgiev3/git/FoRL/scripts/outputs/2024-03-14/15-28-46/logs/best_policy.pt",
        map_location=device,
    )
    actor.load_state_dict(chkpt["actor"])
    obs_rms = chkpt["obs_rms"].to(device)

    from tdmpc2 import TDMPC2
    from common.parser import parse_cfg
    from common.math import two_hot_inv

    cfg = parse_cfg(cfg)
    cfg.obs_shape = {"state": env.obs_space.shape}
    cfg.action_dim = env.act_space.shape[0]
    cfg.episode_length = env.episode_length
    tdmpc = TDMPC2(cfg)
    tdmpc.load(
        "/storage/home/hcoda1/7/igeorgiev3/git/FoWM/wmlab/logs/dflex-ant/1/default/models/20000.pt"
    )

    obs = env.reset()
    obs = env.initialize_trajectory()
    obs = obs_rms.normalize(obs)

    z = tdmpc.model.encode(obs, None)
    act = actor(obs)

    z_next = tdmpc.model.next(z, act, None)

    now = time()
    jac = jacobian(z_next, z)
    jac = jacobian(z_next, act)
    print("Took {:.2f}s".format(time() - now))

    now = time()
    jac = jac2(tdmpc.model.next, (z, act, torch.zeros((1,))))
    print("Took {:.2f}s".format(time() - now))


if __name__ == "__main__":
    main()
