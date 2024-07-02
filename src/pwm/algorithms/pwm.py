import os, time
import wandb
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Optional, List, Tuple
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from gym import Env
import tensordict
from tensordict import TensorDict
import torch.nn.functional as F
from collections import OrderedDict


from pwm.utils.common import *
import pwm.utils.torch_utils as tu
from pwm.utils.running_mean_std import RunningMeanStd
from pwm.utils.dataset import CriticDataset
from pwm.utils.time_report import TimeReport
from pwm.utils.average_meter import AverageMeter
from pwm.models.model_utils import Ensemble
from pwm.utils.buffer import Buffer

tensordict.set_lazy_legacy(False).set()


class PWM:
    """
    Policy learning with large World Models
    """

    def __init__(
        self,
        env: Env,
        actor_config: DictConfig,
        critic_config: DictConfig,
        world_model_config: DictConfig,
        horizon: int,  # horizon for short rollouts
        max_epochs: int,  # number of short rollouts to do (i.e. epochs)
        logdir: str,
        latent_dim: int,
        obs_dim: int,
        act_dim: int,
        actor_grad_norm: Optional[float] = None,  # clip grad norms during training
        critic_grad_norm: Optional[float] = None,  # clip grad norms during training
        num_critics: int = 3,  # for critic ensembling
        actor_lr: float = 2e-3,
        critic_lr: float = 2e-3,
        model_lr: float = 2e-3,
        betas: Tuple[float, float] = (0.7, 0.95),
        lr_schedule: str = "linear",
        gamma: float = 0.99,  # discount factor
        lam: float = 0.95,  # for TD(lambda)
        obs_rms: bool = False,  # running normalization of observations
        rew_rms: bool = False,
        ret_rms: bool = False,  # running normalization of returns
        critic_iterations: int = 16,
        critic_batches: int = 4,
        critic_method: str = "td-lambda",
        wm_batch_size: int = 256,
        wm_iterations: int = 8,
        wm_grad_norm: float = 20.0,
        wm_buffer_size: int = 1_000_000,
        save_interval: int = 500,  # how often to save policy
        device: str = "cuda",
        save_data: bool = False,
        log: bool = False,
        detach: bool = False,
    ):
        # sanity check parameters
        assert horizon > 0
        assert max_epochs >= 0
        assert actor_lr >= 0
        assert critic_lr >= 0
        assert lr_schedule in ["linear", "constant"]
        assert 0 < gamma <= 1
        assert 0 < lam <= 1
        assert critic_iterations > 0
        assert critic_batches > 0
        assert critic_method in ["one-step", "td-lambda"]
        assert save_interval > 0

        self.env = env
        if env is not None:
            self.num_envs = self.env.num_envs
            self.num_obs = self.env.observation_space.shape[0]
            self.num_actions = self.env.action_space.shape[0]
            self.episode_data = [None] * self.env.num_envs
        else:
            self.num_obs = obs_dim
            self.num_actions = act_dim
        self.latent_dim = latent_dim
        self.device = torch.device(device)
        self.save_data = save_data

        self.horizon = horizon
        self.max_epochs = max_epochs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.model_lr = model_lr
        self.lr_schedule = lr_schedule
        self.gamma = gamma
        self.lam = lam
        self.detach = detach
        self.critic_batches = critic_batches

        self.critic_method = critic_method
        self.critic_iterations = critic_iterations
        # self.critic_batch_size = self.num_envs * self.horizon // critic_batches
        self.wm_iterations = wm_iterations
        self.wm_batch_size = wm_batch_size
        self.wm_grad_norm = wm_grad_norm
        self.wm_bootstrapped = False

        self.obs_rms = None
        if obs_rms:
            self.obs_rms = RunningMeanStd(shape=(self.num_obs,), device=self.device)

        self.rew_rms = None
        if rew_rms:
            self.rew_rms = RunningMeanStd(shape=(1,), device=self.device)

        self.ret_rms = None
        if ret_rms:
            self.ret_rms = RunningMeanStd(shape=(1,), device=self.device)

        # Buffer contains un-normalized data
        self.buffer = Buffer(
            buffer_size=wm_buffer_size,
            batch_size=self.wm_batch_size,
            horizon=self.horizon,
            device=device,
        )

        self.actor_grad_norm = actor_grad_norm
        self.critic_grad_norm = critic_grad_norm
        self.save_interval = save_interval

        self.log = log
        self.log_dir = logdir
        os.makedirs(self.log_dir, exist_ok=True)

        # Create actor and critic
        self.actor = instantiate(
            actor_config,
            obs_dim=latent_dim,
            action_dim=self.num_actions,
        ).to(self.device)

        critics = [
            instantiate(
                critic_config,
                obs_dim=latent_dim,
            ).to(self.device)
            for _ in range(num_critics)
        ]
        self.critic = Ensemble(critics)

        self.wm = instantiate(
            world_model_config,
            observation_dim=self.num_obs,
            action_dim=self.num_actions,
            latent_dim=self.latent_dim,
        ).to(self.device)

        # initialize optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            self.actor_lr,
            betas,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            self.critic_lr,
            betas,
        )

        self.wm_optimizer = torch.optim.Adam(
            [
                {"params": self.wm._encoder.parameters()},
                {"params": self.wm._dynamics.parameters()},
                {"params": self.wm._reward.parameters()},
                {"params": (self.wm._task_emb.parameters() if False else [])},
            ],
            lr=self.model_lr,
        )

        # counting variables
        self.iter_count = 0
        self.step_count = 0

        print(self.actor)
        print(self.critic)
        print(self.wm)

    def init_buffers(self):
        # replay buffer
        self.obs_buf = torch.zeros(
            (self.horizon, self.num_envs, self.latent_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.rew_buf = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.done_mask = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.term_buf = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.next_values = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.target_values = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.ret = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

        # loss variables
        self.episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_primal = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_length = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        self.best_policy_loss = torch.inf
        self.actor_grad_norm_before_clip = torch.inf
        self.actor_grad_norm_after_clip = torch.inf
        self.critic_grad_norm_val = torch.inf
        self.early_termination = 0
        self.episode_end = 0
        self.last_log_steps = 0

        # average meter
        self.episode_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_discounted_loss_meter = AverageMeter(1, 100).to(self.device)
        self.episode_primal_meter = AverageMeter(1, 100).to(self.device)
        self.episode_length_meter = AverageMeter(1, 100).to(self.device)
        self.horizon_length_meter = AverageMeter(1, 100).to(self.device)

        # timer
        self.time_report = TimeReport()

    @property
    def mean_horizon(self):
        return self.horizon_length_meter.get_mean()

    def compute_actor_loss(self, obs=None, task=None):

        if obs is None:
            # initialize trajectory to cut off gradients between epochs
            try:
                # Note this doesn't reset the env, just re-inits the gradients
                obs = self.env.reset(grads=True)
            except:
                print_error(
                    "Your environment should have a reset method that accepts grads=True"
                )
                raise AttributeError

        bsz, obs_dim = obs.shape

        self.obs_buf = torch.zeros(
            (self.horizon, bsz, self.latent_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self.rew_buf = torch.zeros(
            (self.horizon, bsz), dtype=torch.float32, device=self.device
        )
        self.done_mask = torch.zeros(
            (self.horizon, bsz), dtype=torch.float32, device=self.device
        )
        self.next_values = torch.zeros(
            (self.horizon, bsz), dtype=torch.float32, device=self.device
        )
        self.target_values = torch.zeros(
            (self.horizon, bsz), dtype=torch.float32, device=self.device
        )

        # NOTE: this horizon + 1 is not consistent with pwm_offline
        rew_acc = torch.zeros(
            (self.horizon + 1, bsz), dtype=torch.float32, device=self.device
        )
        gamma = torch.ones(bsz, dtype=torch.float32, device=self.device)
        next_values = torch.zeros(
            (self.horizon + 1, bsz), dtype=torch.float32, device=self.device
        )

        actor_loss = torch.zeros(bsz, dtype=torch.float32, device=self.device)
        primal = None

        # update and normalize obs
        if self.obs_rms:
            obs = self.obs_rms.normalize(obs)

        z = self.wm.encode(obs, task)

        # keeps track of the current length of the rollout
        rollout_len = torch.zeros((bsz,), device=self.device)

        # Start short horizon rollout
        for i in range(self.horizon):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = z.clone()

            # act in environment
            if self.detach:
                actions = self.actor(z.detach())
            else:
                actions = self.actor(z)

            actions = torch.tanh(actions)

            # NOTE term is not consistent here
            z, rew = self.wm.step(z, actions, task)
            rew = self.wm.almost_two_hot_inv(rew).squeeze()

            if torch.any(torch.isnan(rew)):
                print_warning("NaN reward from model!")
                rew = torch.nan_to_num(rew, 0.0, 0.0, 0.0)

            if self.env:
                obs, gt_rew, gt_done, info = self.env.step(actions)
                term = info["termination"]
                gt_term = info["termination"]
                gt_trunc = info["truncation"]
                real_obs = info["obs_before_reset"]
                primal = info["primal"]

                # sanity check; remove?
                if (~torch.isfinite(obs)).sum() > 0:
                    print_warning("Got inf obs from sim")
                    nan_idx = torch.any(~torch.isfinite(obs), dim=-1)
                    obs[nan_idx] = 0.0

                if (~torch.isfinite(real_obs)).sum() > 0:
                    print_warning("Got inf real_obs from sim")
                    nan_idx = torch.any(~torch.isfinite(real_obs), dim=-1)
                    real_obs[nan_idx] = 0.0

                nan_idx = torch.any(real_obs.abs() > 1e6, dim=-1)
                if nan_idx.sum() > 0:
                    print_warning("Got large real_obs from sim")
                    real_obs[nan_idx] = 0.0

                nan_idx = torch.any(obs.abs() > 1e6, dim=-1)
                if nan_idx.sum() > 0:
                    print_warning("Got large obs from sim")
                    obs[nan_idx] = 0.0

                # log data to buffer
                with torch.no_grad():

                    for j in range(self.num_envs):
                        td = TensorDict(
                            dict(
                                obs=real_obs[j].unsqueeze(0),
                                action=actions[j].unsqueeze(0),
                                reward=gt_rew[j][None],
                                term=gt_term[j][None],
                            ),
                            (1,),
                        )
                        self.episode_data[j].append(td)

                    gt_done_env_ids = gt_done.nonzero(as_tuple=False).squeeze(-1)
                    for j in gt_done_env_ids:
                        td = torch.cat(self.episode_data[j])
                        self.buffer.add(td)

                        # reinint data tracker with with nan action and rewards
                        a = torch.full_like(
                            torch.zeros(1, self.num_actions), torch.nan
                        ).to(self.device)
                        r = torch.full_like(
                            torch.zeros(
                                1,
                            ),
                            torch.nan,
                        ).to(self.device)
                        tt = torch.full_like(
                            torch.zeros(
                                1,
                            ),
                            torch.nan,
                            dtype=torch.bool,
                        ).to(self.device)
                        td = TensorDict(
                            dict(obs=obs[j].unsqueeze(0), action=a, reward=r, term=tt),
                            (1,),
                        )
                        self.episode_data[j] = [td]

                with torch.no_grad():
                    raw_rew = gt_rew.clone()

                # update and normalize obs
                if self.obs_rms:
                    obs = self.obs_rms.normalize(obs)

            # self.episode_length += 1
            rollout_len += 1

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            next_values[i + 1] = self.critic(z).min(dim=0).values.squeeze()

            if self.env is not None:
                # handle terminated environments which stopped for some bad reason
                # since the reason is bad we set their value to 0
                term_env_ids = term.nonzero(as_tuple=False).squeeze(-1)
                for id in term_env_ids:
                    next_values[i + 1, id] = 0.0

            # sanity check
            if (next_values > 1e6).sum() > 0 or (next_values < -1e6).sum() > 0:
                print_error("next value error")
                raise ValueError

            # for all done envs we reset observations and cut off gradients
            # Note this is important to do after critic next value compuataion!
            if self.env:
                done = gt_term | gt_trunc
                done_env_ids = gt_done.nonzero(as_tuple=False).squeeze(-1)
                gt_z = self.wm.encode(obs, task)
                z = torch.where(done[..., None], gt_z, z)

                self.early_termination += torch.sum(term).item()
                self.episode_end += torch.sum(gt_trunc).item()

                if i < self.horizon - 1:
                    # first terminate all rollouts which are 'done'
                    returns = (
                        -rew_acc[i + 1, done_env_ids]
                        - self.gamma
                        * gamma[done_env_ids]
                        * next_values[i + 1, done_env_ids]
                    )
                    actor_loss[done_env_ids] += returns

            # compute gamma for next step
            gamma = gamma * self.gamma

            if self.env is not None:
                # clear up gamma and rew_acc for done envs
                gamma[done_env_ids] = 1.0
                rew_acc[i + 1, done_env_ids] = 0.0

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.horizon - 1 and self.env is not None:
                    self.done_mask[i] = gt_done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                if self.env is not None:
                    self.term_buf[i] = gt_term.clone().to(torch.float32)
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            if self.env is not None:
                with torch.no_grad():
                    # collect episode stats
                    self.episode_loss -= raw_rew
                    self.episode_discounted_loss -= self.episode_gamma * raw_rew
                    self.episode_primal -= primal
                    self.episode_gamma *= self.gamma

                    # dump data from done episodes
                    self.episode_loss_meter.update(self.episode_loss[gt_done_env_ids])
                    self.episode_discounted_loss_meter.update(
                        self.episode_discounted_loss[gt_done_env_ids]
                    )
                    self.episode_primal_meter.update(
                        self.episode_primal[gt_done_env_ids]
                    )
                    self.episode_length_meter.update(
                        self.episode_length[gt_done_env_ids]
                    )
                    self.horizon_length_meter.update(rollout_len[gt_done_env_ids])

                    # reset trackers
                    rollout_len[gt_done_env_ids] = 0
                    self.episode_loss[gt_done_env_ids] = 0.0
                    self.episode_discounted_loss[gt_done_env_ids] = 0.0
                    self.episode_primal[gt_done_env_ids] = 0.0
                    self.episode_length[gt_done_env_ids] = 0
                    self.episode_gamma[gt_done_env_ids] = 1.0

        # terminate all envs because we reached the end of our rollout
        returns = -rew_acc[-1, :] - self.gamma * gamma * next_values[-1, :]
        actor_loss += returns

        # self.horizon_length_meter.update(rollout_len)

        if self.ret_rms is not None:
            self.ret_rms.update(actor_loss)
            actor_loss /= torch.sqrt(self.ret_rms.var + 1e-5)
        else:
            actor_loss /= self.horizon

        actor_loss = actor_loss.mean()

        self.step_count += self.horizon * bsz

        return actor_loss

    @torch.no_grad()
    def eval(self, num_games, deterministic=True):
        episode_length_his = []
        episode_loss_his = []
        episode_discounted_loss_his = []
        episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        episode_length = torch.zeros(self.num_envs, dtype=int)
        episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        obs = self.env.reset()
        if self.obs_rms is not None:
            obs = self.obs_rms.normalize(obs)
        z = self.wm.encode(obs, task=None)

        games_cnt = 0
        while games_cnt < num_games:

            actions = self.actor(z, deterministic=deterministic)
            actions = torch.tanh(actions)
            z, rew, trunc = self.wm.step(z, actions, task=None)

            _, _, done, _ = self.env.step(actions)

            episode_length += 1

            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            episode_loss -= rew
            episode_discounted_loss -= episode_gamma * rew
            episode_gamma *= self.gamma
            if len(done_env_ids) > 0:
                for done_env_id in done_env_ids:
                    print(
                        "loss = {:.2f}, len = {}".format(
                            episode_loss[done_env_id].item(),
                            episode_length[done_env_id],
                        )
                    )
                    episode_loss_his.append(episode_loss[done_env_id].item())
                    episode_discounted_loss_his.append(
                        episode_discounted_loss[done_env_id].item()
                    )
                    episode_length_his.append(episode_length[done_env_id].item())
                    episode_loss[done_env_id] = 0.0
                    episode_discounted_loss[done_env_id] = 0.0
                    episode_length[done_env_id] = 0
                    episode_gamma[done_env_id] = 1.0
                    games_cnt += 1

        mean_episode_length = torch.mean(torch.Tensor(episode_length_his))
        mean_policy_loss = torch.mean(torch.Tensor(episode_loss_his))
        mean_policy_discounted_loss = torch.mean(
            torch.Tensor(episode_discounted_loss_his)
        )

        return mean_policy_loss, mean_policy_discounted_loss, mean_episode_length

    @torch.no_grad()
    def compute_target_values(self):
        if self.critic_method == "one-step":
            self.target_values = self.rew_buf + self.gamma * self.next_values
        elif self.critic_method == "td-lambda":
            bsz = self.rew_buf.shape[1]
            Ai = torch.zeros(bsz, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(bsz, dtype=torch.float32, device=self.device)
            lam = torch.ones(bsz, dtype=torch.float32, device=self.device)
            for i in reversed(range(self.horizon)):
                lam = lam * self.lam * (1.0 - self.done_mask[i]) + self.done_mask[i]
                Ai = (1.0 - self.done_mask[i]) * (
                    self.lam * self.gamma * Ai
                    + self.gamma * self.next_values[i]
                    + (1.0 - lam) / (1.0 - self.lam) * self.rew_buf[i]
                )
                Bi = (
                    self.gamma
                    * (
                        self.next_values[i] * self.done_mask[i]
                        + Bi * (1.0 - self.done_mask[i])
                    )
                    + self.rew_buf[i]
                )
                self.target_values[i] = (1.0 - self.lam) * Ai + lam * Bi
        else:
            raise NotImplementedError

    def compute_critic_loss(self, batch_sample):
        predicted_values = self.critic(batch_sample["obs"]).squeeze(-2)
        target_values = batch_sample["target_values"]
        critic_loss = ((predicted_values - target_values) ** 2).mean()
        return critic_loss

    def train(self):

        self.init_buffers()

        # save initial policy for reproducibility
        self.save("init_policy")

        self.start_time = time.time()

        # add timers
        self.time_report.add_timer("algorithm")
        self.time_report.add_timer("compute actor loss")
        self.time_report.add_timer("forward simulation")
        self.time_report.add_timer("backward simulation")
        self.time_report.add_timer("prepare critic dataset")
        self.time_report.add_timer("actor training")
        self.time_report.add_timer("critic training")
        self.time_report.add_timer("world model training")
        self.time_report.start_timer("algorithm")

        # initializations
        obs = self.env.reset()
        self.episode_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_discounted_loss = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_primal = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )
        self.episode_length = torch.zeros(
            self.num_envs, dtype=torch.int, device=self.device
        )
        self.episode_gamma = torch.ones(
            self.num_envs, dtype=torch.float32, device=self.device
        )

        bsz = self.num_envs

        with torch.no_grad():
            # save data with nan action and rewards

            for id in range(self.num_envs):
                act = torch.full_like(torch.zeros(1, self.num_actions), torch.nan).to(
                    self.device
                )
                rew = torch.full_like(
                    torch.zeros(
                        1,
                    ),
                    torch.nan,
                ).to(self.device)
                term = torch.full_like(
                    torch.zeros(
                        1,
                    ),
                    torch.nan,
                    dtype=torch.bool,
                ).to(self.device)
                td = TensorDict(
                    dict(obs=obs[id].unsqueeze(0), action=act, reward=rew, term=term),
                    (1,),
                )
                self.episode_data[id] = [td]

        def actor_closure():
            self.actor_optimizer.zero_grad()

            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            actor_loss = self.compute_actor_loss()
            if torch.isnan(actor_loss):
                print_error("NaN actor loss")
            self.time_report.end_timer("forward simulation")

            self.time_report.start_timer("backward simulation")
            actor_loss.backward()
            self.time_report.end_timer("backward simulation")

            self.actor_grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
            self.actor_grad_norm_after_clip = clip_grad_norm_(
                self.actor.parameters(), self.actor_grad_norm
            )

            # sanity check
            if torch.isnan(self.actor_grad_norm_before_clip):
                print_error("NaN gradient")
                # ugly fix for simulation nan problem
                for params in self.actor.parameters():
                    params.grad.nan_to_num_(0.0, 0.0, 0.0)
                # raise ValueError

            self.time_report.end_timer("compute actor loss")

            return actor_loss

        # main training process
        for epoch in range(self.max_epochs):

            if self.buffer.num_eps == 0:
                with torch.no_grad():
                    self.compute_actor_loss()
                continue

            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == "linear":
                # actor learning rate
                actor_lr = (1e-5 - self.actor_lr) * float(
                    epoch / self.max_epochs
                ) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = actor_lr
                lr = actor_lr

                # critic learning rate
                critic_lr = (1e-5 - self.critic_lr) * float(
                    epoch / self.max_epochs
                ) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = critic_lr

                # world model learning rate
                model_lr = (1e-5 - self.model_lr) * float(
                    epoch / self.max_epochs
                ) + self.model_lr
                for param_group in self.wm_optimizer.param_groups:
                    param_group["lr"] = model_lr
            else:
                lr = self.actor_lr

            # train actor
            self.time_report.start_timer("actor training")
            actor_loss = self.actor_optimizer.step(actor_closure)
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            critic_batch_size = bsz * self.horizon // self.critic_batches
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(
                    critic_batch_size,
                    self.obs_buf,
                    self.target_values,
                )
            self.time_report.end_timer("prepare critic dataset")

            # critic training!
            self.time_report.start_timer("critic training")
            value_loss = 0.0
            for j in range(self.critic_iterations):
                total_critic_loss = 0.0
                batch_cnt = 0
                for i in range(len(dataset)):
                    batch_sample = dataset[i]
                    self.critic_optimizer.zero_grad()
                    training_critic_loss = self.compute_critic_loss(batch_sample)
                    training_critic_loss.backward()

                    # ugly fix for simulation nan problem
                    for params in self.critic.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)

                    critic_grad_norm = clip_grad_norm_(
                        self.critic.parameters(), self.critic_grad_norm
                    )
                    self.critic_optimizer.step()

                    total_critic_loss += training_critic_loss
                    batch_cnt += 1

                value_loss = total_critic_loss / batch_cnt
                print(
                    f"value iter {j + 1}/{self.critic_iterations}, loss = {value_loss:.2f}",
                    end="\r",
                )

            self.time_report.end_timer("critic training")

            self.time_report.start_timer("world model training")

            # world model training!
            tot_wm_loss = tot_dynamics_loss = tot_reward_loss = tot_term_loss = 0.0
            sample_rew_mean = sample_rew_var = 0.0
            sample_obs_mean = sample_obs_var = 0.0
            if self.wm_bootstrapped:
                iters = self.wm_iterations
            else:
                iters = self.env.episode_length
                print(f"training wm for {iters} iterations")
                self.wm_bootstrapped = True

            for i in range(0, iters):
                obs, act, rew = self.buffer.sample()

                if torch.any(torch.isnan(obs)):
                    print("WARN: NaN obs sampled!")
                    obs = torch.nan_to_num(obs)

                if torch.any(torch.isnan(rew)):
                    print("WARN: NaN reward sampled!")

                if self.obs_rms:
                    self.obs_rms.update(obs.reshape((-1, self.num_obs)))
                    obs = self.obs_rms.normalize(obs)

                if self.rew_rms:
                    self.rew_rms.update(rew.reshape((-1, 1)))
                    rew = self.rew_rms.normalize(rew)

                if torch.any(torch.isnan(rew)):
                    print("WARN: NaN reward post-processed!")

                sample_rew_mean += rew.mean().item()
                sample_rew_var += rew.var().item()
                sample_obs_mean += obs.mean(dim=0).mean().item()
                sample_obs_var += obs.var(dim=0).mean().item()

                self.wm_optimizer.zero_grad()
                loss, dyn_loss, rew_loss = self.compute_wm_loss(obs, act, rew)
                loss.backward()
                wm_grad_norm = clip_grad_norm_(self.wm.parameters(), self.wm_grad_norm)
                if torch.isnan(wm_grad_norm):
                    print_warning("world model NaN gradient")
                    for params in self.wm.parameters():
                        params.grad.nan_to_num_(0.0, 0.0, 0.0)
                self.wm_optimizer.step()
                tot_wm_loss += loss.item()
                tot_dynamics_loss += dyn_loss
                tot_reward_loss += rew_loss
                # tot_term_loss += term_loss
                print(f"wm iter {i+1}/{iters}, loss = {loss:.2f}", end="\r")

            # normalize for logging; TODO simplify
            tot_wm_loss /= iters
            tot_dynamics_loss /= iters
            tot_reward_loss /= iters
            tot_term_loss /= iters
            sample_rew_mean /= iters
            sample_rew_var /= iters
            sample_obs_mean /= iters
            sample_obs_var /= iters

            self.time_report.end_timer("world model training")

            self.iter_count += 1
            time_end_epoch = time.time()
            fps = self.horizon * self.num_envs / (time_end_epoch - time_start_epoch)
            mean_episode_length = self.episode_length_meter.get_mean()
            mean_policy_loss = self.episode_loss_meter.get_mean()
            mean_policy_discounted_loss = self.episode_discounted_loss_meter.get_mean()
            mean_episode_primal = self.episode_primal_meter.get_mean()
            ac_stddev = self.actor.get_logstd().exp().mean().detach().cpu().item()

            if mean_policy_loss < self.best_policy_loss:
                print_info("save best policy with loss {:.2f}".format(mean_policy_loss))
                self.save(f"best_policy")
                self.best_policy_loss = mean_policy_loss

            metrics = {
                "actor_lr": lr,
                "actor_loss": actor_loss,
                "value_loss": value_loss,
                "wm_loss": tot_wm_loss,
                "dynamics_loss": tot_dynamics_loss,
                "reward_loss": tot_reward_loss,
                "term_loss": tot_term_loss,
                "rollout_len": self.mean_horizon,
                "fps": fps,
                "policy_loss": mean_policy_loss,
                "rewards": -mean_policy_loss,
                "primal": -mean_episode_primal,
                "policy_discounted_loss": mean_policy_discounted_loss,
                "best_policy_loss": self.best_policy_loss,
                "episode_lengths": mean_episode_length,
                "actor_std": ac_stddev,
                "actor_grad_norm": self.actor_grad_norm_before_clip,
                "critic_grad_norm": critic_grad_norm,
                "wm_grad_norm": wm_grad_norm,
                "episode_end": self.episode_end,
                "early_termination": self.early_termination,
                "sample_rew_mean": sample_rew_mean,
                "sample_rew_var": sample_rew_var,
                "sample_obs_mean": sample_obs_mean,
                "sample_obs_var": sample_obs_var,
            }
            if self.rew_rms:
                metrics.update(
                    dict(
                        rew_rms_mean=self.rew_rms.mean.item(),
                        rew_rms_var=self.rew_rms.var.item(),
                    )
                )

            if self.ret_rms:
                metrics.update(
                    dict(
                        ret_rms_mean=self.ret_rms.mean.item(),
                        ret_rms_var=self.ret_rms.var.item(),
                    )
                )
            metrics = filter_dict(metrics)
            if self.log and (self.iter_count % 50 == 0):
                wandb.log(metrics, step=self.step_count)

            print(
                "[{:}/{:}]  R:{:.2f}  T:{:.1f}  H:{:.1f}  S:{:}  FPS:{:0.0f}  pi_loss:{:.2f}  pi_grad:{:.2f}/{:.2f}  v_loss:{:.2f}  wm_loss:{:.2f}  rew_loss:{:.2f}  dyn_loss:{:.2f}".format(
                    self.iter_count,
                    self.max_epochs,
                    -mean_policy_loss,
                    mean_episode_length,
                    self.mean_horizon,
                    self.step_count,
                    fps,
                    actor_loss,
                    self.actor_grad_norm_before_clip,
                    self.actor_grad_norm_after_clip,
                    value_loss,
                    tot_wm_loss,
                    tot_reward_loss,
                    tot_dynamics_loss,
                )
            )

            if self.iter_count % self.save_interval == 0:
                name = self.__class__.__name__
                name = name + f"_iter{self.iter_count}_rew{-mean_policy_loss:0.0f}"
                self.save(name)

        self.time_report.end_timer("algorithm")

        self.time_report.report()

        self.save("final_policy", buffer=True)

    def update(self, obs, act, rew, task, finetune_wm=False):

        L, bsz, obs_dim = obs.shape

        # train world model
        if finetune_wm:
            self.wm_optimizer.zero_grad()
            wm_loss, dyn_loss, rew_loss = self.compute_wm_loss(obs, act, rew, task)
            wm_loss.backward()
            wm_grad_norm = clip_grad_norm_(self.wm.parameters(), self.wm_grad_norm)
            self.wm_optimizer.step()

        # train actor
        self.actor_optimizer.zero_grad()

        # NOTE not sure about dimensionality below
        actor_loss = self.compute_actor_loss(obs[0], task)
        actor_loss.backward()

        self.actor_grad_norm_before_clip = tu.grad_norm(self.actor.parameters())
        self.actor_grad_norm_after_clip = clip_grad_norm_(
            self.actor.parameters(), self.actor_grad_norm
        )

        # sanity check
        if torch.isnan(self.actor_grad_norm_before_clip):
            print_error("NaN gradient")
            raise ValueError

        self.actor_optimizer.step()

        # prepare dataset
        critic_batch_size = bsz * self.horizon // self.critic_batches
        with torch.no_grad():
            self.compute_target_values()
            dataset = CriticDataset(
                critic_batch_size,
                self.obs_buf,
                self.target_values,
            )

        # critic training!
        value_loss = 0.0
        for j in range(self.critic_iterations):
            total_critic_loss = 0.0
            batch_cnt = 0
            for i in range(len(dataset)):
                batch_sample = dataset[i]
                self.critic_optimizer.zero_grad()
                training_critic_loss = self.compute_critic_loss(batch_sample)
                training_critic_loss.backward()

                # ugly fix for simulation nan problem
                for params in self.critic.parameters():
                    params.grad.nan_to_num_(0.0, 0.0, 0.0)

                critic_grad_norm = clip_grad_norm_(
                    self.critic.parameters(), self.critic_grad_norm
                )
                self.critic_optimizer.step()

                total_critic_loss += training_critic_loss
                batch_cnt += 1

            value_loss += total_critic_loss / batch_cnt

        value_loss /= self.critic_iterations

        ac_stddev = self.actor.get_logstd().exp().mean().detach().cpu().item()

        metrics = {
            "actor_loss": actor_loss.item(),
            "value_loss": value_loss.item(),
            "actor_grad_norm": self.actor_grad_norm_before_clip.item(),
            "critic_grad_norm": critic_grad_norm.item(),
        }
        if finetune_wm:
            metrics["wm_loss"] = wm_loss
            metrics["dynamics_loss"] = dyn_loss
            metrics["reward_loss"] = rew_loss
            metrics["wm_grad_norm"] = wm_grad_norm
        metrics = filter_dict(metrics)
        return metrics

    def save(self, filename, log_dir=None, buffer=False):
        log_dir = self.log_dir if log_dir is None else log_dir
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "world_model": self.wm.state_dict(),
                "obs_rms": self.obs_rms,
                "rew_rms": self.rew_rms,
                "ret_rms": self.ret_rms,
                "actor_opt": self.actor_optimizer.state_dict(),
                "critic_opt": self.critic_optimizer.state_dict(),
                "world_model_opt": self.wm_optimizer.state_dict(),
            },
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )
        if buffer:
            self.buffer.save(os.path.join(self.log_dir, "{}.buffer".format(filename)))

    def load(self, path, buffer=False):
        print("Loading policy from", path)
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.to(self.device)
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic.to(self.device)
        self.wm.load_state_dict(checkpoint["world_model"])
        self.wm.to(self.device)
        self.obs_rms = (
            checkpoint["obs_rms"].to(self.device)
            if checkpoint["obs_rms"] is not None
            else None
        )
        self.rew_rms = (
            checkpoint["rew_rms"].to(self.device)
            if checkpoint["rew_rms"] is not None
            else None
        )
        self.ret_rms = (
            checkpoint["ret_rms"].to(self.device)
            if checkpoint["ret_rms"] is not None
            else None
        )

        # need to also load last learning rates as they will be used to continue training
        self.actor_optimizer.load_state_dict(checkpoint["actor_opt"])
        self.actor_lr = checkpoint["actor_opt"]["param_groups"][0]["lr"]
        self.critic_optimizer.load_state_dict(checkpoint["critic_opt"])
        self.critic_lr = checkpoint["critic_opt"]["param_groups"][0]["lr"]
        self.wm_optimizer.load_state_dict(checkpoint["world_model_opt"])
        self.model_lr = checkpoint["world_model_opt"]["param_groups"][0]["lr"]

        if buffer:
            print("Loading buffer too")
            self.buffer.load(path.replace(".pt", ".buffer"))
            self.buffer._num_eps = 100  # placeholder to avoid initialization

    def load_wm(self, path):
        print("Loading world model from", path)
        checkpoint = torch.load(path)
        checkpoint = checkpoint["model"]
        new_odict = OrderedDict()
        for key, value in checkpoint.items():
            if "_pi" in key:
                pass
            elif "_Qs" in key:
                pass
            else:
                if "_encoder" in key:
                    key = key.replace("state.", "")
                new_odict[key] = value

        self.wm.load_state_dict(new_odict)

    def pretrain_wm(self, paths, num_iters, actually_train=True):
        if type(paths) != List:
            paths = [paths]
        for path in paths:
            print("loading", path)
            td = torch.load(path).to("cpu")

            # fetch stats for normalizing
            if self.obs_rms:
                obs = td["obs"]
                # NOTE: if this fails probably load wrong data
                obs = obs.reshape((-1, self.num_obs))
                obs = torch.nan_to_num(obs)
                self.obs_rms.update(obs)

            if self.rew_rms:
                self.rew_rms = self.rew_rms.to("cpu")
                rew = td["reward"]
                rew = rew.reshape((-1, 1))
                rew = torch.nan_to_num(rew)
                self.rew_rms.update(rew)
                self.rew_rms = self.rew_rms.to(self.device)

            self.buffer.add_batch(td)

        if not actually_train:
            return

        print(f"Pretraining world model for {num_iters} iters")
        log_freq = num_iters // 100
        save_at = num_iters // 5
        for i in range(0, num_iters):
            obs, act, rew = self.buffer.sample()
            if self.obs_rms:
                obs = self.obs_rms.normalize(obs)
            if self.rew_rms:
                rew = self.rew_rms.normalize(rew)
            self.wm_optimizer.zero_grad()
            loss, dyn_loss, rew_loss = self.compute_wm_loss(obs, act, rew)
            loss.backward()
            wm_grad_norm = clip_grad_norm_(self.wm.parameters(), self.wm_grad_norm)
            self.wm_optimizer.step()
            if i % log_freq == 0 and self.log:
                metrics = {
                    "pretrain/wm_loss": loss.item(),
                    "pretrain/wm_grad_norm": wm_grad_norm,
                    "pretrain/dynamics_loss": dyn_loss,
                    "pretrain/reward_loss": rew_loss,
                }
                wandb.log(metrics, step=i)
                print(
                    f"[{i}/{num_iters}]  L:{loss.item():.3f}  GN:{wm_grad_norm:.3f}  DL:{dyn_loss:.3f}  RL:{rew_loss:.3f}",
                )

            if i % save_at == 0:
                self.save(f"pretrained_{i}")
        self.wm_bootstrapped = True
        self.save("pretrained", buffer=True)

    def compute_wm_loss(self, obs, act, rew, task=None):
        horizon, batch_size, _ = obs.shape
        assert horizon == self.horizon + 1
        discount = (
            (self.gamma ** torch.arange(self.horizon))
            .view((self.horizon, 1, 1))
            .to(self.device)
        )

        # Compute targets
        with torch.no_grad():
            next_z = self.wm.encode(obs[1:], task)

        # Latent rollout
        zs = torch.empty(
            self.horizon + 1,
            batch_size,
            self.latent_dim,
            device=self.device,
        )

        z = self.wm.encode(obs[0], task)
        zs[0] = z

        dynamics_loss = 0.0
        for t in range(self.horizon):
            z = self.wm.next(z, act[t], task)
            dynamics_loss += F.mse_loss(z, next_z[t]) * self.gamma**t
            zs[t + 1] = z

        _zs = zs[:-1]
        rew_hat = self.wm.reward(_zs, act, task)
        reward_loss = (rew_hat - rew) ** 2 * discount
        reward_loss = reward_loss.mean()

        total_loss = dynamics_loss + reward_loss
        total_loss /= self.horizon
        return (
            total_loss,
            dynamics_loss / self.horizon,
            reward_loss.item() / self.horizon,
        )

    def act(self, obs, t0=False, deterministic=False, task=None):
        obs = torch.tensor(obs, dtype=torch.float32, device=self.device)[None]
        z = self.wm.encode(obs, task)
        a = self.actor(z, deterministic)
        return torch.tanh(a).cpu().detach().flatten()

    def update_lrs(self, epoch):
        # learning rate schedule
        if self.lr_schedule == "linear":
            # actor learning rate
            actor_lr = (1e-5 - self.actor_lr) * float(
                epoch / self.max_epochs
            ) + self.actor_lr
            for param_group in self.actor_optimizer.param_groups:
                param_group["lr"] = actor_lr

            # critic learning rate
            critic_lr = (1e-5 - self.critic_lr) * float(
                epoch / self.max_epochs
            ) + self.critic_lr
            for param_group in self.critic_optimizer.param_groups:
                param_group["lr"] = critic_lr

            # world model learning rate
            model_lr = (1e-5 - self.model_lr) * float(
                epoch / self.max_epochs
            ) + self.model_lr
            for param_group in self.wm_optimizer.param_groups:
                param_group["lr"] = model_lr

            return actor_lr, critic_lr, model_lr
        else:
            return self.actor_lr, self.critic_lr, self.model_lr
