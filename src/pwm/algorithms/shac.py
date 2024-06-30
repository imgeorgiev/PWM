import os, time, copy
import wandb
from omegaconf import DictConfig
from hydra.utils import instantiate
from typing import Optional, List, Tuple
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from gym import Env
import tensordict
from tensordict import TensorDict
from pwm.utils.common import *
import pwm.utils.torch_utils as tu
from pwm.utils.running_mean_std import RunningMeanStd
from pwm.utils.dataset import CriticDataset
from pwm.utils.time_report import TimeReport
from pwm.utils.average_meter import AverageMeter
from pwm.models.model_utils import Ensemble

tensordict.set_lazy_legacy(False).set()


class SHAC:
    """
    Short Horizon Actor Critic (SHAC) algorithm based on the paper
    Xu et al. Accelerated Policy Learning with Parallel Differentiable Simulation
    https://arxiv.org/abs/2204.07137
    """

    def __init__(
        self,
        env: Env,
        actor_config: DictConfig,
        critic_config: DictConfig,
        horizon: int,  # horizon for short rollouts
        max_epochs: int,  # number of short rollouts to do (i.e. epochs)
        logdir: str,
        actor_grad_norm: Optional[float] = None,  # clip grad norms during training
        critic_grad_norm: Optional[float] = None,  # clip grad norms during training
        num_critics: int = 3,  # for critic ensembling
        actor_lr: float = 2e-3,
        critic_lr: float = 2e-3,
        betas: Tuple[float, float] = (0.7, 0.95),
        lr_schedule: str = "linear",
        gamma: float = 0.99,  # discount factor
        lam: float = 0.95,  # for TD(lambda)
        obs_rms: bool = False,  # running normalization of observations
        ret_rms: bool = False,  # running normalization of returns
        critic_iterations: int = 16,
        critic_batches: int = 4,
        critic_method: str = "td-lambda",
        save_interval: int = 500,  # how often to save policy
        device: str = "cuda",
        save_data: bool = False,
        log: bool = False,
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
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.observation_space.shape[0]
        self.num_actions = self.env.action_space.shape[0]
        self.device = torch.device(device)
        self.save_data = save_data
        if save_data:
            self.episode_data = []
            if env.early_termination:
                raise RuntimeError("Environment should not have early_termination=True")

        self.horizon = horizon
        self.max_epochs = max_epochs
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.lr_schedule = lr_schedule
        self.gamma = gamma
        self.lam = lam

        self.critic_method = critic_method
        self.critic_iterations = critic_iterations
        self.critic_batch_size = self.num_envs * self.horizon // critic_batches

        self.obs_rms = None
        if obs_rms:
            self.obs_rms = RunningMeanStd(shape=(self.num_obs,), device=self.device)

        self.ret_rms = None
        if ret_rms:
            self.ret_rms = RunningMeanStd(shape=(1,), device=self.device)

        self.env_name = self.env.__class__.__name__
        self.name = self.__class__.__name__ + "_" + self.env_name

        self.actor_grad_norm = actor_grad_norm
        self.critic_grad_norm = critic_grad_norm
        self.save_interval = save_interval

        self.log = log
        self.log_dir = logdir
        os.makedirs(self.log_dir, exist_ok=True)

        # Create actor and critic
        self.actor = instantiate(
            actor_config,
            obs_dim=self.num_obs,
            action_dim=self.num_actions,
        ).to(self.device)

        critics = [
            instantiate(
                critic_config,
                obs_dim=self.num_obs,
            ).to(self.device)
            for _ in range(num_critics)
        ]
        self.critic = Ensemble(critics)

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

        # replay buffer
        self.obs_buf = torch.zeros(
            (self.horizon, self.num_envs, self.num_obs),
            dtype=torch.float32,
            device=self.device,
        )
        self.rew_buf = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.done_mask = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.next_values = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.target_values = torch.zeros(
            (self.horizon, self.num_envs), dtype=torch.float32, device=self.device
        )
        self.ret = torch.zeros((self.num_envs), dtype=torch.float32, device=self.device)

        # counting variables
        self.iter_count = 0
        self.step_count = 0

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

    def compute_actor_loss(self, deterministic=False):
        rew_acc = torch.zeros(
            (self.horizon + 1, self.num_envs), dtype=torch.float32, device=self.device
        )
        gamma = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        next_values = torch.zeros(
            (self.horizon + 1, self.num_envs), dtype=torch.float32, device=self.device
        )

        actor_loss = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        primal = None

        # copy running mean and std so that we don't change during training
        if self.obs_rms is not None:
            obs_rms = copy.deepcopy(self.obs_rms)

        # initialize trajectory to cut off gradients between epochs
        try:
            obs = self.env.reset(grads=True)
        except:
            print_error(
                "Your environment should have a reset method that accepts grads=True"
            )
            raise AttributeError

        # update and normalize obs
        if self.obs_rms is not None:
            self.obs_rms.update(obs)
            obs = obs_rms.normalize(obs)

        # keeps track of the current length of the rollout
        rollout_len = torch.zeros((self.num_envs,), device=self.device)

        # Start short horizon rollout
        for i in range(self.horizon):
            # collect data for critic training
            with torch.no_grad():
                self.obs_buf[i] = obs.clone()

            # act in environment
            actions = self.actor(obs, deterministic=deterministic)
            obs, rew, done, info = self.env.step(torch.tanh(actions))
            term = info["termination"]
            trunc = info["truncation"]
            real_obs = info["obs_before_reset"]
            primal = info["primal"]

            if self.save_data:
                with torch.no_grad():
                    td = TensorDict(
                        dict(
                            obs=real_obs.clone().unsqueeze(0),
                            action=actions.clone().unsqueeze(0),
                            reward=rew.clone().unsqueeze(0),
                        ),
                        (1,),
                    )
                    self.per_episode_data.append(td)

                    if done.all():
                        print("Episode terminated and dumping data")
                        data = TensorDict(
                            torch.cat(self.per_episode_data),
                            batch_size=(self.env.episode_length + 1, self.num_envs),
                        ).permute(1, 0)
                        self.episode_data.append(data)

                        # now need to reset per_episode_data
                        self.per_episode_data = []

                        # save data with nan action and rewards
                        a = torch.full_like(
                            torch.zeros(1, self.num_envs, self.num_actions), torch.nan
                        ).to(self.device)
                        r = torch.full_like(
                            torch.zeros(1, self.num_envs), torch.nan
                        ).to(self.device)
                        td = TensorDict(
                            dict(obs=obs.clone().unsqueeze(0), action=a, reward=r), (1,)
                        )
                        self.per_episode_data = [td]

            with torch.no_grad():
                raw_rew = rew.clone()

            # update and normalize obs
            if self.obs_rms is not None:
                self.obs_rms.update(obs)
                obs = obs_rms.normalize(obs)
                real_obs = obs_rms.normalize(real_obs)

            self.episode_length += 1
            rollout_len += 1

            # sanity check
            if (~torch.isfinite(real_obs)).sum() > 0:
                print_warning("Got inf obs")

            next_values[i + 1] = self.critic(real_obs).min(dim=0).values.squeeze()

            # handle terminated environments which stopped for some bad reason
            # since the reason is bad we set their value to 0
            term_env_ids = term.nonzero(as_tuple=False).squeeze(-1)
            for id in term_env_ids:
                next_values[i + 1, id] = 0.0

            # sanity check
            if (next_values > 1e6).sum() > 0 or (next_values < -1e6).sum() > 0:
                print_error("next value error")
                raise ValueError

            rew_acc[i + 1, :] = rew_acc[i, :] + gamma * rew

            done = term | trunc
            done_env_ids = done.nonzero(as_tuple=False).squeeze(-1)

            self.early_termination += torch.sum(term).item()
            self.episode_end += torch.sum(trunc).item()

            if i < self.horizon - 1:
                # first terminate all rollouts which are 'done'
                returns = (
                    -rew_acc[i + 1, done_env_ids]
                    - self.gamma
                    * gamma[done_env_ids]
                    * next_values[i + 1, done_env_ids]
                )
                actor_loss[done_env_ids] += returns
            else:
                # terminate all envs because we reached the end of our rollout
                returns = (
                    -rew_acc[i + 1, :] - self.gamma * gamma * next_values[i + 1, :]
                )
                actor_loss += returns

            # compute gamma for next step
            gamma = gamma * self.gamma

            # clear up gamma and rew_acc for done envs
            gamma[done_env_ids] = 1.0
            rew_acc[i + 1, done_env_ids] = 0.0

            # collect data for critic training
            with torch.no_grad():
                self.rew_buf[i] = rew.clone()
                if i < self.horizon - 1:
                    self.done_mask[i] = done.clone().to(torch.float32)
                else:
                    self.done_mask[i, :] = 1.0
                self.next_values[i] = next_values[i + 1].clone()

            # collect episode loss
            with torch.no_grad():
                # collect episode stats
                self.episode_loss -= raw_rew
                self.episode_discounted_loss -= self.episode_gamma * raw_rew
                self.episode_primal -= primal
                self.episode_gamma *= self.gamma

                # dump data from done episodes
                self.episode_loss_meter.update(self.episode_loss[done_env_ids])
                self.episode_discounted_loss_meter.update(
                    self.episode_discounted_loss[done_env_ids]
                )
                self.episode_primal_meter.update(self.episode_primal[done_env_ids])
                self.episode_length_meter.update(self.episode_length[done_env_ids])
                self.horizon_length_meter.update(rollout_len[done_env_ids])

                # reset trackers
                rollout_len[done_env_ids] = 0
                self.episode_loss[done_env_ids] = 0.0
                self.episode_discounted_loss[done_env_ids] = 0.0
                self.episode_primal[done_env_ids] = 0.0
                self.episode_length[done_env_ids] = 0
                self.episode_gamma[done_env_ids] = 1.0

        self.horizon_length_meter.update(rollout_len)

        if self.ret_rms is not None:
            self.ret_rms.update(actor_loss)
            actor_loss /= torch.sqrt(self.ret_rms.var + 1e-5)
        else:
            actor_loss /= self.horizon

        actor_loss = actor_loss.mean()

        self.step_count += self.horizon * self.num_envs

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

        games_cnt = 0
        while games_cnt < num_games:
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)

            actions = self.actor(obs, deterministic=deterministic)

            obs, rew, done, _ = self.env.step(torch.tanh(actions))

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
            Ai = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            Bi = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
            lam = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
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
        self.per_episode_data = []

        if self.save_data:
            with torch.no_grad():
                # save data with nan action and rewards
                act = torch.full_like(
                    torch.zeros(1, self.num_envs, self.num_actions), torch.nan
                ).to(self.device)
                rew = torch.full_like(torch.zeros(1, self.num_envs), torch.nan).to(
                    self.device
                )
                td = TensorDict(
                    dict(obs=obs.clone().unsqueeze(0), action=act, reward=rew), (1,)
                )
                self.per_episode_data.append(td)

        def actor_closure():
            self.actor_optimizer.zero_grad()

            self.time_report.start_timer("compute actor loss")

            self.time_report.start_timer("forward simulation")
            actor_loss = self.compute_actor_loss()
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
                raise ValueError

            self.time_report.end_timer("compute actor loss")

            return actor_loss

        # main training process
        for epoch in range(self.max_epochs):
            time_start_epoch = time.time()

            # learning rate schedule
            if self.lr_schedule == "linear":
                actor_lr = (1e-5 - self.actor_lr) * float(
                    epoch / self.max_epochs
                ) + self.actor_lr
                for param_group in self.actor_optimizer.param_groups:
                    param_group["lr"] = actor_lr
                lr = actor_lr
                critic_lr = (1e-5 - self.critic_lr) * float(
                    epoch / self.max_epochs
                ) + self.critic_lr
                for param_group in self.critic_optimizer.param_groups:
                    param_group["lr"] = critic_lr
            else:
                lr = self.actor_lr

            # train actor
            self.time_report.start_timer("actor training")
            actor_loss = self.actor_optimizer.step(actor_closure)
            self.time_report.end_timer("actor training")

            # train critic
            # prepare dataset
            self.time_report.start_timer("prepare critic dataset")
            with torch.no_grad():
                self.compute_target_values()
                dataset = CriticDataset(
                    self.critic_batch_size,
                    self.obs_buf,
                    self.target_values,
                )
            self.time_report.end_timer("prepare critic dataset")

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
                "episode_end": self.episode_end,
                "early_termination": self.early_termination,
            }
            metrics = filter_dict(metrics)
            if self.log:
                wandb.log(metrics, step=self.step_count)

            print(
                "[{:}/{:}]  R:{:.2f}  T:{:.1f}  H:{:.1f}  S:{:}  FPS:{:0.0f}  pi_loss:{:.2f}  pi_grad:{:.2f}/{:.2f}  v_loss:{:.2f}".format(
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
                )
            )

            if self.iter_count % self.save_interval == 0:
                name = self.name + f"_iter{self.iter_count}_rew{-mean_policy_loss:0.0f}"
                self.save(name)

        self.time_report.end_timer("algorithm")

        self.time_report.report()

        self.save("final_policy")

        if self.save_data:
            data = torch.cat(self.episode_data)
            eps = len(data)
            torch.save(data, f"{self.log_dir}/ep_data_{self.env_name}_ep{eps}.pt")

    def save(self, filename):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "obs_rms": self.obs_rms,
                "ret_rms": self.ret_rms,
            },
            os.path.join(self.log_dir, "{}.pt".format(filename)),
        )

    def load(self, path):
        print("Loading policy from", path)
        checkpoint = torch.load(path)
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.to(self.device)
        self.critic.load_state_dict(checkpoint["critic"])
        self.critic.to(self.device)
        self.obs_rms = (
            checkpoint["obs_rms"].to(self.device)
            if checkpoint["obs_rms"] is not None
            else None
        )
        self.ret_rms = (
            checkpoint["ret_rms"].to(self.device)
            if checkpoint["ret_rms"] is not None
            else None
        )
