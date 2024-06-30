import torch
from torch.distributions.normal import Normal
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pwm.models.mlp import mlp, SimNorm
from torch.optim import Adam
import torch.nn as nn
import numpy as np

from IPython.core import ultratb
import sys

# For debugging
sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)

sns.set()
torch.manual_seed(0)

# gravity, height, width
g, h, w = -9.81, 0.0, 5


def f(x, v, th, a, t):
    ty = (-v * torch.cos(th) + (v**2 * torch.cos(th) ** 2 + a * w) ** 0.5) / a
    y = v * torch.sin(th) * ty + g / 2 * ty**2
    out = x + v * torch.cos(th) * t + 1 / 2 * a * t**2
    out = torch.where((h > y) & (ty < t), w, out)
    return out


# simulation variables
samples = 1000
xx = torch.linspace(-torch.pi, torch.pi, samples)
x, v, a, t = 0, 10, 1, 2
yy = -f(x, v, xx, a, t)
std = 0.1  # noise for policy
N = 5000  # data samples
epochs = 100  # for optimization
batch_size = 56
lr = 2e-3

# train simply MLP
torch.manual_seed(0)
model0 = mlp(1, [32, 32], 1, last_layer="linear", last_layer_kwargs={})
opt = Adam(model0.parameters(), lr=lr)
steps = samples // batch_size
print("Training...")
model0.train()
losses0 = []
with tqdm(range(epochs), unit="epoch", total=epochs) as tepoch:
    for epoch in tepoch:
        epoch_loss = 0
        for step in range(steps):
            idx = torch.randint(0, samples, (batch_size,))
            _xx = xx[idx].unsqueeze(1)
            _yy = yy[idx].unsqueeze(1)
            pred = model0(_xx)
            loss = torch.mean((pred - _yy) ** 2)
            model0.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            losses0.append(loss.item())
        epoch_loss /= steps
        tepoch.set_postfix(loss=epoch_loss)


# train TDMPC model
torch.manual_seed(0)
model = mlp(
    1,
    [32],
    32,
    last_layer="normedlinear",
    last_layer_kwargs={"act": SimNorm(8)},
)
losses1 = []
decoder = mlp(32, [], 1, last_layer="linear", last_layer_kwargs={})
opt = Adam([{"params": model.parameters()}, {"params": decoder.parameters()}], lr=lr)
print("Training...")
model.train()
with tqdm(range(epochs), unit="epoch", total=epochs) as tepoch:
    for epoch in tepoch:
        epoch_loss = 0
        for step in range(steps):
            idx = torch.randint(0, samples, (batch_size,))
            _xx = xx[idx].unsqueeze(1)
            _yy = yy[idx].unsqueeze(1)
            pred = decoder(model(_xx))
            loss = torch.mean((pred - _yy) ** 2)
            model.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            losses1.append(loss.item())
        epoch_loss /= steps
        tepoch.set_postfix(loss=epoch_loss)

model1 = lambda x: decoder(model(x))

fig, ax1 = plt.subplots(1, 1, figsize=(3, 2.6))

print("Plotting the problem landscape")
ax1.plot(xx, -f(x, v, xx, a, t), label=r"$J(\theta)$")
models = {0: "ReLU", 1: "SimNorm", 2: "Spectral MLP"}
predictions = {}
for i, m in enumerate([model0, model1]):
    est = m(xx.unsqueeze(1)).flatten()
    ax1.plot(xx, est.detach(), label=models[i])
    error = torch.mean((est - yy) ** 2).item() ** 0.5
    print(f"Model has {error:.3f} approx error")
    predictions[i] = est

opt_value = torch.min(yy).item()
plt.plot(xx[0], yy[0], "x", color="black")
ii = 328
plt.plot(xx[ii], yy[ii], "x", color="tab:blue")
print(f"Opt error GT {yy[ii]-opt_value}")
est = predictions[0]
argmin = torch.argmin(est[: len(xx) // 2])
plt.plot(xx[argmin], est.detach()[argmin], color="tab:orange", marker="x")
plt.plot(xx[ii], yy[ii], "x", color="tab:blue")
print(f"Opt error MLP {est[argmin]-opt_value}")
est = predictions[1]
argmin = torch.argmin(est)
plt.plot(xx[argmin], est.detach()[argmin], color="tab:green", marker="x")
print(f"Opt error MLP SymNorm {est[argmin]-opt_value}")

ax1.set_xlabel(r"$\theta$")
ax1.legend(loc="upper center", bbox_to_anchor=(0.5, 1.3))

plt.tight_layout()
plt.savefig("ball_wall.pdf", bbox_inches="tight", pad_inches=0)


fig, ax = plt.subplots(1, 1, figsize=(3, 2.2))
cutoff = 1000
ax.plot(np.array(losses0)[:cutoff], label="ReLU", color="tab:orange")
ax.plot(np.array(losses1)[:cutoff], label="SimNorm", color="tab:green")
ax.set_xlabel("Iteration")
ax.set_ylabel("Loss")
plt.savefig("ball_wall_losses.pdf", bbox_inches="tight", pad_inches=0)
