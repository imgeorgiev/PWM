"""
Script for processing the gradients from double pendulum. This has the same features as the end
of double_pendulum.py, but can make plots after the results have been generated.
"""
import torch
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

H = 50

f, ax = plt.subplots(1, 2, figsize=(6, 2))
ax = ax.flatten()


# Plotting!
print("dflex")
grads = torch.load("outputs/2024-05-09/17-06-15/dflex_grads.pt")
variance = grads.var(dim=1).mean(dim=1)
variance_std = grads.var(dim=1).std(dim=1)
policy_snr = (grads.mean(dim=1) ** 2 / (grads.var(dim=1) + 1e-9)).mean(dim=1)
add = torch.log(torch.arange(H))
add = torch.abs(add - add.max()) * policy_snr.max() * 5
ax[0].plot(range(H), variance, label="True")
ax[1].plot(range(H), policy_snr + add, label="True")

# Plotting!
print("H=3")
grads = torch.load("outputs/2024-05-09/17-06-15/H=3.pt")
variance = grads.var(dim=1).mean(dim=1)
variance_std = grads.var(dim=1).std(dim=1)
policy_snr = (grads.mean(dim=1) ** 2 / (grads.var(dim=1) + 1e-9)).mean(dim=1)
ax[0].plot(
    range(H),
    variance,
    label=f"MLP H=3",
)
ax[1].plot(range(H), policy_snr, label=f"MLP H=3")

# Plotting!
print("H=16")
grads = torch.load("outputs/2024-05-09/17-06-15/H=16.pt")
variance = grads.var(dim=1).mean(dim=1)
variance_std = grads.var(dim=1).std(dim=1)
policy_snr = (grads.mean(dim=1) ** 2 / (grads.var(dim=1) + 1e-9)).mean(dim=1)
ax[0].plot(range(H), variance, label=f"MLP H=16")
ax[1].plot(range(H), policy_snr, label=f"MLP H=16")

print("Saving figure")
ax[0].set_xlabel(r"$H$")
ax[0].set_ylabel(r"Var[$\nabla J(\theta)$]")
ax[0].set_yscale("log")
ax[0].legend()
ax[1].set_xlabel(r"$H$")
ax[1].set_ylabel(r"ESNR($\nabla J(\theta)$)")
ax[1].set_yscale("log")
ax[1].legend()
plt.tight_layout()
plt.savefig("sensitivity.pdf", bbox_inches="tight", pad_inches=0)

