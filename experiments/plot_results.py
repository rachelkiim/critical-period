import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open("experiments/results.json"))
epoch = d["epoch"]
switch = next(e for e, p in zip(epoch, d["phase"]) if p == "B")
freeze_epoch = next(e for e, f in zip(epoch, d["frozen_l1"]) if f)

fig, axes = plt.subplots(3, 1, figsize=(8, 10), sharex=True)

ax = axes[0]
ax.plot(epoch, d["accA_continue"], label="Task A acc, CONTINUE", color="tab:blue", linestyle="--")
ax.plot(epoch, d["accA_latch"], label="Task A acc, LATCH", color="tab:blue")
ax.plot(epoch, d["accB_continue"], label="Task B acc, CONTINUE", color="tab:orange", linestyle="--")
ax.plot(epoch, d["accB_latch"], label="Task B acc, LATCH", color="tab:orange")
ax.axvline(switch, color="gray", linestyle=":", label="Task A -> B switch")
ax.axvline(freeze_epoch, color="green", linestyle=":", label=f"auto latch (epoch {freeze_epoch})")
ax.set_ylabel("test accuracy")
ax.legend(fontsize=7, loc="center left")
ax.set_title("Task accuracy: CONTINUE (SET forever) vs LATCH (CP closure)")

ax = axes[1]
ax.plot(epoch, d["churn_continue_l1"], label="churn L1, CONTINUE", color="tab:blue", linestyle="--")
ax.plot(epoch, d["churn_continue_l2"], label="churn L2, CONTINUE", color="tab:red", linestyle="--")
ax.plot(epoch, d["churn_latch_l1"], label="churn L1, LATCH (until frozen)", color="tab:blue")
ax.plot(epoch, d["churn_latch_l2"], label="churn L2, LATCH (until frozen)", color="tab:red")
ax.axvline(switch, color="gray", linestyle=":")
ax.axvline(freeze_epoch, color="green", linestyle=":")
ax.set_ylabel("smoothed churn (mask turnover)")
ax.legend(fontsize=7, loc="upper right")
ax.set_title("Churn now genuinely kinks -- driven by the loss-slope signal, not raw bookkeeping")

ax = axes[2]
ax.plot(epoch, d["mask_divergence_l1"], label="mask divergence L1 (CONTINUE vs LATCH)", color="tab:purple")
ax.plot(epoch, d["mask_divergence_l2"], label="mask divergence L2 (CONTINUE vs LATCH)", color="tab:brown")
ax.axvline(switch, color="gray", linestyle=":")
ax.axvline(freeze_epoch, color="green", linestyle=":")
ax.set_ylabel("fraction of connections differing")
ax.set_xlabel("epoch")
ax.legend(fontsize=7, loc="upper left")
ax.set_title("Topology divergence: CONTINUE vs LATCH")

fig.tight_layout()
fig.savefig("experiments/results.png", dpi=130)
print("saved experiments/results.png")
