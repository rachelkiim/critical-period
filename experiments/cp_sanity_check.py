"""
Mini sanity experiment for 0831-mockexp.md, Section 5.1.

Question: does latching structural plasticity (CP closure) at a churn-based
"maturity" point actually change weight/topology dynamics later, compared to
a network that keeps doing dynamic sparse training (SET) forever?

Setup:
  - Task: image classification on sklearn's `digits` dataset (8x8 grayscale,
    10 classes). No internet download needed.
  - Continual-learning split (required per spec Sec. 5, condition A - a
    static single task can't reveal any difference): Task A = digits 0-4,
    Task B = digits 5-9. Train on A first, then switch to B (A data no
    longer shown -> genuine distribution shift + forgetting pressure).
  - Model: MLP 64 -> 128 (sparse) -> 128 (sparse) -> 10 (dense head).
    Hidden layers are ~90% sparse and use SET-style dynamic sparse
    training (magnitude prune + random regrow) on a fixed cycle.
  - Two networks, IDENTICAL initialization (same seed, same initial mask):
      * CONTINUE : SET rewiring never stops (baseline / ablation, "no CP").
      * LATCH    : per-layer churn is tracked; once churn drops to a
                   fraction of its running peak (the "kink", not the floor -
                   spec Sec 3.2), that layer's topology is frozen forever.
                   Weights on surviving connections keep training normally.
  - Everything (weight init, mask init, batch order, rewire schedule) is
    seeded identically between the two networks so any divergence is
    attributable to the latch mechanism, not RNG noise.

Primary measurement (spec Sec. 5.1, measurement #1 - "run this before
anything else"): the epoch at which the two networks' topologies (masks)
start to diverge. Success = divergence appears AFTER Task B begins (i.e.
freezing genuinely blocks the network from recruiting new structure for the
new distribution). Failure = no divergence, or divergence that happens for
trivial/unrelated reasons before Task B even starts.

Secondary measurements: per-layer churn curves (does a churn "kink" even
exist to trigger on?), and downstream behavior (Task A retention / Task B
acquisition) for LATCH vs CONTINUE, which is what would eventually connect
to the P1-P4 evaluation axes in the full spec.
"""

import json
import numpy as np
from sklearn.datasets import load_digits

RNG_SEED = 0
rng_data = np.random.default_rng(RNG_SEED)

# ---------------------------------------------------------------------------
# Data: Task A = digits 0-4, Task B = digits 5-9
# ---------------------------------------------------------------------------
digits = load_digits()
X = digits.data.astype(np.float64) / 16.0  # 8x8 pixel images -> 64-dim, [0,1]
y = digits.target.astype(np.int64)

def make_split(classes, test_frac=0.2):
    idx = np.where(np.isin(y, classes))[0]
    rng_data.shuffle(idx)
    n_test = int(len(idx) * test_frac)
    return idx[n_test:], idx[:n_test]

taskA_classes = [0, 1, 2, 3, 4]
taskB_classes = [5, 6, 7, 8, 9]
trainA, testA = make_split(taskA_classes)
trainB, testB = make_split(taskB_classes)

N_IN, N_H, N_OUT = 64, 128, 10

# ---------------------------------------------------------------------------
# Sparse MLP with manual forward/backward (no autograd needed - it's tiny)
# ---------------------------------------------------------------------------
class SparseMLP:
    def __init__(self, seed, sparsity=0.90, lr=0.05):
        rng = np.random.default_rng(seed)
        self.lr = lr
        self.sparsity = sparsity

        def init_layer(n_in, n_out, sparse):
            scale = np.sqrt(2.0 / n_in)
            W = rng.normal(0, scale, size=(n_in, n_out))
            b = np.zeros(n_out)
            if sparse:
                mask = (rng.random((n_in, n_out)) > sparsity).astype(np.float64)
                W = W * mask
            else:
                mask = np.ones((n_in, n_out))
            return W, b, mask

        self.W1, self.b1, self.M1 = init_layer(N_IN, N_H, sparse=True)
        self.W2, self.b2, self.M2 = init_layer(N_H, N_H, sparse=True)
        self.W3, self.b3, self.M3 = init_layer(N_H, N_OUT, sparse=False)

        # bookkeeping for churn / maturity, per sparse layer index 0,1
        self.frozen = [False, False]
        self.freeze_epoch = [None, None]
        self.prev_regrown_pos = [None, None]   # positions regrown last cycle
        self.age = [np.zeros_like(self.M1, dtype=int), np.zeros_like(self.M2, dtype=int)]  # cycles since (re)grown
        self.churn_history = [[], []]      # raw per-cycle churn (diagnostic only now)
        self.churn_smooth = [[], []]       # windowed moving average of churn
        self.churn_ema = [None, None]
        self.churn_peak = [0.0, 0.0]
        self.mask_history = [[], []]  # snapshots (copy) at each rewire cycle

        # Maturity is now driven by an EXTERNAL signal (training loss slope),
        # not by the mask-churn bookkeeping itself -- churn plateaus to a
        # stable non-zero equilibrium under fixed-rate threshold pruning and
        # never gives a real kink to trigger on (see plot_results.py from the
        # first pass). Loss-plateau is the cheap proxy for spec Sec 3.3's
        # "representation has converged" signal. It also *drives* pruning
        # aggressiveness (eps_frac_effective): full exploration while loss is
        # still dropping fast, tapering off as the task saturates -- so churn
        # becomes a downstream *consequence* of real convergence instead of
        # an independent, never-converging bookkeeping quantity.
        self.loss_ema_fast = None
        self.loss_ema_slow = None
        self.loss_slow_at_last_cycle = None
        self.loss_slope_ema = None
        self.loss_slope_peak = 0.0
        self.maturity_streak = 0
        self.eps_frac_effective = None

    # ---- forward / backward ----
    def forward(self, X):
        Weff1 = self.W1 * self.M1
        Weff2 = self.W2 * self.M2
        z1 = X @ Weff1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ Weff2 + self.b2
        a2 = np.maximum(z2, 0)
        z3 = a2 @ self.W3 + self.b3
        z3s = z3 - z3.max(axis=1, keepdims=True)
        ez = np.exp(z3s)
        probs = ez / ez.sum(axis=1, keepdims=True)
        cache = (X, z1, a1, z2, a2, probs)
        return probs, cache

    def backward(self, cache, y_onehot):
        X, z1, a1, z2, a2, probs = cache
        n = X.shape[0]
        dz3 = (probs - y_onehot) / n
        dW3 = a2.T @ dz3
        db3 = dz3.sum(axis=0)

        da2 = dz3 @ self.W3.T
        dz2 = da2 * (z2 > 0)
        dW2 = (a1.T @ dz2) * self.M2  # masked-out weights get zero grad
        db2 = dz2.sum(axis=0)

        da1 = dz2 @ (self.W2 * self.M2).T
        dz1 = da1 * (z1 > 0)
        dW1 = (X.T @ dz1) * self.M1
        db1 = dz1.sum(axis=0)

        self.W3 -= self.lr * dW3; self.b3 -= self.lr * db3
        self.W2 -= self.lr * dW2; self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1; self.b1 -= self.lr * db1

    def train_step(self, Xb, yb_onehot):
        probs, cache = self.forward(Xb)
        self.backward(cache, yb_onehot)
        loss = float(-np.mean(np.sum(yb_onehot * np.log(probs + 1e-9), axis=1)))
        af, as_ = 0.15, 0.03
        self.loss_ema_fast = loss if self.loss_ema_fast is None else af * loss + (1 - af) * self.loss_ema_fast
        self.loss_ema_slow = loss if self.loss_ema_slow is None else as_ * loss + (1 - as_) * self.loss_ema_slow

    def predict(self, X):
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)

    # ---- SET-style dynamic sparse training ----
    # Pruning is THRESHOLD-based (not a fixed quota): a connection is prunable
    # once |W| falls below a fraction of the mean active-weight magnitude.
    # This is what makes churn an *endogenous* maturity signal (Sec 3.1/3.2):
    # as the layer's weight distribution matures and pulls away from zero,
    # fewer connections qualify, so raw turnover genuinely declines on its
    # own -- it isn't just tracking an externally hand-scheduled decay rate.
    # Newly grown connections get a multi-cycle grace period (else, since
    # they always start smaller than mature weights, they'd be re-pruned
    # ~100% of the time regardless of real convergence -- that was the bug
    # in the first version of this script). A 1-cycle grace still produced a
    # strong 2-cycle "burst" oscillation (everything grown at t survives the
    # grace period together and is then jointly re-evaluated at t+2, causing
    # periodic mass-eligibility spikes) -- GRACE_CYCLES=3 plus a 2-point
    # moving average on top removes that artifact.
    GRACE_CYCLES = 3

    def _rewire_layer(self, li, eps_frac, rewire_rng):
        W = [self.W1, self.W2][li]
        M = [self.M1, self.M2][li]
        age = self.age[li]

        active = np.argwhere(M > 0)
        n_active = len(active)
        mags = np.abs(W[active[:, 0], active[:, 1]])
        mean_mag = mags.mean() if n_active > 0 else 0.0
        threshold = eps_frac * mean_mag

        ages_active = age[active[:, 0], active[:, 1]]
        eligible_mask = ages_active >= self.GRACE_CYCLES
        below = mags < threshold
        prune_sel = eligible_mask & below
        pruned_pos = active[prune_sel]
        n_change = len(pruned_pos)
        for (i, j) in pruned_pos:
            M[i, j] = 0.0
            W[i, j] = 0.0
            age[i, j] = 0

        # regrow: same count, at random currently-inactive positions (keeps sparsity fixed)
        inactive = np.argwhere(M == 0)
        n_in, n_out = M.shape
        n_regrow = min(n_change, len(inactive))
        if n_regrow > 0:
            regrow_choice = rewire_rng.choice(len(inactive), size=n_regrow, replace=False)
            regrown_pos = inactive[regrow_choice]
        else:
            regrown_pos = np.empty((0, 2), dtype=int)
        scale = np.sqrt(2.0 / n_in)
        for (i, j) in regrown_pos:
            M[i, j] = 1.0
            W[i, j] = rewire_rng.normal(0, scale * 0.1)  # small init, "young" synapse
            age[i, j] = 0

        # everything still active (and not just-grown-this-cycle) ages by one cycle
        age[M > 0] += 1
        for (i, j) in regrown_pos:
            age[i, j] = 0  # newly grown connections start at age 0, not 1

        # churn = fraction of active connections replaced this cycle (mask turnover)
        churn = n_change / n_active if n_active > 0 else 0.0
        self.churn_history[li].append(churn)

        # Window=8 spans at least two full oscillation periods (empirically
        # ~4 cycles, driven by the synchronized grace-period cohort), so it
        # cancels the oscillation itself and exposes any real underlying
        # trend rather than aliasing on it (a window=2 average from the
        # first pass was still riding the oscillation, not cancelling it).
        hist = self.churn_history[li]
        smoothed = float(np.mean(hist[-8:]))
        self.churn_smooth[li].append(smoothed)

        alpha = 0.3
        if self.churn_ema[li] is None:
            self.churn_ema[li] = smoothed
        else:
            self.churn_ema[li] = alpha * smoothed + (1 - alpha) * self.churn_ema[li]
        self.churn_peak[li] = max(self.churn_peak[li], self.churn_ema[li])
        self.prev_regrown_pos[li] = regrown_pos
        self.mask_history[li].append(M.copy())

    def update_maturity_signal(self, eps_frac_base, eps_floor_frac=0.08, warmup_cycles=8,
                                kink_ratio=0.5, persist_needed=4):
        """External signal: how fast is training loss (EMA-smoothed) still
        falling? Sets self.eps_frac_effective (used by _rewire_layer this
        cycle -- full pruning aggressiveness while loss is still dropping
        fast, tapering toward a small floor as it saturates) and returns
        True the cycle maturity is confirmed (slope has stayed under
        kink_ratio * its own running peak for persist_needed consecutive
        cycles) -- i.e. "representation has converged, close the window"."""
        if self.loss_slow_at_last_cycle is None or self.loss_ema_slow is None:
            slope = 0.0
        else:
            slope = self.loss_slow_at_last_cycle - self.loss_ema_slow  # >0 while loss is falling
        self.loss_slow_at_last_cycle = self.loss_ema_slow

        slope = max(slope, 0.0)
        alpha = 0.3
        self.loss_slope_ema = slope if self.loss_slope_ema is None else alpha * slope + (1 - alpha) * self.loss_slope_ema
        self.loss_slope_peak = max(self.loss_slope_peak, self.loss_slope_ema)

        if self.loss_slope_peak > 0:
            ratio = np.clip(self.loss_slope_ema / self.loss_slope_peak, eps_floor_frac, 1.0)
        else:
            ratio = 1.0
        self.eps_frac_effective = eps_frac_base * ratio

        n_cycles = len(self.churn_history[0]) + 1  # about to run this cycle
        matured = False
        if n_cycles >= warmup_cycles and self.loss_slope_peak > 0:
            if self.loss_slope_ema <= kink_ratio * self.loss_slope_peak:
                self.maturity_streak += 1
            else:
                self.maturity_streak = 0
            matured = self.maturity_streak >= persist_needed
        return matured

    def rewire_cycle(self, eps_frac_base, rewire_rng, use_latch, **maturity_kwargs):
        """One SET rewiring cycle. eps_frac this cycle is set by the external
        loss-slope signal (update_maturity_signal), not a fixed constant. If
        use_latch, freeze BOTH sparse layers together (irreversible) once
        that signal confirms maturity (global signal -- no per-layer timing
        in this mini experiment, hierarchy/curriculum is out of scope here)."""
        matured = self.update_maturity_signal(eps_frac_base, **maturity_kwargs)
        for li in range(2):
            if use_latch and self.frozen[li]:
                self.mask_history[li].append(([self.M1, self.M2][li]).copy())
                continue
            self._rewire_layer(li, self.eps_frac_effective, rewire_rng)

        if use_latch and matured and not self.frozen[0]:
            n_cycles = len(self.churn_history[0])
            self.frozen[0] = self.frozen[1] = True
            self.freeze_epoch[0] = self.freeze_epoch[1] = n_cycles


def one_hot(y, n_classes=10):
    out = np.zeros((len(y), n_classes))
    out[np.arange(len(y)), y] = 1.0
    return out


def accuracy(model, X, y):
    if len(y) == 0:
        return float("nan")
    pred = model.predict(X)
    return float((pred == y).mean())


# ---------------------------------------------------------------------------
# Training loop: identical schedule for both networks, seeded identically.
# ---------------------------------------------------------------------------
EPOCHS_A = 80
EPOCHS_B = 40
BATCH = 32
REWIRE_EVERY = 2      # rewiring cycle every 2 epochs
EPS_FRAC = 0.35        # prune connections with |W| < EPS_FRAC * mean(|active W|) (endogenous, not a fixed quota)

# The endogenous churn-kink auto-trigger (Sec 3.2) is tested as-is below, and
# may legitimately never fire (that result is reported, not hidden). To still
# get a decisive answer to the actual question this sanity check is about
# ("does freezing topology change anything"), also force a freeze at a fixed
# epoch safely inside Task A if the auto-trigger hasn't fired by then -- this
# mirrors the spec's own fallback ("실패 시 threshold를 당기는 루프").
MANUAL_FREEZE_EPOCH = 69  # 0-indexed; well before the Task A/B switch at epoch 80

net_continue = SparseMLP(seed=42)
net_latch = SparseMLP(seed=42)  # identical init

log = {
    "epoch": [], "phase": [],
    "accA_continue": [], "accB_continue": [],
    "accA_latch": [], "accB_latch": [],
    "mask_divergence_l1": [], "mask_divergence_l2": [],
    "churn_continue_l1": [], "churn_continue_l2": [],
    "churn_latch_l1": [], "churn_latch_l2": [],
    "frozen_l1": [], "frozen_l2": [],
}

train_rng = np.random.default_rng(123)
rewire_rng_continue = np.random.default_rng(999)
rewire_rng_latch = np.random.default_rng(999)  # same seed -> same regrow draws while both active

freeze_snapshot = [None, None]  # per layer: mask + CONTINUE's weights, captured the instant LATCH froze

total_epochs = EPOCHS_A + EPOCHS_B
cycle_idx = 0
for epoch in range(total_epochs):
    phase = "A" if epoch < EPOCHS_A else "B"
    if phase == "A":
        train_idx, Xtr, ytr = trainA, X[trainA], y[trainA]
    else:
        train_idx, Xtr, ytr = trainB, X[trainB], y[trainB]

    order = train_rng.permutation(len(train_idx))
    for start in range(0, len(order), BATCH):
        bidx = order[start:start + BATCH]
        Xb = Xtr[bidx]
        yb = one_hot(ytr[bidx])
        net_continue.train_step(Xb, yb)
        net_latch.train_step(Xb, yb)

    if (epoch + 1) % REWIRE_EVERY == 0:
        cycle_idx += 1
        prev_frozen = list(net_latch.frozen)
        net_continue.rewire_cycle(EPS_FRAC, rewire_rng_continue, use_latch=False)
        net_latch.rewire_cycle(EPS_FRAC, rewire_rng_latch, use_latch=True)

        for li in range(2):
            if net_latch.frozen[li] and not prev_frozen[li] and freeze_snapshot[li] is None:
                freeze_snapshot[li] = {
                    "epoch": epoch, "trigger": "auto-churn-kink",
                    "mask": ([net_latch.M1, net_latch.M2][li]).copy(),
                    "Wc": ([net_continue.W1, net_continue.W2][li]).copy(),
                }

        if epoch == MANUAL_FREEZE_EPOCH:
            for li in range(2):
                if not net_latch.frozen[li]:
                    net_latch.frozen[li] = True
                    net_latch.freeze_epoch[li] = cycle_idx
                    freeze_snapshot[li] = {
                        "epoch": epoch, "trigger": "manual-fallback",
                        "mask": ([net_latch.M1, net_latch.M2][li]).copy(),
                        "Wc": ([net_continue.W1, net_continue.W2][li]).copy(),
                    }

        div_l1 = float(np.mean(net_continue.M1 != net_latch.M1))
        div_l2 = float(np.mean(net_continue.M2 != net_latch.M2))
    else:
        div_l1 = log["mask_divergence_l1"][-1] if log["mask_divergence_l1"] else 0.0
        div_l2 = log["mask_divergence_l2"][-1] if log["mask_divergence_l2"] else 0.0

    log["epoch"].append(epoch)
    log["phase"].append(phase)
    log["accA_continue"].append(accuracy(net_continue, X[testA], y[testA]))
    log["accB_continue"].append(accuracy(net_continue, X[testB], y[testB]))
    log["accA_latch"].append(accuracy(net_latch, X[testA], y[testA]))
    log["accB_latch"].append(accuracy(net_latch, X[testB], y[testB]))
    log["mask_divergence_l1"].append(div_l1)
    log["mask_divergence_l2"].append(div_l2)
    log["churn_continue_l1"].append(net_continue.churn_smooth[0][-1] if net_continue.churn_smooth[0] else None)
    log["churn_continue_l2"].append(net_continue.churn_smooth[1][-1] if net_continue.churn_smooth[1] else None)
    log["churn_latch_l1"].append(net_latch.churn_smooth[0][-1] if net_latch.churn_smooth[0] else None)
    log["churn_latch_l2"].append(net_latch.churn_smooth[1][-1] if net_latch.churn_smooth[1] else None)
    log["frozen_l1"].append(net_latch.frozen[0])
    log["frozen_l2"].append(net_latch.frozen[1])

# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
print("=" * 78)
print("CP-latch sanity check (spec Sec 5.1) -- digits dataset, Task A(0-4) -> Task B(5-9)")
print("=" * 78)
print(f"Task A train/test: {len(trainA)}/{len(testA)}   Task B train/test: {len(trainB)}/{len(testB)}")
print(f"Task switch at epoch {EPOCHS_A} (0-indexed). Total epochs: {total_epochs}. Rewire every {REWIRE_EVERY} epochs.\n")

for li, name in [(0, "Layer1 (64->128)"), (1, "Layer2 (128->128)")]:
    snap = freeze_snapshot[li]
    if snap is None:
        print(f"[LATCH] {name}: NEVER FROZE within training window.")
    else:
        when = "Task A (pre-switch)" if snap["epoch"] < EPOCHS_A else "Task B (post-switch)"
        print(f"[LATCH] {name}: froze at epoch {snap['epoch']} via {snap['trigger']} -> {when}")

print()

# Find first cycle where divergence becomes non-trivial (> 1% of connections differ)
def first_divergence_epoch(div_series, threshold=0.01):
    for e, d in zip(log["epoch"], div_series):
        if d > threshold:
            return e
    return None

fd1 = first_divergence_epoch(log["mask_divergence_l1"])
fd2 = first_divergence_epoch(log["mask_divergence_l2"])
print(f"First epoch mask divergence > 1% -- Layer1: {fd1}, Layer2: {fd2}  (Task B starts at epoch {EPOCHS_A})")
for e, l, name in [(fd1, 0, "Layer1"), (fd2, 1, "Layer2")]:
    if e is None:
        print(f"  {name}: NO divergence ever -> SANITY CHECK FAILED for this layer (latch had no effect, or never froze).")
    elif e >= EPOCHS_A:
        print(f"  {name}: divergence appears AFTER Task B onset -> consistent with the mechanism working as designed.")
    else:
        print(f"  {name}: divergence appears BEFORE Task B onset -> likely just reflects normal SET noise pre-switch, not a CP effect by itself.")

print()
print(f"Final epoch (E={total_epochs-1}) accuracies:")
print(f"  CONTINUE (baseline, no CP): TaskA={log['accA_continue'][-1]:.3f}  TaskB={log['accB_continue'][-1]:.3f}")
print(f"  LATCH    (CP closure)    : TaskA={log['accA_latch'][-1]:.3f}  TaskB={log['accB_latch'][-1]:.3f}")
print(f"  Task A retention delta (LATCH - CONTINUE): {log['accA_latch'][-1]-log['accA_continue'][-1]:+.3f}")
print(f"  Task B acquisition delta (LATCH - CONTINUE): {log['accB_latch'][-1]-log['accB_continue'][-1]:+.3f}")

print()
print("-" * 78)
print("Churn resurgence check: does CONTINUE's structural search specifically")
print("reopen at the Task B switch, or just keep decaying like it was already?")
print("-" * 78)
def window_slope(series, lo, hi):
    vals = [v for e, v in zip(log["epoch"], series) if lo <= e < hi]
    return (vals[-1] - vals[0]) / max(1, len(vals)) if len(vals) > 1 else float("nan")

for li, name, key in [(0, "Layer1", "churn_continue_l1"), (1, "Layer2", "churn_continue_l2")]:
    pre = window_slope(log[key], EPOCHS_A - 10, EPOCHS_A)
    post = window_slope(log[key], EPOCHS_A, EPOCHS_A + 10)
    pre_val = log[key][EPOCHS_A - 1]
    post_val = log[key][EPOCHS_A + 9]
    direction = "RESURGES" if post_val > pre_val * 1.5 else "no clear resurgence"
    print(f"  {name}: churn {pre_val:.4f} (end of Task A, quiet) -> {post_val:.4f} (10 epochs into Task B) -> {direction}")

print()
print("-" * 78)
print("New-structure contribution check (spec Sec 5.1 measurement #2/#3 proxy):")
print("Of CONTINUE's connections active at the end of training, how many/much")
print("weight now sits on positions that DID NOT EXIST at the moment LATCH froze?")
print("LATCH cannot have any weight there by construction -- this is the concrete")
print("thing the CP mechanism is claimed to take away.")
print("-" * 78)
for li, name in [(0, "Layer1 (64->128)"), (1, "Layer2 (128->128)")]:
    snap = freeze_snapshot[li]
    if snap is None:
        print(f"  {name}: n/a (never froze)")
        continue
    M_frozen = snap["mask"]
    Wc_at_freeze = snap["Wc"]
    W_now = [net_continue.W1, net_continue.W2][li]
    M_now = [net_continue.M1, net_continue.M2][li]

    new_pos = (M_now == 1) & (M_frozen == 0)
    old_pos = (M_now == 1) & (M_frozen == 1)
    n_new, n_old = int(new_pos.sum()), int(old_pos.sum())

    mass_new = float(np.abs(W_now[new_pos]).sum())
    mass_old_now = float(np.abs(W_now[old_pos]).sum())
    mass_old_delta = float(np.abs(W_now[old_pos] - Wc_at_freeze[old_pos]).sum())
    frac_mass_new = mass_new / (mass_new + mass_old_now) if (mass_new + mass_old_now) > 0 else float("nan")

    print(f"  {name}: {n_new} connections active now did not exist at freeze time "
          f"({n_new/(n_new+n_old)*100:.1f}% of currently-active connections).")
    print(f"    -> they carry {frac_mass_new*100:.1f}% of this layer's total active |W| mass "
          f"(vs. {mass_old_delta:.2f} summed |Delta W| accumulated on the {n_old} shared/old connections since freeze).")

with open("experiments/results.json", "w") as f:
    json.dump(log, f, indent=1)
print("\nFull per-epoch log written to experiments/results.json")
