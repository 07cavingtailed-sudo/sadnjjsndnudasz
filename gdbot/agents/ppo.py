"""Proximal Policy Optimisation (optional; needs ``torch``).

The gradient-based alternative to :mod:`gdbot.agents.cem`.  It learns from the
dense per-tick reward the simulator provides, which makes it far more
sample-efficient there -- and useless against the real game, which provides no
per-tick reward worth the name.  So PPO is a simulator pretraining tool.

The actor has exactly the shape of :class:`~gdbot.agents.policy.MLPPolicy`, and
:func:`to_numpy_policy` copies its weights across.  A PPO-trained policy can
therefore be used everywhere the numpy policy is -- including against the real
game on a machine without torch installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

try:  # pragma: no cover - import guard
    import torch
    from torch import nn
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "the PPO agent needs torch: pip install 'gdbot[torch]'"
    ) from exc

from gdbot.agents.policy import MLPPolicy
from gdbot.env.base import AttemptEnv
from gdbot.types import Outcome


@dataclass
class PPOConfig:
    updates: int = 40
    steps_per_update: int = 4096
    epochs: int = 4
    minibatch: int = 512
    gamma: float = 0.995
    lam: float = 0.95
    clip: float = 0.2
    lr: float = 3e-4
    entropy: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    #: Fraction of episodes started at a random point already reached, so the
    #: policy practises the whole level rather than only its opening.
    practice_frac: float = 0.3
    max_episode_ticks: int = 60 * 240
    #: Hold each decision for this many ticks.  Per-tick random actions are a
    #: poor way to explore Geometry Dash -- what matters is *when* a press
    #: happens, and independent coin flips every 16 ms mostly produce noise.
    #: Deciding every few ticks (Atari-style frame skip) shortens the horizon
    #: and makes exploration coherent.
    action_repeat: int = 4
    #: Initial logit gap towards releasing.  With a 50/50 start PPO settles into
    #: "hold forever", which clears evenly spaced spikes and nothing else; a
    #: start that mostly releases has to discover individual presses.
    release_bias: float = 2.0
    seed: int = 0


def _mlp(sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(zip(sizes, sizes[1:])):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class ActorCritic(nn.Module):
    def __init__(self, n_in: int, hidden: Sequence[int], release_bias: float = 0.0) -> None:
        super().__init__()
        self.hidden = tuple(int(h) for h in hidden)
        self.actor = _mlp([n_in, *self.hidden, 2])
        self.critic = _mlp([n_in, 64, 64, 1])
        with torch.no_grad():
            last = self.actor[-1]
            assert isinstance(last, nn.Linear)
            last.bias.copy_(torch.tensor([release_bias / 2.0, -release_bias / 2.0]))

    def forward(self, x: torch.Tensor) -> tuple[torch.distributions.Categorical, torch.Tensor]:
        return torch.distributions.Categorical(logits=self.actor(x)), self.critic(x).squeeze(-1)


def to_numpy_policy(model: ActorCritic, n_in: int) -> MLPPolicy:
    """Copy the actor's weights into an equivalent numpy policy."""
    policy = MLPPolicy(n_in=n_in, hidden=model.hidden)
    chunks: list[np.ndarray] = []
    for layer in model.actor:
        if isinstance(layer, nn.Linear):
            # torch stores (out, in); MLPPolicy multiplies x @ W with W as (in, out)
            chunks.append(layer.weight.detach().cpu().numpy().T.reshape(-1))
            chunks.append(layer.bias.detach().cpu().numpy().reshape(-1))
    policy.set_params(np.concatenate(chunks).astype(np.float32))
    return policy


def train_ppo(
    env: AttemptEnv,
    cfg: PPOConfig | None = None,
    *,
    hidden: Sequence[int] = (32,),
    on_update: Callable[[int, dict], None] | None = None,
) -> tuple[MLPPolicy, list[dict]]:
    """Train on ``env`` and return the greedy policy as a numpy MLP."""
    cfg = cfg or PPOConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    n_in = env.n_features
    model = ActorCritic(n_in, hidden, cfg.release_bias)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    history: list[dict] = []
    frontier = 0.0

    def begin_episode() -> np.ndarray:
        if frontier > 0.05 and rng.random() < cfg.practice_frac:
            try:
                return env.spawn_at(float(rng.uniform(0.0, frontier))).features
            except NotImplementedError:
                pass
        return env.reset().features

    obs = begin_episode()
    ep_ticks = 0
    for update in range(cfg.updates):
        buf_obs, buf_act, buf_logp, buf_rew, buf_done, buf_val = [], [], [], [], [], []
        finished: list[float] = []
        for _ in range(cfg.steps_per_update):
            x = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                dist, value = model(x)
                action = dist.sample()
            a = int(action.item())
            reward = 0.0
            for _ in range(max(1, cfg.action_repeat)):
                result = env.step(a)
                reward += float(result.reward)
                ep_ticks += 1
                if result.done:
                    break
            done = result.done or ep_ticks >= cfg.max_episode_ticks
            buf_obs.append(obs)
            buf_act.append(a)
            buf_logp.append(float(dist.log_prob(action).item()))
            buf_rew.append(reward)
            buf_done.append(done)
            buf_val.append(float(value.item()))
            if done:
                if result.outcome is not Outcome.RUNNING:
                    finished.append(env.progress)
                obs = begin_episode()
                ep_ticks = 0
            else:
                obs = result.obs.features

        with torch.no_grad():
            _, last_value = model(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
        rewards = np.asarray(buf_rew, dtype=np.float32)
        values = np.asarray(buf_val + [float(last_value.item())], dtype=np.float32)
        dones = np.asarray(buf_done, dtype=np.float32)
        adv = np.zeros_like(rewards)
        gae = 0.0
        for t in reversed(range(len(rewards))):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + cfg.gamma * values[t + 1] * nonterminal - values[t]
            gae = delta + cfg.gamma * cfg.lam * nonterminal * gae
            adv[t] = gae
        returns = adv + values[:-1]

        t_obs = torch.as_tensor(np.asarray(buf_obs), dtype=torch.float32)
        t_act = torch.as_tensor(np.asarray(buf_act), dtype=torch.int64)
        t_logp = torch.as_tensor(np.asarray(buf_logp), dtype=torch.float32)
        t_adv = torch.as_tensor(adv, dtype=torch.float32)
        t_adv = (t_adv - t_adv.mean()) / (t_adv.std() + 1e-8)
        t_ret = torch.as_tensor(returns, dtype=torch.float32)

        n = len(buf_act)
        for _ in range(cfg.epochs):
            perm = torch.randperm(n)
            for start in range(0, n, cfg.minibatch):
                idx = perm[start : start + cfg.minibatch]
                dist, value = model(t_obs[idx])
                logp = dist.log_prob(t_act[idx])
                ratio = torch.exp(logp - t_logp[idx])
                surr = torch.min(
                    ratio * t_adv[idx],
                    torch.clamp(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip) * t_adv[idx],
                )
                loss = (
                    -surr.mean()
                    + cfg.value_coef * (t_ret[idx] - value).pow(2).mean()
                    - cfg.entropy * dist.entropy().mean()
                )
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                opt.step()

        # Greedy evaluation from the level start: the number that matters.
        greedy = to_numpy_policy(model, n_in)
        evaluation = env.run_policy(greedy, max_ticks=cfg.max_episode_ticks)
        frontier = max(frontier, evaluation.end_progress)
        stats = {
            "update": update,
            "episodes": len(finished),
            "mean_progress": float(np.mean(finished)) if finished else 0.0,
            "greedy_progress": evaluation.end_progress,
            "frontier": frontier,
        }
        history.append(stats)
        if on_update is not None:
            on_update(update, stats)
        obs = begin_episode()
        ep_ticks = 0

    return to_numpy_policy(model, n_in), history
