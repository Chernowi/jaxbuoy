import json
from pathlib import Path

import numpy as np


class MinimalRNNPolicy:
    def __init__(self, model_path, metadata_path=None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        self.weights = np.load(model_path)
        self.metadata = {}
        if metadata_path is not None:
            metadata_path = Path(metadata_path)
            if metadata_path.exists():
                self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        self.obs_dim = int(self.weights["encoder_w"].shape[0])
        self.hidden_dim = int(self.weights["encoder_w"].shape[1])
        self.action_dim = int(self.weights["actor_out_b"].shape[0])
        self.activation = str(self.metadata.get("activation", "tanh"))

        self.hidden = np.zeros((self.hidden_dim,), dtype=np.float32)

    def reset(self):
        self.hidden.fill(0.0)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-x))

    def _act_fn(self, x):
        if self.activation == "relu":
            return np.maximum(x, 0.0)
        return np.tanh(x)

    def _gru_step(self, x, h):
        r = self._sigmoid(
            x @ self.weights["gru_ir_w"]
            + self.weights["gru_ir_b"]
            + h @ self.weights["gru_hr_w"]
        )
        z = self._sigmoid(
            x @ self.weights["gru_iz_w"]
            + self.weights["gru_iz_b"]
            + h @ self.weights["gru_hz_w"]
        )
        n = np.tanh(
            x @ self.weights["gru_in_w"]
            + self.weights["gru_in_b"]
            + r * (h @ self.weights["gru_hn_w"] + self.weights["gru_hn_b"])
        )
        return (1.0 - z) * n + z * h

    def infer(self, obs, done=False, deterministic=True, clip_actions=True, rng=None):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.shape != (self.obs_dim,):
            raise ValueError(f"Expected obs shape {(self.obs_dim,)}, got {obs.shape}")

        if done:
            self.reset()

        emb = self._act_fn(obs @ self.weights["encoder_w"] + self.weights["encoder_b"])
        self.hidden = self._gru_step(emb, self.hidden)

        actor_hidden = self._act_fn(
            self.hidden @ self.weights["actor_hidden_w"] + self.weights["actor_hidden_b"]
        )
        mean = actor_hidden @ self.weights["actor_out_w"] + self.weights["actor_out_b"]

        if deterministic:
            action = mean
        else:
            if rng is None:
                rng = np.random.default_rng()
            std = np.exp(self.weights["log_std"])
            action = mean + std * rng.standard_normal(size=(self.action_dim,), dtype=np.float32)

        if clip_actions:
            action = np.clip(action, -1.0, 1.0)

        return action.astype(np.float32)
