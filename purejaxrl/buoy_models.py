from typing import Sequence

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.linen.initializers import constant, orthogonal
import functools


class ActorCriticContinuous(nn.Module):
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh

        actor = x
        for h in self.hidden_sizes:
            actor = nn.Dense(
                h,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(actor)
            actor = activation_fn(actor)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        actor_log_std = self.param(
            "log_std", nn.initializers.zeros, (self.action_dim,)
        )
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))

        critic = x
        for h in self.hidden_sizes:
            critic = nn.Dense(
                h,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(critic)
            critic = activation_fn(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)
        return pi, jnp.squeeze(value, axis=-1)


class ScannedRNN(nn.Module):
    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, jnp.newaxis],
            self.initialize_carry(ins.shape[0], self.hidden_size),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=self.hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        return jnp.zeros((batch_size, hidden_size), dtype=jnp.float32)


class ActorCriticContinuousRNN(nn.Module):
    action_dim: int
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "tanh"
    rnn_hidden_size: int = 128

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh

        embedding = obs
        for h in self.hidden_sizes:
            embedding = nn.Dense(
                h,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(embedding)
            embedding = activation_fn(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.rnn_hidden_size)(hidden, rnn_in)

        actor = embedding
        for h in self.hidden_sizes:
            actor = nn.Dense(
                h,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(actor)
            actor = activation_fn(actor)
        actor_mean = nn.Dense(
            self.action_dim,
            kernel_init=orthogonal(0.01),
            bias_init=constant(0.0),
        )(actor)
        actor_log_std = self.param(
            "log_std", nn.initializers.zeros, (self.action_dim,)
        )
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_log_std))

        critic = embedding
        for h in self.hidden_sizes:
            critic = nn.Dense(
                h,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(critic)
            critic = activation_fn(critic)
        value = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(critic)

        return hidden, pi, jnp.squeeze(value, axis=-1)
