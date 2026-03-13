from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.training.train_state import TrainState

from buoy_env import BuoySearchEnv
from buoy_models import ActorCriticContinuous
from buoy_wrappers import LogWrapper, NormalizeVecObservation, NormalizeVecReward, VecEnv


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: Any


def make_train(ppo_cfg, env_cfg, wandb_enabled=False):
    cfg = dict(ppo_cfg)
    cfg["NUM_UPDATES"] = cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"]
    cfg["MINIBATCH_SIZE"] = (
        cfg["NUM_ENVS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]
    )

    env, env_params = BuoySearchEnv(env_cfg), None
    env = LogWrapper(env)
    env = VecEnv(env)
    if cfg.get("NORMALIZE_OBS", False):
        env = NormalizeVecObservation(env)
    if cfg.get("NORMALIZE_REWARD", False):
        env = NormalizeVecReward(env, cfg["GAMMA"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (cfg["NUM_MINIBATCHES"] * cfg["UPDATE_EPOCHS"]))
            / cfg["NUM_UPDATES"]
        )
        return cfg["LR"] * frac

    def train(rng):
        network = ActorCriticContinuous(
            action_dim=env.action_space(env_params).shape[0],
            hidden_sizes=tuple(cfg["HIDDEN_SIZES"]),
            activation=cfg["ACTIVATION"],
        )

        rng, net_rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape, dtype=jnp.float32)
        network_params = network.init(net_rng, init_x)

        optimizer = optax.chain(
            optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
            optax.adam(
                learning_rate=linear_schedule if cfg.get("ANNEAL_LR", False) else cfg["LR"],
                eps=1e-5,
            ),
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optimizer,
        )

        rng, reset_rng = jax.random.split(rng)
        env_keys = jax.random.split(reset_rng, cfg["NUM_ENVS"])
        obs, env_state = env.reset(env_keys, env_params)

        def _update_step(runner_state, update_idx):
            def _env_step(runner_state, _):
                train_state, env_state, last_obs, rng = runner_state

                rng, act_rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=act_rng)
                log_prob = pi.log_prob(action)

                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, cfg["NUM_ENVS"])
                next_obs, next_env_state, reward, done, info = env.step(
                    step_keys, env_state, action, env_params
                )

                transition = Transition(done, action, value, reward, log_prob, last_obs, info)
                next_runner_state = (train_state, next_env_state, next_obs, rng)
                return next_runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, cfg["NUM_STEPS"]
            )

            train_state, env_state, last_obs, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _gae_scan(carry, transition):
                    gae, next_value = carry
                    delta = (
                        transition.reward
                        + cfg["GAMMA"] * next_value * (1.0 - transition.done)
                        - transition.value
                    )
                    gae = delta + cfg["GAMMA"] * cfg["GAE_LAMBDA"] * (1.0 - transition.done) * gae
                    return (gae, transition.value), gae

                (_, _), advantages = jax.lax.scan(
                    _gae_scan,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                returns = advantages + traj_batch.value
                return advantages, returns

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    batch_traj, batch_adv, batch_targets = batch_info

                    def _loss_fn(params):
                        pi, value = network.apply(params, batch_traj.obs)
                        log_prob = pi.log_prob(batch_traj.action)

                        value_pred_clipped = batch_traj.value + (
                            value - batch_traj.value
                        ).clip(-cfg["CLIP_EPS"], cfg["CLIP_EPS"])
                        value_losses = jnp.square(value - batch_targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - batch_targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        norm_adv = (batch_adv - batch_adv.mean()) / (batch_adv.std() + 1e-8)
                        ratio = jnp.exp(log_prob - batch_traj.log_prob)
                        loss_actor_1 = ratio * norm_adv
                        loss_actor_2 = (
                            jnp.clip(ratio, 1.0 - cfg["CLIP_EPS"], 1.0 + cfg["CLIP_EPS"])
                            * norm_adv
                        )
                        actor_loss = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = actor_loss + cfg["VF_COEF"] * value_loss - cfg["ENT_COEF"] * entropy
                        return total_loss, (value_loss, actor_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux_losses), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (total_loss, aux_losses)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, perm_rng = jax.random.split(rng)

                batch_size = cfg["MINIBATCH_SIZE"] * cfg["NUM_MINIBATCHES"]
                permutation = jax.random.permutation(perm_rng, batch_size)

                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((cfg["NUM_MINIBATCHES"], -1) + x.shape[1:]),
                    shuffled,
                )

                train_state, loss_info = jax.lax.scan(
                    _update_minibatch,
                    train_state,
                    minibatches,
                )
                return (train_state, traj_batch, advantages, targets, rng), loss_info

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                update_state,
                None,
                cfg["UPDATE_EPOCHS"],
            )

            train_state = update_state[0]
            rng = update_state[-1]
            info = traj_batch.info

            episode_mask = info["returned_episode"].astype(jnp.float32)
            completed = jnp.sum(episode_mask)

            def safe_mean(values):
                return jnp.where(
                    completed > 0,
                    jnp.sum(values * episode_mask) / (completed + 1e-8),
                    jnp.nan,
                )

            metric = {
                "global_step": (update_idx + 1) * cfg["NUM_STEPS"] * cfg["NUM_ENVS"],
                "episodic_return": safe_mean(info["returned_episode_returns"]),
                "episodic_length": safe_mean(info["returned_episode_lengths"]),
                "success_rate": safe_mean(info["success"].astype(jnp.float32)),
                "out_of_bounds_rate": safe_mean(info["out_of_bounds"].astype(jnp.float32)),
                "timeout_rate": safe_mean(info["timed_out"].astype(jnp.float32)),
                "explore_reward_mean": jnp.mean(info["explore_reward"]),
                "loss_total": jnp.mean(loss_info[0]),
                "loss_value": jnp.mean(loss_info[1][0]),
                "loss_actor": jnp.mean(loss_info[1][1]),
                "entropy": jnp.mean(loss_info[1][2]),
            }

            if cfg.get("DEBUG", False):
                def debug_callback(m):
                    print(
                        f"step={int(m['global_step'])} return={float(m['episodic_return']):.3f} success={float(m['success_rate']):.3f}"
                    )

                jax.debug.callback(debug_callback, metric)

            if wandb_enabled:
                def wb_callback(m):
                    payload = {
                        "train/episodic_return": float(m["episodic_return"]),
                        "train/episodic_length": float(m["episodic_length"]),
                        "train/success_rate": float(m["success_rate"]),
                        "train/out_of_bounds_rate": float(m["out_of_bounds_rate"]),
                        "train/timeout_rate": float(m["timeout_rate"]),
                        "train/explore_reward_mean": float(m["explore_reward_mean"]),
                        "loss/total": float(m["loss_total"]),
                        "loss/value": float(m["loss_value"]),
                        "loss/actor": float(m["loss_actor"]),
                        "loss/entropy": float(m["entropy"]),
                    }
                    payload = {
                        k: v for k, v in payload.items() if not np.isnan(v)
                    }
                    if payload:
                        wandb.log(payload, step=int(m["global_step"]))

                jax.debug.callback(wb_callback, metric)

            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        rng, loop_rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, loop_rng)
        update_indices = jnp.arange(cfg["NUM_UPDATES"])
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, update_indices)
        return {"runner_state": runner_state, "metrics": metrics}

    return train
