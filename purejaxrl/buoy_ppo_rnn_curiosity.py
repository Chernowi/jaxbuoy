from typing import Any, NamedTuple, Sequence

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState

from buoy_env import BuoySearchEnv
from buoy_models import ActorCriticContinuousRNN, ScannedRNN
from buoy_wrappers import LogWrapper, NormalizeVecObservation, NormalizeVecReward, VecEnv


class RNDNetwork(nn.Module):
    output_dim: int = 64
    hidden_sizes: Sequence[int] = (256, 256)
    activation: str = "relu"

    @nn.compact
    def __call__(self, obs):
        activation_fn = nn.relu if self.activation == "relu" else nn.tanh
        x = obs
        for h in self.hidden_sizes:
            x = nn.Dense(
                h,
                kernel_init=orthogonal(np.sqrt(2.0)),
                bias_init=constant(0.0),
            )(x)
            x = activation_fn(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(1.0),
            bias_init=constant(0.0),
        )(x)
        return x


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    ext_reward: jnp.ndarray
    int_reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    info: Any


def make_train(ppo_cfg, env_cfg, wandb_enabled=False):
    cfg = dict(ppo_cfg)
    if cfg["NUM_ENVS"] % cfg["NUM_MINIBATCHES"] != 0:
        raise ValueError("NUM_ENVS must be divisible by NUM_MINIBATCHES for RNN PPO")
    cfg["NUM_UPDATES"] = cfg["TOTAL_TIMESTEPS"] // cfg["NUM_STEPS"] // cfg["NUM_ENVS"]
    cfg["MINIBATCH_SIZE"] = (
        cfg["NUM_ENVS"] * cfg["NUM_STEPS"] // cfg["NUM_MINIBATCHES"]
    )

    curiosity_coef = float(cfg.get("CURIOSITY_COEF", 0.01))
    curiosity_lr = float(cfg.get("CURIOSITY_LR", cfg["LR"]))
    curiosity_output_dim = int(cfg.get("CURIOSITY_OUTPUT_DIM", 64))
    curiosity_hidden_sizes = tuple(cfg.get("CURIOSITY_HIDDEN_SIZES", [256, 256]))
    curiosity_activation = cfg.get("CURIOSITY_ACTIVATION", "relu")

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
        network = ActorCriticContinuousRNN(
            action_dim=env.action_space(env_params).shape[0],
            hidden_sizes=tuple(cfg["HIDDEN_SIZES"]),
            activation=cfg["ACTIVATION"],
            rnn_hidden_size=cfg["RNN_HIDDEN_SIZE"],
        )

        predictor = RNDNetwork(
            output_dim=curiosity_output_dim,
            hidden_sizes=curiosity_hidden_sizes,
            activation=curiosity_activation,
        )
        target = RNDNetwork(
            output_dim=curiosity_output_dim,
            hidden_sizes=curiosity_hidden_sizes,
            activation=curiosity_activation,
        )

        rng, net_rng, pred_rng, tgt_rng = jax.random.split(rng, 4)
        obs_shape = env.observation_space(env_params).shape
        init_obs_batch = jnp.zeros((cfg["NUM_ENVS"], *obs_shape), dtype=jnp.float32)
        init_x = (
            jnp.zeros(
                (1, cfg["NUM_ENVS"], *obs_shape),
                dtype=jnp.float32,
            ),
            jnp.zeros((1, cfg["NUM_ENVS"]), dtype=bool),
        )
        init_hstate = ScannedRNN.initialize_carry(
            cfg["NUM_ENVS"], cfg["RNN_HIDDEN_SIZE"]
        )
        policy_params = network.init(net_rng, init_hstate, init_x)
        predictor_params = predictor.init(pred_rng, init_obs_batch)
        target_params = target.init(tgt_rng, init_obs_batch)

        policy_optimizer = optax.chain(
            optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
            optax.adam(
                learning_rate=linear_schedule if cfg.get("ANNEAL_LR", False) else cfg["LR"],
                eps=1e-5,
            ),
        )
        curiosity_optimizer = optax.chain(
            optax.clip_by_global_norm(cfg["MAX_GRAD_NORM"]),
            optax.adam(curiosity_lr, eps=1e-5),
        )

        train_state = TrainState.create(
            apply_fn=network.apply,
            params=policy_params,
            tx=policy_optimizer,
        )
        curiosity_state = TrainState.create(
            apply_fn=predictor.apply,
            params=predictor_params,
            tx=curiosity_optimizer,
        )

        rng, reset_rng = jax.random.split(rng)
        env_keys = jax.random.split(reset_rng, cfg["NUM_ENVS"])
        obs, env_state = env.reset(env_keys, env_params)
        hstate = ScannedRNN.initialize_carry(cfg["NUM_ENVS"], cfg["RNN_HIDDEN_SIZE"])
        last_done = jnp.zeros((cfg["NUM_ENVS"],), dtype=bool)

        def _update_step(runner_state, update_idx):
            def _env_step(runner_state, _):
                (
                    train_state,
                    curiosity_state,
                    env_state,
                    last_obs,
                    last_done,
                    hstate,
                    rng,
                ) = runner_state

                rng, act_rng = jax.random.split(rng)
                ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
                hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
                action = pi.sample(seed=act_rng).squeeze(axis=0)
                log_prob = pi.log_prob(action).squeeze(axis=0)
                value = value.squeeze(axis=0)

                rng, step_rng = jax.random.split(rng)
                step_keys = jax.random.split(step_rng, cfg["NUM_ENVS"])
                next_obs, next_env_state, ext_reward, done, info = env.step(
                    step_keys, env_state, action, env_params
                )

                pred_feat = predictor.apply(curiosity_state.params, next_obs)
                target_feat = target.apply(target_params, next_obs)
                int_reward = curiosity_coef * jnp.mean(
                    jnp.square(pred_feat - jax.lax.stop_gradient(target_feat)),
                    axis=-1,
                )
                total_reward = ext_reward + int_reward

                transition = Transition(
                    last_done,
                    action,
                    value,
                    total_reward,
                    ext_reward,
                    int_reward,
                    log_prob,
                    last_obs,
                    next_obs,
                    info,
                )
                next_runner_state = (
                    train_state,
                    curiosity_state,
                    next_env_state,
                    next_obs,
                    done,
                    hstate,
                    rng,
                )
                return next_runner_state, transition

            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, cfg["NUM_STEPS"]
            )

            (
                train_state,
                curiosity_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
            ) = runner_state
            ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(axis=0)

            def _calculate_gae(traj_batch, last_val, last_done):
                def _gae_scan(carry, transition):
                    gae, next_value, next_done = carry
                    delta = (
                        transition.reward
                        + cfg["GAMMA"] * next_value * (1.0 - next_done)
                        - transition.value
                    )
                    gae = (
                        delta
                        + cfg["GAMMA"]
                        * cfg["GAE_LAMBDA"]
                        * (1.0 - next_done)
                        * gae
                    )
                    return (gae, transition.value, transition.done), gae

                (_, _, _), advantages = jax.lax.scan(
                    _gae_scan,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                returns = advantages + traj_batch.value
                return advantages, returns

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            def _update_epoch(update_state, _):
                def _update_minibatch(states, batch_info):
                    policy_state, curiosity_state = states
                    mb_hstate, mb_traj, mb_adv, mb_targets = batch_info
                    mb_hstate = mb_hstate[0]

                    def _policy_loss_fn(params):
                        _, pi, value = network.apply(
                            params,
                            mb_hstate,
                            (mb_traj.obs, mb_traj.done),
                        )
                        log_prob = pi.log_prob(mb_traj.action)

                        value_pred_clipped = mb_traj.value + (
                            value - mb_traj.value
                        ).clip(-cfg["CLIP_EPS"], cfg["CLIP_EPS"])
                        value_losses = jnp.square(value - mb_targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - mb_targets)
                        value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        norm_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)
                        ratio = jnp.exp(log_prob - mb_traj.log_prob)
                        loss_actor_1 = ratio * norm_adv
                        loss_actor_2 = (
                            jnp.clip(ratio, 1.0 - cfg["CLIP_EPS"], 1.0 + cfg["CLIP_EPS"])
                            * norm_adv
                        )
                        actor_loss = -jnp.minimum(loss_actor_1, loss_actor_2).mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            actor_loss
                            + cfg["VF_COEF"] * value_loss
                            - cfg["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, actor_loss, entropy)

                    def _curiosity_loss_fn(params):
                        next_obs = mb_traj.next_obs.reshape(
                            (-1, mb_traj.next_obs.shape[-1])
                        )
                        pred_feat = predictor.apply(params, next_obs)
                        target_feat = target.apply(target_params, next_obs)
                        loss = jnp.mean(
                            jnp.square(pred_feat - jax.lax.stop_gradient(target_feat))
                        )
                        return loss

                    policy_grad_fn = jax.value_and_grad(_policy_loss_fn, has_aux=True)
                    (policy_loss, policy_aux), policy_grads = policy_grad_fn(policy_state.params)
                    policy_state = policy_state.apply_gradients(grads=policy_grads)

                    curiosity_grad_fn = jax.value_and_grad(_curiosity_loss_fn)
                    curiosity_loss, curiosity_grads = curiosity_grad_fn(curiosity_state.params)
                    curiosity_state = curiosity_state.apply_gradients(grads=curiosity_grads)

                    losses = (policy_loss, policy_aux, curiosity_loss)
                    return (policy_state, curiosity_state), losses

                (
                    train_state,
                    curiosity_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, perm_rng = jax.random.split(rng)

                permutation = jax.random.permutation(perm_rng, cfg["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1),
                    batch,
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        x.reshape(
                            (
                                x.shape[0],
                                cfg["NUM_MINIBATCHES"],
                                -1,
                            )
                            + x.shape[2:]
                        ),
                        1,
                        0,
                    ),
                    shuffled,
                )

                (train_state, curiosity_state), loss_info = jax.lax.scan(
                    _update_minibatch,
                    (train_state, curiosity_state),
                    minibatches,
                )
                return (
                    train_state,
                    curiosity_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ), loss_info

            init_hstate = initial_hstate[jnp.newaxis, :]
            update_state = (
                train_state,
                curiosity_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch,
                update_state,
                None,
                cfg["UPDATE_EPOCHS"],
            )

            train_state = update_state[0]
            curiosity_state = update_state[1]
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
                "extrinsic_reward_mean": jnp.mean(traj_batch.ext_reward),
                "intrinsic_reward_mean": jnp.mean(traj_batch.int_reward),
                "reward_total_mean": jnp.mean(traj_batch.reward),
                "loss_total": jnp.mean(loss_info[0]),
                "loss_value": jnp.mean(loss_info[1][0]),
                "loss_actor": jnp.mean(loss_info[1][1]),
                "entropy": jnp.mean(loss_info[1][2]),
                "loss_curiosity": jnp.mean(loss_info[2]),
            }

            if cfg.get("DEBUG", False):
                def debug_callback(m):
                    print(
                        f"step={int(m['global_step'])} return={float(m['episodic_return']):.3f} "
                        f"success={float(m['success_rate']):.3f} int_rew={float(m['intrinsic_reward_mean']):.4f}"
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
                        "train/extrinsic_reward_mean": float(m["extrinsic_reward_mean"]),
                        "train/intrinsic_reward_mean": float(m["intrinsic_reward_mean"]),
                        "train/reward_total_mean": float(m["reward_total_mean"]),
                        "loss/total": float(m["loss_total"]),
                        "loss/value": float(m["loss_value"]),
                        "loss/actor": float(m["loss_actor"]),
                        "loss/entropy": float(m["entropy"]),
                        "loss/curiosity": float(m["loss_curiosity"]),
                    }
                    payload = {k: v for k, v in payload.items() if not np.isnan(v)}
                    if payload:
                        wandb.log(payload, step=int(m["global_step"]))

                jax.debug.callback(wb_callback, metric)

            runner_state = (
                train_state,
                curiosity_state,
                env_state,
                last_obs,
                last_done,
                hstate,
                rng,
            )
            return runner_state, metric

        rng, loop_rng = jax.random.split(rng)
        runner_state = (
            train_state,
            curiosity_state,
            env_state,
            obs,
            last_done,
            hstate,
            loop_rng,
        )
        update_indices = jnp.arange(cfg["NUM_UPDATES"])
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, update_indices)
        return {"runner_state": runner_state, "metrics": metrics}

    return train
