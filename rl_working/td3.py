from jax.lib import xla_bridge

print("Platform used: ", xla_bridge.get_backend().platform)

import time
import jax
import wandb
import os
import sys
from functools import partial

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
print(jax.devices())
jax.config.update("jax_default_device", jax.devices()[0])
print("JaX backend: ", jax.default_backend())

import jax.numpy as jnp
import numpy as np
from jax import jit, config, vmap, block_until_ready
from flax import struct
import flax.linen as nn
import optax
from flax.linen.initializers import constant, orthogonal
from typing import NamedTuple, Sequence, Tuple, Optional
from flax.training.train_state import TrainState
import distrax
import chex
import pickle
import jax.lax as lax

# Placeholder imports (uncomment and adjust paths as needed)
from envs.utils.wrappers import VecEnv, LogWrapper
from envs.single_rydberg_env import RydbergEnv
from envs.single_stirap_env import SimpleStirap
from envs.multistep_stirap_env import MultiStirap
from envs.single_rydberg_two_photon_env import RydbergTwoEnv
from envs.single_transmon_reset_env import TransmonResetEnv

from env_configs.configs import get_plot_elem_names, get_simple_stirap_params, get_multi_stirap_params, get_rydberg_cz_params, get_rydberg_two_params, get_transmon_reset_params
import matplotlib.pyplot as plt
import argparse

# Check if GPU is available
if ("cuda" in str(jax.devices())) or ("Cuda" in str(jax.devices())):
    print("Connected to a GPU")
    processor = "gpu"
    default_dtype = jnp.float32
    default_int = jnp.int32
else:
    jax.config.update("jax_platform_name", "cpu")
    print("Not connected to a GPU")
    jax.config.update("jax_enable_x64", True)
    processor = "cpu"
    default_dtype = jnp.float64
    default_int = jnp.int64

envs_class_dict = {
    "simple_stirap": SimpleStirap,
    "multi_stirap": MultiStirap,
    "rydberg": RydbergEnv,
    "rydberg_two": RydbergTwoEnv,
    "transmon_reset": TransmonResetEnv,
}


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict
    next_obs: jnp.ndarray


class Actor(nn.Module):
    action_dim: int
    activation: str = "relu"
    layer_size: int = 256

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "elu":
            activation = nn.elu
        elif self.activation == "leaky_relu":
            activation = nn.leaky_relu
        elif self.activation == "relu6":
            activation = nn.relu6
        elif self.activation == "selu":
            activation = nn.selu
        else:
            activation = nn.tanh
        x = nn.Dense(self.layer_size)(x)
        x = activation(x)
        x = nn.Dense(self.layer_size)(x)
        x = activation(x)
        x = nn.Dense(self.action_dim)(x)
        actions = nn.tanh(x)  # Outputs in [-1, 1], scaled later if needed
        return actions


class Critic(nn.Module):
    activation: str = "relu"
    layer_size: int = 256

    @nn.compact
    def __call__(self, obs, action):
        if self.activation == "relu":
            activation = nn.relu
        elif self.activation == "elu":
            activation = nn.elu
        elif self.activation == "leaky_relu":
            activation = nn.leaky_relu
        elif self.activation == "relu6":
            activation = nn.relu6
        elif self.activation == "selu":
            activation = nn.selu
        else:
            activation = nn.tanh
        x = jnp.concatenate([obs, action], axis=-1)
        x = nn.Dense(self.layer_size)(x)
        x = activation(x)
        x = nn.Dense(self.layer_size)(x)
        x = activation(x)
        q_value = nn.Dense(1)(x)
        return jnp.squeeze(q_value, axis=-1)


@struct.dataclass
class ReplayBufferState:
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray
    position: jnp.ndarray
    full: jnp.ndarray
    size: int = struct.field(pytree_node=False)


def init_replay_buffer(buffer_size: int, obs_shape: tuple,
                       action_shape: tuple) -> ReplayBufferState:
    return ReplayBufferState(obs=jnp.zeros((buffer_size, ) + obs_shape,
                                           dtype=default_dtype),
                             actions=jnp.zeros((buffer_size, ) + action_shape,
                                               dtype=default_dtype),
                             rewards=jnp.zeros((buffer_size, ),
                                               dtype=default_dtype),
                             next_obs=jnp.zeros((buffer_size, ) + obs_shape,
                                                dtype=default_dtype),
                             dones=jnp.zeros((buffer_size, ), dtype=jnp.bool_),
                             position=jnp.array(0, dtype=default_int),
                             full=jnp.array(False, dtype=jnp.bool_),
                             size=buffer_size)


@jax.jit
def add_to_buffer(buffer_state: ReplayBufferState,
                  transition: Transition) -> ReplayBufferState:
    batch_size = transition.obs.shape[0]
    buffer_size = buffer_state.size
    current_position = buffer_state.position
    new_position = (current_position + batch_size) % buffer_size
    indices = (jnp.arange(batch_size) + current_position) % buffer_size

    new_buffer_state = buffer_state.replace(
        obs=buffer_state.obs.at[indices].set(transition.obs),
        actions=buffer_state.actions.at[indices].set(transition.action),
        rewards=buffer_state.rewards.at[indices].set(transition.reward),
        next_obs=buffer_state.next_obs.at[indices].set(transition.next_obs),
        dones=buffer_state.dones.at[indices].set(transition.done),
        position=new_position,
        full=buffer_state.full | (new_position < current_position))
    return new_buffer_state


@partial(jax.jit, static_argnums=(2, ))
def sample_from_buffer(buffer_state: ReplayBufferState, rng: chex.PRNGKey,
                       batch_size: int) -> Transition:
    current_size = jax.lax.cond(buffer_state.full, lambda: buffer_state.size,
                                lambda: buffer_state.position)
    indices = jax.random.randint(rng, (batch_size, ), 0, current_size)
    return Transition(done=buffer_state.dones[indices],
                      action=buffer_state.actions[indices],
                      value=jnp.zeros((batch_size, ), dtype=jnp.float32),
                      reward=buffer_state.rewards[indices],
                      log_prob=jnp.zeros((batch_size, ), dtype=jnp.float32),
                      obs=buffer_state.obs[indices],
                      info={},
                      next_obs=buffer_state.next_obs[indices])


class TD3TrainState(NamedTuple):
    actor_state: TrainState
    target_actor_state: TrainState
    critic1_state: TrainState
    critic2_state: TrainState
    target_critic1_state: TrainState
    target_critic2_state: TrainState
    env_state: any
    last_obs: jnp.ndarray
    rng: chex.PRNGKey
    step: int
    buffer_state: ReplayBufferState


def td3_make_train(config):
    env = envs_class_dict[config["ENV_NAME"]](**config["ENV_PARAMS"])
    env = LogWrapper(env)
    env = VecEnv(env)
    env_params = env.default_params

    action_low = env.action_space(env_params).low
    action_high = env.action_space(env_params).high
    action_scale = (action_high - action_low) / 2.0
    action_bias = (action_high + action_low) / 2.0

    def linear_schedule_actor(count):
        frac = 1.0 - (count / config["NUM_UPDATES"])
        return frac * config["LR_ACTOR"]

    def linear_schedule_critic(count):
        frac = 1.0 - (count / config["NUM_UPDATES"])
        return frac * config["LR_CRITIC"]

    def train(rng: chex.PRNGKey):
        rng, actor_key, critic1_key, critic2_key = jax.random.split(rng, 4)
        action_dim = env.action_space(env_params).shape[0]
        obs_dim = env.observation_space(env_params).shape[0]

        actor = Actor(action_dim, config["ACTIVATION"], config["LAYER_SIZE"])
        critic = Critic(config["ACTIVATION"], config["LAYER_SIZE"])

        init_obs = jnp.zeros((1, obs_dim))
        init_action = jnp.zeros((1, action_dim))

        actor_params = actor.init(actor_key, init_obs)
        critic1_params = critic.init(critic1_key, init_obs, init_action)
        critic2_params = critic.init(critic2_key, init_obs, init_action)

        target_actor_params = jax.tree_map(lambda x: x.copy(), actor_params)
        target_critic1_params = jax.tree_map(lambda x: x.copy(),
                                             critic1_params)
        target_critic2_params = jax.tree_map(lambda x: x.copy(),
                                             critic2_params)

        if config["ANNEAL_LR"]:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule_actor, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule_critic, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR_ACTOR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR_CRITIC"], eps=1e-5),
            )

        actor_state = TrainState.create(apply_fn=actor.apply,
                                        params=actor_params,
                                        tx=actor_tx)
        critic1_state = TrainState.create(apply_fn=critic.apply,
                                          params=critic1_params,
                                          tx=critic_tx)
        critic2_state = TrainState.create(apply_fn=critic.apply,
                                          params=critic2_params,
                                          tx=critic_tx)

        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        buffer_state = init_replay_buffer(
            config["BUFFER_SIZE"],
            obs_shape=env.observation_space(env_params).shape,
            action_shape=env.action_space(env_params).shape)
        start_time = time.time()

        train_state = TD3TrainState(
            actor_state=actor_state,
            target_actor_state=actor_state.replace(params=target_actor_params),
            critic1_state=critic1_state,
            critic2_state=critic2_state,
            target_critic1_state=critic1_state.replace(
                params=target_critic1_params),
            target_critic2_state=critic2_state.replace(
                params=target_critic2_params),
            env_state=env_state,
            last_obs=obsv,
            rng=rng,
            step=0,
            buffer_state=buffer_state)

        def _update_step(runner_state, unused):

            def _env_step(runner_state, unused):
                train_state = runner_state
                rng = train_state.rng

                rng, action_key, noise_key = jax.random.split(rng, 3)

                # Use jax.lax.cond instead of if for warm-up logic
                def random_action(_):
                    return jax.random.uniform(action_key,
                                              shape=(config["NUM_ENVS"],
                                                     action_dim),
                                              minval=-1.0,
                                              maxval=1.0)

                def policy_action(_):
                    action = train_state.actor_state.apply_fn(
                        train_state.actor_state.params, train_state.last_obs)
                    noise_scale = config["EXPLORATION_NOISE"]
                    noise = noise_scale * jax.random.normal(
                        noise_key, action.shape)
                    return jnp.clip(action + noise, -1.0, 1.0)

                total_steps = train_state.step * config["NUM_ENVS"]
                action = jax.lax.cond(total_steps < config["LEARNING_STARTS"],
                                      random_action, policy_action, None)

                # Scale actions to environment bounds
                action = action * action_scale + action_bias

                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                next_obsv, env_state, reward, done, info = env.step(
                    rng_step, train_state.env_state, action, env_params)

                scaled_reward = config["REWARD_SCALE"] * reward
                transition = Transition(
                    done=done,
                    action=action,
                    value=jnp.zeros_like(
                        reward),  # Placeholder, computed later
                    reward=scaled_reward,
                    log_prob=jnp.zeros_like(reward),
                    obs=train_state.last_obs,
                    info=info,
                    next_obs=next_obsv)

                new_buffer_state = add_to_buffer(train_state.buffer_state,
                                                 transition)

                new_train_state = train_state._replace(
                    env_state=env_state,
                    last_obs=next_obsv,
                    rng=rng,
                    step=train_state.step + 1,
                    buffer_state=new_buffer_state)
                return new_train_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state,
                                                    None, config["NUM_STEPS"])

            def _update_networks(train_state, unused):
                rng, sample_rng, target_noise_rng = jax.random.split(
                    train_state.rng, 3)
                sampled_batch = sample_from_buffer(train_state.buffer_state,
                                                   sample_rng,
                                                   config["BATCH_SIZE"])

                # Target policy smoothing
                next_actions = train_state.target_actor_state.apply_fn(
                    train_state.target_actor_state.params,
                    sampled_batch.next_obs)
                clipped_noise = jnp.clip(
                    config["POLICY_NOISE"] *
                    jax.random.normal(target_noise_rng, next_actions.shape),
                    -config["NOISE_CLIP"], config["NOISE_CLIP"])
                next_actions = jnp.clip(next_actions + clipped_noise, -1.0,
                                        1.0) * action_scale + action_bias

                # Twin critic targets
                next_q1 = train_state.target_critic1_state.apply_fn(
                    train_state.target_critic1_state.params,
                    sampled_batch.next_obs, next_actions)
                next_q2 = train_state.target_critic2_state.apply_fn(
                    train_state.target_critic2_state.params,
                    sampled_batch.next_obs, next_actions)
                next_q = jnp.minimum(next_q1, next_q2)
                target_q = sampled_batch.reward + config["GAMMA"] * (
                    1 - sampled_batch.done) * next_q

                def critic_loss_fn(critic_params, critic_state):
                    current_q = critic_state.apply_fn(critic_params,
                                                      sampled_batch.obs,
                                                      sampled_batch.action)
                    return jnp.mean(jnp.square(current_q - target_q))

                critic1_loss, critic1_grads = jax.value_and_grad(
                    critic_loss_fn)(train_state.critic1_state.params,
                                    train_state.critic1_state)
                critic2_loss, critic2_grads = jax.value_and_grad(
                    critic_loss_fn)(train_state.critic2_state.params,
                                    train_state.critic2_state)
                critic1_state = train_state.critic1_state.apply_gradients(
                    grads=critic1_grads)
                critic2_state = train_state.critic2_state.apply_gradients(
                    grads=critic2_grads)

                def actor_loss_fn(actor_params):
                    actions = train_state.actor_state.apply_fn(
                        actor_params, sampled_batch.obs)
                    actions = actions * action_scale + action_bias
                    q_values = train_state.critic1_state.apply_fn(
                        critic1_state.params, sampled_batch.obs, actions)
                    return -jnp.mean(q_values)

                actor_loss, actor_grads = jax.value_and_grad(actor_loss_fn)(
                    train_state.actor_state.params)
                actor_state = train_state.actor_state.apply_gradients(
                    grads=actor_grads)

                # Soft updates
                target_critic1_params = optax.incremental_update(
                    critic1_state.params,
                    train_state.target_critic1_state.params, config["TAU"])
                target_critic2_params = optax.incremental_update(
                    critic2_state.params,
                    train_state.target_critic2_state.params, config["TAU"])
                target_actor_params = optax.incremental_update(
                    actor_state.params, train_state.target_actor_state.params,
                    config["TAU"])

                new_train_state = train_state._replace(
                    actor_state=actor_state,
                    target_actor_state=train_state.target_actor_state.replace(
                        params=target_actor_params),
                    critic1_state=critic1_state,
                    critic2_state=critic2_state,
                    target_critic1_state=train_state.target_critic1_state.
                    replace(params=target_critic1_params),
                    target_critic2_state=train_state.target_critic2_state.
                    replace(params=target_critic2_params),
                    rng=jax.random.split(rng)[0])
                return new_train_state, (critic1_loss, critic2_loss,
                                         actor_loss)

            def _skip_update(train_state, unused):
                return train_state, (jnp.array(0.0), jnp.array(0.0),
                                     jnp.array(0.0))

            total_steps = runner_state.step * config["NUM_ENVS"] * config[
                "NUM_STEPS"]
            should_update = total_steps >= config["LEARNING_STARTS"]
            train_state, losses = jax.lax.cond(should_update, _update_networks,
                                               _skip_update, runner_state,
                                               None)
            critic1_loss, critic2_loss, actor_loss = losses

            # Delayed actor update
            should_update_actor = should_update & (runner_state.step %
                                                   config["POLICY_FREQ"] == 0)
            train_state = jax.lax.cond(
                should_update_actor, lambda ts: ts._replace(
                    actor_state=train_state.actor_state,
                    target_actor_state=train_state.target_actor_state),
                lambda ts: ts, train_state)

            metric = traj_batch.info
            if config.get("LOGGING"):

                def callback(infos):
                    info, loss_info, step = infos
                    critic1_loss, critic2_loss, actor_loss = loss_info
                    timesteps = (info["timestep"][info["returned_episode"]] *
                                 config["NUM_ENVS"])
                    if step % config["LOG_FREQ"] != 0:
                        return

                    timestep = config["NUM_ENVS"] * config["NUM_STEPS"] * step
                    min_fidelity = jnp.min(info["fid"])
                    max_fidelity = jnp.max(info["fid"])
                    std_fidelity = jnp.std(info["fid"])

                    speed_1k = (time.time() - start_time) * 1e3 / timestep
                    print(f"time per 1k steps: {speed_1k} seconds")

                    wandb_log_dict = {
                        "timestep": timestep,
                        f"time_per_1k_steps": speed_1k,
                        f"max_fidelity": max_fidelity,
                        f"min_fidelity": min_fidelity,
                        f"std_fidelity": std_fidelity,
                        f"critic1_loss": jnp.mean(critic1_loss),
                        f"critic2_loss": jnp.mean(critic2_loss),
                        f"actor_loss": jnp.mean(actor_loss),
                    }

                    if (wandb.run and timestep %
                        (config["NUM_ENVS"] * config["LOG_FREQ"]) == 0):
                        elem_names = get_plot_elem_names(config["ENV_NAME"])

                        n_elem_names = len(elem_names)
                        fig, ax = plt.subplots(1,
                                               n_elem_names,
                                               figsize=(3 * n_elem_names, 3))

                        saved_elem = []
                        # if wandb.run and timestep % (config["LOG_FREQ"] * 10) == 0:
                        for elem_i, elem_name in enumerate(elem_names):
                            best_elem = info[elem_name][info["fid"] ==
                                                        max_fidelity][0]
                            saved_elem.append(best_elem)
                            x_values = np.linspace(0, 1, len(best_elem))
                            ax[elem_i].plot(x_values, best_elem)
                            ax[elem_i].set_title(f"{elem_name} vs Time")

                        # timestr with miliseconds
                        timestr = str(time.time()).replace(".", "")
                        # Define the directory path
                        output_dir_img = "output_images_temp"

                        # Check if the directory exists, and if not, create it
                        if not os.path.exists(output_dir_img):
                            os.makedirs(output_dir_img)
                            print(f"{output_dir_img} created")

                        # Create the full file path for the image
                        fpath = os.path.join(output_dir_img, f"{timestr}.png")

                        # Save the plot to the file
                        plt.savefig(fpath)
                        plt.close()

                        if config.get("LOG_WAND"):
                            run_id = wandb.run.id

                        # Open the file in write-binary mode and save the data
                        env_name_var = config["ENV_NAME"]
                        directory = f"saved_data/{env_name_var}/{run_id}"
                        # Check if the directory exists
                        if not os.path.exists(directory):
                            # Create the directory if it doesn't exist
                            os.makedirs(directory)
                            print(f"Directory '{directory}' created.")

                        saved_env_name = config["ENV_NAME"]
                        data_fpath = os.path.join(
                            f"saved_data/{env_name_var}/{run_id}/{saved_env_name}_{run_id}_{timestep}.pkl"
                        )
                        with open(data_fpath, "wb") as file:
                            pickle.dump(saved_elem, file)

                        if config.get("LOG_WAND"):
                            wandb_log_dict[f"action_fig"] = wandb.Image(fpath)

                    for log_elem in info.keys():
                        if "returned_episode" not in log_elem:
                            continue

                        return_values = info[log_elem][
                            info["returned_episode"]]

                        log_val_name = log_elem.split("_")[-1]

                        mean_value = np.mean(return_values)
                        print(
                            f"global step={timestep}, episodic {log_val_name} mean={mean_value}"
                        )
                        wandb_log_dict[
                            f"episodic_{log_val_name}_mean"] = mean_value
                    if config.get("LOG_WAND"):
                        if wandb.run:
                            wandb.log(wandb_log_dict)

                    if config.get("LOCAL_LOGGING"):
                        env_name_var = config["ENV_NAME"]
                        save_name = config["LOCAL_SAVE_NAME"]
                        average_directory = f"episodic_data/{env_name_var}"

                        # Ensure the directory exists
                        os.makedirs(average_directory, exist_ok=True)

                        # Define the full PKL file path
                        data_fpath = os.path.join(
                            average_directory,
                            f"{env_name_var}_{save_name}.pkl")

                        # Check if file exists, if not, initialize an empty list
                        if not os.path.exists(data_fpath):
                            with open(data_fpath, "wb") as file:
                                pickle.dump([], file)

                        # Create data entry
                        data_entry = {
                            "timestep": timestep,
                            "mean_reward": jnp.mean(info["reward"]),
                            "mean_fidelity": jnp.mean(info["fid"]),
                            "max_fidelity": jnp.max(info["fid"]),
                            "min_fidelity": jnp.min(info["fid"]),
                            "std_fidelity": jnp.std(info["fid"]),
                        }

                        # Append data to the PKL file
                        with open(data_fpath, "rb+") as file:
                            data = pickle.load(file)  # Load previous data
                            data.append(data_entry)  # Append new entry
                            file.seek(0)  # Move cursor to the start
                            pickle.dump(
                                data, file)  # Overwrite file with updated list

                jax.debug.callback(callback,
                                   (metric, losses, train_state.step))
            return train_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = train_state._replace(rng=_rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None,
                                            config["NUM_UPDATES"])
        return {"runner_state": runner_state}

    return train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    #HYPERPARAMETERS
    parser.add_argument("--seed",
                        type=int,
                        default=10,
                        help="Initial seed for the run")

    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--num_envs",
                        type=int,
                        default=16,
                        help="Number of environments run in parallel")
    parser.add_argument("--num_updates",
                        type=int,
                        default=5000,
                        help="Number of updates to run")
    parser.add_argument("--num_minibatches", type=int, default=4)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--ent_coef", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--layer_size", type=int, default=256)
    parser.add_argument("--anneal_lr", type=int, default=0)

    #CHOOSE ENVIRONMENT
    parser.add_argument(
        "--env",
        choices=[
            "multi_stirap",
            "simple_stirap",
            "rydberg",
            "rydberg_two",
            "transmon_reset",
        ],
        default="simple_stirap",
        help="Environment to run",
    )

    #NOISE PARAMS
    parser.add_argument(
        "--noise",
        choices=["None", "ou", "g"],
        default="None",
        help="Which nosie to use for the environment",
    )
    parser.add_argument("--sigma_phase",
                        type=float,
                        default=0,
                        help="Value for amp_1 parameter")
    parser.add_argument("--sigma_amp",
                        type=float,
                        default=0,
                        help="Value for amp_2 parameter")
    parser.add_argument("--mu_phase",
                        type=float,
                        default=0,
                        help="Range to sample mu from for ou noise")
    parser.add_argument("--mu_amp",
                        type=float,
                        default=0,
                        help="Range to sample mu from for ou noise")

    #SIMPLE STIRAP PARAMS
    parser.add_argument(
        "--gamma_ss",
        type=float,
        default=1,
        help="Value for gamma parameter in simple stirap",
    )
    parser.add_argument(
        "--omega_ss",
        type=float,
        default=30,
        help="Value for omega parameter in simple stirap",
    )
    parser.add_argument(
        "--delta_ss",
        type=float,
        default=20,
        help="Value for delta parameter in simple stirap",
    )
    parser.add_argument(
        "--x_detuning_ss",
        type=float,
        default=100,
        help="Value for x_detuning parameter in simple stirap",
    )
    parser.add_argument(
        "--final_state_zero_ss",
        type=float,
        default=0.0,
        help="Value for final state [0] parameter in simple_stirap",
    )

    #GENERAL PULSE PARAMS
    parser.add_argument(
        "--area_pen_ss",
        type=float,
        default=0.0,
        help="Value for area penalty parameter in simple_stirap",
    )
    parser.add_argument(
        "--smoothness_pen_ss",
        type=float,
        default=0.001,
        help="Value for smoothness penalty in simple_stirap",
    )
    parser.add_argument(
        "--smoothness_pen_ss_det",
        type=float,
        default=0.001,
        help="Value for smoothness penalty in simple_stirap",
    )

    parser.add_argument(
        "--fix_endpoints_ss",
        type=int,
        default=1,
        help="Whether to fix endpoints in simple_stirap",
    )
    parser.add_argument(
        "--smoothness_calc_amp",
        type=str,
        default="second_derivative",
        help="Method to calculate smoothness for amplitude",
    )
    parser.add_argument(
        "--smoothness_calc_det",
        type=str,
        default="second_derivative",
        help="Method to calculate smoothness for detuning",
    )
    parser.add_argument(
        "--smoothness_cutoff_freq",
        type=float,
        default=5.0,
        help="Method to penalize smoothness for amplitude",
    )
    parser.add_argument(
        "--log_fidelity",
        type=int,
        default=0,
        help="Whether to use log fidelity",
    )
    parser.add_argument(
        "--kernel_std_amp",
        type=float,
        default=4.0,
        help="Kernel std for amplitude",
    )

    parser.add_argument(
        "--kernel_std_freq",
        type=float,
        default=4.0,
        help="Kernel std for detuning",
    )

    #RYDBERG PARAMS
    parser.add_argument(
        "--blockade_strength",
        type=float,
        default=500,
        help="Value for blockade strength in rydberg sim",
    )
    parser.add_argument(
        "--const_freq_pump_rydberg_two",
        type=int,
        default=0,
        help="Whether to use constant frequency for pump in rydberg_two",
    )
    parser.add_argument(
        "--const_amp_stokes_rydberg_two",
        type=int,
        default=0,
        help="Whether to use constant stokes amp for rydberg_two",
    )

    #MULTI STIRAP PARAMS
    parser.add_argument(
        "--n_sections_multi",
        type=int,
        default=1,
        help="How many sections to split the RL action in",
    )

    #whether to use mask for multi stirap
    parser.add_argument(
        "--multi_use_beta",
        type=int,
        default=0,
        help="Whether to use a beta distribution for sampling noise mu",
    )

    #MX_STEPS PARAMS
    parser.add_argument(
        "--mxstep_solver",
        type=int,
        default=1000,
        help=
        "Maximum number of steps for the solver to use per timestep to evaluate the ODE",
    )
    parser.add_argument(
        "--mx_step_penalty",
        type=float,
        default=-10.0,
        help="Penalty for exceeding the maximum number of steps",
    )

    args = parser.parse_args()
    config = {
        # Replay Buffer
        "BUFFER_SIZE": 100000,  # Reasonable size for 512k total steps
        "BATCH_SIZE": 256,  # Standard batch size for stability

        # Training Schedule
        "LEARNING_STARTS":
        1000,  # Warm-up period (1.28M steps with NUM_ENVS=256)
        "NUM_ENVS": 256,  # Your vectorized setup
        "NUM_STEPS": 1,  # Single-step episodes (adjust if multi-step needed)
        "NUM_UPDATES": 30000,  # Total updates (1.024M steps)

        # Exploration and Noise
        "EXPLORATION_NOISE": 0.15,  # Initial exploration noise
        "POLICY_NOISE": 0.2,  # Target policy smoothing noise
        "NOISE_CLIP": 0.5,  # Clipping for target noise

        # Learning Rates
        "LR_ACTOR": 3e-4,  # Slightly higher for faster policy adjustment
        "LR_CRITIC": 3e-4,  # Matches CleanRL default
        "ANNEAL_LR": False,  # Gradual decay to stabilize late training

        # TD3-Specific
        "POLICY_FREQ": 1,  # Delayed actor updates
        "TAU": 0.01,  # Soft target update rate
        "GAMMA": 0.99,  # Discount factor

        # Network and Optimization
        "MAX_GRAD_NORM": 0.5,  # Gradient clipping
        "ACTIVATION": "relu6",  # Matches CleanRL, good default
        "LAYER_SIZE": 256,  # Sufficient for simple_stirap

        # Environment and Logging
        "ENV_NAME": "simple_stirap",
        "REWARD_SCALE": 20.0,  # Amplify reward signal
        "DEBUG": True,
        "DEBUG_NOJIT": False,  # Enable JIT for speed
        "LOGGING": True,
        "LOG_FREQ": 10,  # Log every 10 updates
        "LOG_WAND": True,
        "LOG_WAND": False,
        "LOCAL_LOGGING": True,
        "LOCAL_SAVE_NAME": "ddpg",
        "MU_PHASE": 0.,
        "MU_AMP": 0.,
        "ALPHA_PHASE": 0.1,
        "ALPHA_AMP": 0.1,
        "SIGMA_PHASE": 0.,
        "SIGMA_AMP": 0.,
    }

    if config["ENV_NAME"] == "simple_stirap":
        config["ENV_PARAMS"] = get_multi_stirap_params(
            args)  # get_simple_stirap_params(args)

    elif config["ENV_NAME"] == "rydberg":
        config["ENV_PARAMS"] = get_rydberg_cz_params(args)
        config["LR"] = 8e-4

    elif config["ENV_NAME"] == "multi_stirap":
        config["NUM_STEPS"] = args.n_sections_multi
        config[
            "MINIBATCH_SIZE"] = config["MINIBATCH_SIZE"] * config["NUM_STEPS"]

        config["ENV_PARAMS"] = get_multi_stirap_params(args)
        config["ENV_PARAMS"]["n_sections"] = args.n_sections_multi
        config["ENV_PARAMS"]["n_action_steps"] = 48
        config["ENV_PARAMS"]["use_mu_beta"] = args.multi_use_beta

    elif config["ENV_NAME"] == "rydberg_two":
        config["ENV_PARAMS"] = get_rydberg_two_params(args)

    elif config["ENV_NAME"] == "transmon_reset":
        config["ENV_PARAMS"] = get_transmon_reset_params(args)
        print(config["ENV_PARAMS"])

    else:
        raise ValueError("Environment not recognized")

    config["ENV_PARAMS"]["ou_noise_params"] = [
        config["MU_PHASE"],
        config["MU_AMP"],
        config["ALPHA_PHASE"],
        config["ALPHA_AMP"],
        config["SIGMA_PHASE"],
        config["SIGMA_AMP"],
    ]

    if config["DEBUG_NOJIT"]:
        jax.disable_jit(disable=True)

    # config["NUM_ENVS"] = args.num_envs

    # assert (config["NUM_MINIBATCHES"] *
    #         config["MINIBATCH_SIZE"] == config["NUM_STEPS"] *
    #         config["NUM_ENVS"])

    seed = 20

    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)

    single_train = jit(td3_make_train(config))

    print(f"Starting a Run of {config['NUM_UPDATES']} Updates")

    #ADD YOUR OWN WAND CONFIG HERE
    if config["LOG_WAND"]:
        wandb.init(project="", entity="", config=config)

    outs = jax.block_until_ready(single_train(rng))
