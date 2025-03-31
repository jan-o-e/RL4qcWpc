from jax.lib import xla_bridge

print("Platform used: ")
print(xla_bridge.get_backend().platform)

import time
import jax
import wandb
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.getcwd(), ".."))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

sys.path.append(os.path.join(os.path.dirname(__file__), 'envs'))
print(jax.devices())
jax.config.update("jax_default_device", jax.devices()[0])
print("JaX backend: ", jax.default_backend())

import jax.numpy as jnp
from jax import jit, config, vmap, block_until_ready
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import distrax
import chex
import pickle

from rl_working.envs.utils.wrappers import VecEnv, LogWrapper
from rl_working.envs.single_rydberg_env import RydbergEnv
from rl_working.envs.single_stirap_env import SimpleStirap
from rl_working.envs.multistep_stirap_env import MultiStirap
from rl_working.envs.single_rydberg_two_photon_env import RydbergTwoEnv
from rl_working.envs.single_transmon_reset_env import TransmonResetEnv

from rl_working.env_configs.configs import get_plot_elem_names, get_simple_stirap_params, get_multi_stirap_params, get_rydberg_cz_params, get_rydberg_two_params, get_transmon_reset_params
import matplotlib.pyplot as plt
import argparse

# Check if GPU is available
if "cuda" in str(jax.devices()):
    print("Connected to a GPU")
    processor = "gpu"
    default_dtype = jnp.float32
else:
    jax.config.update("jax_platform_name", "cpu")
    print("Not connected to a GPU")
    jax.config.update("jax_enable_x64", True)
    processor = "cpu"
    default_dtype = jnp.float64

envs_class_dict = {
    "simple_stirap": SimpleStirap,
    "multi_stirap": MultiStirap,
    "rydberg": RydbergEnv,
    "rydberg_two": RydbergTwoEnv,
    "transmon_reset": TransmonResetEnv,
}


class SeparateActorCritic(nn.Module):
    """
    Actor and Critic with Separate Feed-forward Neural Networks
    """

    action_dim: Sequence[int]
    activation: str = "tanh"
    layer_size: int = 128

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        if self.activation == "elu":
            activation = nn.elu
        if self.activation == "leaky_relu":
            activation = nn.leaky_relu
        if self.activation == "relu6":
            activation = nn.relu6
        if self.activation == "selu":
            activation = nn.selu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(self.layer_size,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.layer_size,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.action_dim,
                              kernel_init=orthogonal(0.01),
                              bias_init=constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros,
                                   (self.action_dim, ))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        critic = nn.Dense(self.layer_size,
                          kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(x)
        critic = activation(critic)
        critic = nn.Dense(self.layer_size,
                          kernel_init=orthogonal(np.sqrt(2)),
                          bias_init=constant(0.0))(critic)
        critic = activation(critic)
        critic = nn.Dense(1,
                          kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(critic)

        return pi, jnp.squeeze(critic, axis=-1)


class CombinedActorCritic(nn.Module):
    """
    Actor and Critic Class with combined Feed-forward Neural Network
    """

    action_dim: Sequence[int]
    activation: str = "tanh"
    layer_size: int = 128

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        if self.activation == "elu":
            activation = nn.elu
        if self.activation == "leaky_relu":
            activation = nn.leaky_relu
        if self.activation == "relu6":
            activation = nn.relu6
        if self.activation == "selu":
            activation = nn.selu
        else:
            activation = nn.tanh
        actor_mean = nn.Dense(self.layer_size,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(self.layer_size,
                              kernel_init=orthogonal(np.sqrt(2)),
                              bias_init=constant(0.0))(actor_mean)
        actor_mean = activation(actor_mean)
        # print(f"action_dim_type:{type(self.action_dim)}")
        actor_mean_val = nn.Dense(self.action_dim,
                                  kernel_init=orthogonal(0.01),
                                  bias_init=constant(0.0))(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros,
                                   (self.action_dim, ))
        pi = distrax.MultivariateNormalDiag(actor_mean_val,
                                            jnp.exp(actor_logtstd))

        critic = nn.Dense(1,
                          kernel_init=orthogonal(1.0),
                          bias_init=constant(0.0))(actor_mean)

        return pi, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    """
    Class for carrying RL State between processes
    """

    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def PPO_make_train(config):
    """
    Function that returns a trainable function for an input configuration dictionary
    """
    env = envs_class_dict[config["ENV_NAME"]](**config["ENV_PARAMS"])
    env = LogWrapper(env)
    env = VecEnv(env)
    env_params = env.default_params

    def linear_schedule(count):
        frac = (1.0 - (count //
                       (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) /
                config["NUM_UPDATES"])
        return config["LR"] * frac

    def train(rng: chex.PRNGKey):

        network = CombinedActorCritic(
            env.action_space(env_params).shape[0],
            activation=config["ACTIVATION"],
            layer_size=config["LAYER_SIZE"],
        )
        rng, _rng = jax.random.split(rng)

        init_x = jnp.zeros(env.observation_space(env_params).shape)

        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)

        start_time = time.time()

        step = 0

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, step, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params)
                transition = Transition(done, action, value, reward, log_prob,
                                        last_obs, info)
                runner_state = (train_state, env_state, obsv, step, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state,
                                                    None, config["NUM_STEPS"])

            # Manually reset the environment after the episode has finished (assuming 1 episode is of length NUM_STEPS)
            train_state, env_state, obsv, step, rng = runner_state
            rng, _rng = jax.random.split(rng)
            reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
            obsv, env_state = env.reset(reset_rng, env_params)

            step = step + 1
            #timestep = config["NUM_ENVS"] * config["NUM_STEPS"] * step

            runner_state = (train_state, env_state, obsv, step, rng)
            '''
            if config.get("DEBUG"):

                def callback2():
                    speed_1k = (time.time() -
                                  start_time) / timestep.astype(int) * 1e6

                    print(f"time per 100k steps: {speed_1k} seconds")

                jax.debug.callback(callback2)
            '''

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, step, rng = runner_state
            _, last_val = network.apply(train_state.params, last_obs)
            step = 0

            def _calculate_gae(traj_batch, last_val):
                last_val = last_val.astype(default_dtype)

                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (
                        1 - done) - value
                    gae = (delta + config["GAMMA"] * config["GAE_LAMBDA"] *
                           (1 - done) * gae)
                    gae = gae.astype(default_dtype)
                    value = value.astype(default_dtype)
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val, dtype=default_dtype), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):

                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value).clip(
                                -config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped -
                                                          targets)
                        value_loss = (0.5 * jnp.maximum(
                            value_losses, value_losses_clipped).mean())

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        ) * gae)
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (loss_actor +
                                      config["VF_COEF"] * value_loss -
                                      config["ENT_COEF"] * entropy)
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch,
                                                advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config[
                    "NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size, ) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] +
                                          list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch,
                                                       train_state,
                                                       minibatches)
                update_state = (train_state, traj_batch, advantages, targets,
                                rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state,
                                                   None,
                                                   config["UPDATE_EPOCHS"])

            train_state = update_state[0]
            metric = traj_batch.info
            global_updatestep = metric["timestep"][0]
            rng = update_state[-1]

            step = runner_state[-2]

            if config.get("LOGGING"):

                def callback(infos):
                    info, loss_info, step = infos
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
                        f"total_loss": jnp.mean(jnp.ravel(loss_info[0])),
                        f"value_loss": jnp.mean(jnp.ravel(loss_info[1][0])),
                        f"actor_loss": jnp.mean(jnp.ravel(loss_info[1][1])),
                        f"entropy": jnp.mean(jnp.ravel(loss_info[1][2])),
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

                jax.debug.callback(callback, (metric, loss_info, step))

            runner_state = (train_state, env_state, last_obs, step, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, step, _rng)
        runner_state, metric = jax.lax.scan(_update_step, runner_state, None,
                                            config["NUM_UPDATES"])
        return {"runner_state": runner_state}  # , "metrics": metric}

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

    assert args.num_envs % args.num_minibatches == 0

    config = {
        "LR": args.lr,
        "NUM_ENVS": args.num_envs,
        "NUM_STEPS": 1,
        "NUM_UPDATES": args.num_updates,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": args.num_minibatches,
        "MINIBATCH_SIZE": args.num_envs // args.num_minibatches,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": args.clip_eps,
        "ENT_COEF": args.ent_coef,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": args.max_grad_norm,
        "ACTIVATION": "relu6",
        "LAYER_SIZE": args.layer_size,
        "ENV_NAME": args.env,
        "ANNEAL_LR": args.anneal_lr,
        "DEBUG": True,
        "DEBUG_NOJIT": False,
        "LOGGING": True,
        "LOG_FREQ": 1,
        "LOG_WAND": False,
        "LOCAL_LOGGING": True,
        "LOCAL_SAVE_NAME": "local_save",
        "MU_PHASE": args.mu_phase,
        "MU_AMP": args.mu_amp,
        "ALPHA_PHASE": 0.1,
        "ALPHA_AMP": 0.1,
        "SIGMA_PHASE": args.sigma_phase,
        "SIGMA_AMP": args.sigma_amp,
    }

    if config["ENV_NAME"] == "simple_stirap":
        config["ENV_PARAMS"] = get_simple_stirap_params(args)

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

    config["NUM_ENVS"] = args.num_envs

    assert (config["NUM_MINIBATCHES"] *
            config["MINIBATCH_SIZE"] == config["NUM_STEPS"] *
            config["NUM_ENVS"])

    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)

    single_train = jit(PPO_make_train(config))

    print(f"Starting a Run of {config['NUM_UPDATES']} Updates")

    #ADD YOUR OWN WAND CONFIG HERE
    if config["LOG_WAND"]:
        wandb.init(project="", entity="", config=config)

    outs = jax.block_until_ready(single_train(rng))
