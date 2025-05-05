# Standard Imports
from typing import Tuple, Optional
import time

# JAX Imports
import jax
import jax.numpy as jnp
from jax import jit, lax, vmap, block_until_ready
import jax.numpy as jnp
from jax.nn import relu
from gymnax.environments import spaces
from flax import struct
import chex

from environment_template import SingleStepEnvironment
from utils.wrappers import VecEnv
from hamiltonian.transmon_reset_hamiltonian import get_hamiltonian_operators
from utils.noise_functions import generate_ou_noise_samples, generate_zero_noise_samples, generate_gaussian_noise_samples
from utils.signal_functions import gen_gauss_kernel, default_small_window, default_kernel

# Check if GPU is available
if "cuda" in str(jax.devices()):
    print("Connected to a GPU")
    processor_array_type = "jax"
    processor = "gpu"
else:
    jax.config.update("jax_platform_name", "cpu")
    print("Not connected to a GPU")
    processor_array_type = "jax_sparse"
    jax.config.update("jax_enable_x64", True)
    processor = "cpu"

# Qiskit Imports
from qiskit_dynamics import Solver, Signal
from diffrax import LinearInterpolation, PIDController, Tsit5, Dopri8, Heun, RESULTS

@struct.dataclass
class EnvState:
    """
    Flax Dataclass used to store Dynamic Environment State
    All relevant params that get updated each step should be stored here
    """

    reward: float
    pulse_reset_transmon: float
    transmon_reset_reward: float
    mean_smooth_waveform_difference: float
    smoothness_reward: float
    pulse_reset_val: float
    amp_reward: float
    num_steps: float
    max_steps_pen: float
    mean_deviation: float
    deviation_reward: float
    max_steps_reached: bool
    noise_amp: chex.Array
    noise_freq: chex.Array
    action: chex.Array
    timestep: int

@struct.dataclass
class EnvParams:
    """
    Flax Dataclass used to store Static Environment Params
    All static env params should be kept here, though they can be equally kept
    in the Jax class as well
    """

    t1: float

    window_length: Optional[int] = 8
    kernel: Optional[chex.Array] = struct.field(default_factory=default_kernel)
    gauss_mean: Optional[float] = 0.0
    gauss_std: Optional[float] = 1.0
    small_window: Optional[chex.Array] = struct.field(
        default_factory=default_small_window)

    t0: Optional[float] = 0.0

    num_actions: Optional[int] = 101
    num_sim: Optional[int] = 2
    num_sim_debug: Optional[int] = 301

    min_action: Optional[float] = -1.0
    max_action: Optional[float] = 1.0

    min_reward: Optional[float] = -1000.0
    max_reward: Optional[float] = 10.0

    min_separation: Optional[float] = 0.0
    max_separation: Optional[float] = 15.0

    min_bandwidth: Optional[float] = 0.0
    max_bandwidth: Optional[float] = 2.0

    min_photon: Optional[float] = 0.0
    max_photon: Optional[float] = 50.0

    min_smoothness: Optional[float] = 0.0
    max_smoothness: Optional[float] = 20.0

    max_steps_in_episode = 1


class TransmonResetEnv(SingleStepEnvironment):
    """
    Jax Compatible Environment for Finding Optimal Transmon Reset Pulses
    with realistic bandwidth constraints
    """

    def __init__(
            self,
            kappa: float,
            chi: float,
            delta: float,
            anharm: float,
            g_coupling: float,
            gamma: float,
            omega_max: float,
            delta_max: float,
            sim_t1: float,
            transmon_reset_coeff: float,
            deviation_coeff: float,
            smoothness_coeff: float,
            amp_pen_coeff: float,
            steps_pen_coeff: float,
            max_grad: float,
            k_factor: float,
            max_deviation: float,
            max_steps: int,
            gauss_std: float,
            noise: str = "None",  #None or ou or g
            gaussian_noise_scaling: jnp.ndarray = [0.01,
                                                   0.01],  # phase, amplitude
            ou_noise_params: jnp.
        ndarray = [
            0,
            0,
            0.1,
            0.1,
            0.5,
            0.5,
        ],  # mu_freq, mu_amp, alpha_freq, alpha_amp, sigma_freq, sigma_amp)
    ):
        #super().__init__()

        assert noise in ["ou", "g", "None"], "Noise must be either 'ou' or 'g'"

        self.int_dtype = jnp.int16
        # dtype
        if processor == "gpu":
            self.float_dtype = jnp.float64
            self.complex_dtype = jnp.complex128
        else:
            self.float_dtype = jnp.float64
            self.complex_dtype = jnp.complex128

        t_pi = 2. * jnp.pi
        self._kappa = kappa
        self._tau = 1. / self._kappa
        self._chi = chi * t_pi
        self._delta = delta * t_pi
        self._anharm = anharm * t_pi
        self._g_coupling = g_coupling * t_pi
        self._gamma = gamma
        self._omega_max = omega_max * t_pi
        self._delta_max = delta_max * t_pi
        self._k_factor = k_factor

        self._g_bar_factor = 1. / jnp.sqrt(2) * self._g_coupling * (
            self._anharm) / (self._delta * (self._delta + self._anharm))

        self._n_trans = 3
        self._n_res = 2

        self.photon_limit = 1.

        self._t1 = sim_t1

        self.gauss_std = gauss_std
        params = self.default_params

        self.len_ts_sim = params.num_sim
        self.len_ts_sim_debug = params.num_sim_debug
        self.len_ts_action = params.num_actions

        self.ts_sim = jnp.linspace(0.0,
                                   self._t1,
                                   self.len_ts_sim,
                                   dtype=self.float_dtype)
        self.ts_sim_debug = jnp.linspace(0.0,
                                         self._t1,
                                         self.len_ts_sim_debug,
                                         dtype=self.float_dtype)
        self.ts_action = jnp.linspace(0.0,
                                      self._t1,
                                      self.len_ts_action,
                                      dtype=self.float_dtype)

        self.noise = noise
        self.ou_noise_params = ou_noise_params
        self.gaussian_noise_scaling_amp = gaussian_noise_scaling[1]
        self.gaussian_noise_scaling_phase = gaussian_noise_scaling[0]
        self.noise_over_sampling_rate = 1

        self.log_vals = {
            "fid":
            0.0,
            "transmon-reset-reward":
            0.0,
            "mean-smooth-waveform-difference":
            0.0,
            "smoothness-reward":
            0.0,
            "pulse-reset-val":
            0.0,
            "amp-reward":
            0.0,
            "amp":
            jnp.zeros(self.len_ts_action),
            "freq":
            jnp.zeros(self.len_ts_action),
            "max-steps-pen":
            0.0,
            "mean-deviation":
            0.0,
            "deviation-reward":
            0.0,
            "max-steps-reached":
            0,
            "noise-amp":
            jnp.zeros(self.len_ts_action * self.noise_over_sampling_rate),
            "noise-freq":
            jnp.zeros(self.len_ts_action * self.noise_over_sampling_rate),
            "num-steps":
            0,
        }

        self.ode_solver = Dopri8()
        self.pid_controller = PIDController(pcoeff=0.4,
                                            icoeff=0.3,
                                            dcoeff=0.,
                                            rtol=1e-6,
                                            atol=1e-8,
                                            jump_ts=self.ts_action)
        self.max_steps = max_steps

        self.H_STATIC, self.H_DISSIPATE, self.H_DRIVE, self.transmon_num_op = get_hamiltonian_operators(
            transmon_dim=self._n_trans,
            res_dim=self._n_res,
            chi=self._chi,
            kappa=self._kappa,
            gamma=self._gamma,
            g_bar_factor=self._g_bar_factor,
            _dtype=self.complex_dtype)

        # initial state is |f0>
        f_ket = jnp.zeros(self._n_trans, dtype=self.complex_dtype)
        f_ket = f_ket.at[2].set(1.)

        res_ket = jnp.zeros(self._n_res, dtype=self.complex_dtype)
        res_ket = res_ket.at[0].set(1.)

        f0_ket = jnp.kron(f_ket, res_ket)
        self.f0_dm = jnp.outer(f0_ket, f0_ket)

        self.transmon_reset_coeff = transmon_reset_coeff
        self.deviation_coeff = deviation_coeff
        self.smoothness_coeff = smoothness_coeff
        self.amp_pen_coeff = amp_pen_coeff
        self.steps_pen_coeff = steps_pen_coeff

        self.kernel = gen_gauss_kernel(window_length=8,
                                       gauss_mean=0.,
                                       gauss_std=self.gauss_std)

        self.pulse_dt = self.ts_action[1] - self.ts_action[0]
        self.max_grad = max_grad

        self.solver = Solver(
            static_hamiltonian=self.H_STATIC,
            hamiltonian_operators=self.H_DRIVE,
            static_dissipators=self.H_DISSIPATE,
            rotating_frame=self.H_STATIC,
            validate=False,
            array_library=processor_array_type,
        )

        self.opt_time = self.determine_theoretical_duration()
        self.ref_waveform = jnp.heaviside(self.opt_time - self.ts_action,
                                          1.).astype(self.float_dtype)

        self.max_deviation = max_deviation

    def determine_theoretical_duration(self):
        max_amp_waveform = jnp.heaviside(0.4 - self.ts_action, 1.).astype(
            self.float_dtype) * self._omega_max
        detuning_waveform = jnp.zeros_like(
            max_amp_waveform, dtype=self.float_dtype) * self._delta_max

        old_k_factor = self._k_factor

        # set k value to zero
        self._k_factor = 0.

        results, steps = self.calc_results_debug(max_amp_waveform,
                                                 detuning_waveform)
        transmon_pop = jnp.trace(results @ self.transmon_num_op,
                                 axis1=1,
                                 axis2=2).real
        time_of_min = self.ts_sim_debug[jnp.argmin(transmon_pop)]
        self._k_factor = old_k_factor
        return time_of_min

    @property
    def default_params(self) -> EnvParams:
        """
        IMPORTANT Retrieving the Default Env Params
        """
        return EnvParams(
            t1=self._t1,
            min_action=-1.,
            max_action=1.,
            gauss_std=self.gauss_std,
        )

    def drive_smoother(self, res_drive: chex.Array):
        """Physics Specific Function"""
        conv_result = jnp.convolve(res_drive, self.kernel, mode="same")
        return conv_result

    def normalize_pulse(self, res_drive: chex.Array):
        normalizing_factor = jnp.clip(
            1. / jnp.absolute(res_drive),
            0.0,
            1.0,
        )
        return res_drive * normalizing_factor

    def limit_gradient(self, res_drive):
        ref_waveform = res_drive
        pulse_dt = self.pulse_dt
        max_grad = self.max_grad  # for t1=0.8 using a gaussian square as used in IBM

        output_waveform = jnp.zeros_like(ref_waveform)

        def update_array_at_index(ind, waveform):
            diff = ref_waveform[ind] - waveform[ind - 1]
            diff_clipped = jnp.clip(diff,
                                    a_min=-pulse_dt * max_grad,
                                    a_max=pulse_dt * max_grad)
            waveform = waveform.at[ind].set(waveform[ind - 1] + diff_clipped)
            return waveform

        output_waveform = jax.lax.fori_loop(lower=1,
                                            upper=len(ref_waveform),
                                            body_fun=update_array_at_index,
                                            init_val=output_waveform)

        return output_waveform

    def prepare_trans_action(self, trans_amp, trans_detuning):
        trans_drive = trans_amp.astype(self.float_dtype)  # Scale up Action
        trans_normed_drive = self.normalize_pulse(trans_drive)
        trans_clipped_drive = jnp.clip(trans_normed_drive, a_min=0., a_max=1.)
        trans_drive = self.drive_smoother(
            trans_clipped_drive)  # Apply Smoother
        # res_drive = self.limit_gradient(res_drive) # Apply Instantaneous Gradient Limits

        trans_detuning = trans_detuning.astype(self.float_dtype)
        trans_normed_detuning = self.normalize_pulse(trans_detuning)
        trans_detuning = self.drive_smoother(trans_normed_detuning)

        # Calculate Predicted Stark Shift
        mean_deviation = jnp.mean(jnp.abs(trans_drive - self.ref_waveform))
        trans_drive *= jnp.heaviside(self.max_deviation - mean_deviation,
                                     1.).astype(self.float_dtype)

        return self._omega_max * trans_drive, self._delta_max * trans_detuning, mean_deviation

    def step_env(
        self, key: chex.PRNGKey, state: EnvState, action: chex.Array,
        params: EnvParams
    ) -> Tuple[chex.Array, EnvState, chex.Array, bool, dict]:
        """
        IMPORTANT Perform Single Episode State Transition
        - key is for RNG, needs to be handled properly if used
        - state is the input state, will be modified to produce new state
        - action is an array corresponding to action space shape
        - params is the appropriate Env Params, this argument shouldn't change during training runs

        Returns Observation, New State, Reward, Dones Signal, and Info based on State
        In this particular task, the observation is always fixed, and the Dones is
        always True since its a single-step environment.
        """
        new_timestep = state.timestep + 1

        action_amp = action[:self.len_ts_action]
        action_detuning = action[self.len_ts_action:]

        # Preparing Action for Simulation
        trans_drive, trans_detuning, mean_deviation = self.prepare_trans_action(
            action_amp, action_detuning)

        rng, _rng = jax.random.split(key)

        # Simulation and Obtaining Reward + Params for New State
        result, solver_steps, mx_steps_reached, noise_env, noise_chirp = self.calc_results(
            rng, trans_drive, trans_detuning)

        reward, updated_state_array = self.calc_reward_and_state(
            _rng,
            result.astype(self.float_dtype),
            solver_steps,
            mean_deviation,
            action_amp,
            action_detuning,
        )

        env_state = EnvState(*updated_state_array, mx_steps_reached, noise_env,
                             noise_chirp, action, new_timestep)

        done = True
        return (
            lax.stop_gradient(self.get_obs()),
            lax.stop_gradient(env_state),
            reward,
            done,
            lax.stop_gradient(self.get_info(env_state)),
        )

    def extract_values(
        self,
        key: chex.PRNGKey,
        results: chex.Array,
        steps: int,
        mean_deviation: float,
        action_amp: chex.Array,
        action_detuning: chex.Array,
    ):
        """Physics Specific Function"""
        # rng, _rng = jax.random.split(key)

        transmon_pop = jnp.trace(results @ self.transmon_num_op,
                                 axis1=1,
                                 axis2=2).real

        # End Transmon Pop
        pulse_reset_transmon = transmon_pop[-1]  # Transmon Pop at end of sim

        # Processing Raw Action
        raw_action_amp = self.normalize_pulse(action_amp)
        clipped_action_amp = jnp.clip(raw_action_amp, a_min=0., a_max=1.)
        smooth_action_amp = self.drive_smoother(clipped_action_amp)

        raw_action_detuning = self.normalize_pulse(action_detuning)
        smooth_action_detuning = self.drive_smoother(raw_action_detuning)

        pulse_reset_val = jnp.abs(smooth_action_amp[-1]) + jnp.abs(
            smooth_action_amp[0])

        mean_smooth_waveform_difference = 0.5 * jnp.sum(
            jnp.abs(smooth_action_amp - raw_action_amp)) / self._t1
        mean_smooth_waveform_difference += 0.5 * jnp.sum(
            jnp.abs(smooth_action_detuning - raw_action_detuning)) / self._t1

        def valid_steps():
            return jnp.array([
                mean_smooth_waveform_difference, pulse_reset_val,
                pulse_reset_transmon, 0.
            ])

        def invalid_steps():
            return jnp.array([100., 2., 2., 1.])

        reward_calc_vals = jax.lax.select(
            (steps < self.max_steps) * (mean_deviation < self.max_deviation),
            valid_steps(), invalid_steps())

        return reward_calc_vals

    def calc_results(
        self, key: chex.PRNGKey, trans_drive: chex.Array,
        trans_detuning: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Physics Specific Function, Function used for ODE Simulation"""
        params = self.default_params

        rng, rng_noise = jax.random.split(key)

        f0g1_control = LinearInterpolation(ts=self.ts_action, ys=trans_drive)
        stark_control = LinearInterpolation(
            ts=self.ts_action,
            ys=self._k_factor * (trans_drive**2 - self._omega_max**2) +
            trans_detuning)

        if self.noise == "ou":
            (ou_noise_chirp, ou_noise_env, noise_samples_env,
             noise_samples_chirp) = generate_ou_noise_samples(
                 rng_noise,
                 self.len_ts_action,
                 self.noise_over_sampling_rate,
                 self.ou_noise_params,
                 self.ts_action[-1],
                 self.float_dtype,
                 n_signals=1)

        elif self.noise == "g":
            (g_noise_env, g_noise_det, noise_samples_env,
             noise_samples_chirp) = generate_gaussian_noise_samples(
                 rng_noise,
                 self.len_ts_action,
                 self._omega_max,
                 self._omega_max,
                 self._delta_max,
                 self._delta_max,
                 self.gaussian_noise_scaling_amp,
                 self.gaussian_noise_scaling_phase,
                 n_signals=1)
        else:
            (
                noise_samples_env,
                noise_samples_chirp,
            ) = generate_zero_noise_samples(
                self.len_ts_action,
                self.noise_over_sampling_rate,
                _dtype=self.float_dtype,
                n_signals=1,
            )

        def get_f0g1(t):
            if self.noise == "g":
                return jnp.abs(f0g1_control.evaluate(t) + g_noise_env)
            elif self.noise == "ou":
                return f0g1_control.evaluate(t) + ou_noise_env.evaluate(t)
            else:
                return f0g1_control.evaluate(t)

        def get_stark(t):
            if self.noise == "g":
                return jnp.abs(stark_control.evaluate(t) + g_noise_det)
            elif self.noise == "ou":
                return stark_control.evaluate(t) + ou_noise_chirp.evaluate(t)
            else:
                return stark_control.evaluate(t)

        vec_get_f0g1 = jnp.vectorize(get_f0g1)
        vec_get_stark = jnp.vectorize(get_stark)

        signals = [
            Signal(vec_get_f0g1),
            Signal(vec_get_stark),
        ]

        sol = self.solver.solve(t_span=[0., self.ts_sim[-1]],
                                signals=signals,
                                y0=self.f0_dm,
                                t_eval=self.ts_sim,
                                convert_results=True,
                                method=self.ode_solver,
                                stepsize_controller=self.pid_controller,
                                max_steps=self.max_steps,
                                throw=False)

        noise_samples_env = noise_samples_env.astype(
            self.float_dtype)
        noise_samples_chirp = noise_samples_chirp.astype(
            self.float_dtype)

        return sol.y, sol.stats[
            "num_steps"], sol.result == RESULTS.max_steps_reached, noise_samples_env, noise_samples_chirp

    def calc_results_debug(
        self, trans_drive: chex.Array, trans_detuning: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Physics Specific Function, Function used for ODE Simulation"""
        params = self.default_params

        f0g1_control = LinearInterpolation(ts=self.ts_action, ys=trans_drive)
        stark_control = LinearInterpolation(
            ts=self.ts_action,
            ys=self._k_factor * (trans_drive**2 - self._omega_max**2) +
            trans_detuning)

        def get_f0g1(t):
            return f0g1_control.evaluate(t).astype(self.float_dtype)

        def get_stark(t):
            return stark_control.evaluate(t).astype(self.float_dtype)

        vec_get_f0g1 = jnp.vectorize(get_f0g1)
        vec_get_stark = jnp.vectorize(get_stark)

        signals = [
            Signal(vec_get_f0g1),
            Signal(vec_get_stark),
        ]

        sol = self.solver.solve(t_span=[0., self.ts_sim[-1]],
                                signals=signals,
                                y0=self.f0_dm,
                                t_eval=self.ts_sim_debug,
                                convert_results=False,
                                method=self.ode_solver,
                                stepsize_controller=self.pid_controller,
                                max_steps=self.max_steps,
                                throw=False)

        return sol.y, sol.stats["num_steps"]

    def calc_reward_and_state(
        self,
        key: chex.PRNGKey,
        result: chex.Array,
        steps: int,
        mean_deviation: float,
        action: chex.Array,
        detuning: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Function holding Reward Calculation and State Param Calculations"""
        rng, _rng = jax.random.split(key)
        (
            mean_smooth_waveform_difference,
            pulse_reset_val,
            pulse_reset_transmon,
            max_steps_flag,
        ) = self.extract_values(_rng, result, steps, mean_deviation, action,
                                detuning)
        # The above function holds physics-specific details

        reward = (+self.transmon_reward(pulse_reset_transmon) +
                  self.smoothness_reward(mean_smooth_waveform_difference) +
                  self.amp_reward(pulse_reset_val) +
                  self.deviation_reward(mean_deviation) -
                  self.max_steps_pen(max_steps_flag))

        state = jnp.array(
            [
                reward, pulse_reset_transmon,
                self.transmon_reward(pulse_reset_transmon),
                mean_smooth_waveform_difference,
                self.smoothness_reward(mean_smooth_waveform_difference),
                pulse_reset_val,
                self.amp_reward(pulse_reset_val), steps,
                -self.max_steps_pen(max_steps_flag), mean_deviation,
                self.deviation_reward(mean_deviation)
            ],
            dtype=self.float_dtype,
        )

        return (reward, state)

    def transmon_reward(self, pulse_reset_transmon):
        return -self.transmon_reset_coeff * jnp.log10(pulse_reset_transmon)

    def smoothness_reward(self, mean_smooth_waveform_difference):
        return -self.smoothness_coeff * mean_smooth_waveform_difference

    def amp_reward(self, pulse_reset_val):
        return -self.amp_pen_coeff * pulse_reset_val

    def max_steps_pen(self, max_steps_flag):
        return self.steps_pen_coeff * max_steps_flag

    def deviation_reward(self, mean_deviation):
        return -self.deviation_coeff * relu(mean_deviation -
                                            self.max_deviation)

    def reset_env(self, key: chex.PRNGKey,
                  params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """IMPORTANT Reset Environment, in this case nothing needs to be done
        so default obs and info are returned"""
        # self.precompile()
        state = EnvState(
            reward=0.,
            pulse_reset_transmon=0.,
            transmon_reset_reward=0.,
            mean_smooth_waveform_difference=0.,
            smoothness_reward=0.,
            pulse_reset_val=0.,
            amp_reward=0.,
            num_steps=0.,
            max_steps_pen=0.,
            mean_deviation=0.,
            deviation_reward=0.,
            max_steps_reached=False,
            noise_amp=jnp.zeros(self.len_ts_action *
                                self.noise_over_sampling_rate),
            noise_freq=jnp.zeros(self.len_ts_action *
                                 self.noise_over_sampling_rate),
            action=jnp.zeros(2 * self.len_ts_action, dtype=self.float_dtype),
            timestep=0,
        )

        return self.get_obs(params), state

    def get_obs(self, params: Optional[EnvParams] = EnvParams) -> chex.Array:
        """IMPORTANT Function to get observation at a given state, as this is a single-step
        episode environment, the observation can be left fixed"""
        return jnp.zeros((1, ), dtype=self.float_dtype)

    def get_info(self, env_state: EnvState) -> dict:
        """IMPORTANT Function to get info for a given input state"""

        info_dict = {
            "reward": env_state.reward,
            "fid": 1 - env_state.pulse_reset_transmon/2,
            "transmon-reset-reward": env_state.transmon_reset_reward,
            "mean-smooth-waveform-difference":
            env_state.mean_smooth_waveform_difference,
            "smoothness-reward": env_state.smoothness_reward,
            "pulse-reset-val": env_state.pulse_reset_val,
            "amp-reward": env_state.amp_reward,
            "amp": env_state.action[:self.len_ts_action],
            "freq": env_state.action[self.len_ts_action:],
            "num-steps": env_state.num_steps,
            "max-steps-pen": env_state.max_steps_pen,
            "mean-deviation": env_state.mean_deviation,
            "deviation-reward": env_state.deviation_reward,
            "max-steps-reached": env_state.max_steps_reached,
            "noise-amp": env_state.noise_amp,
            "noise-freq": env_state.noise_freq,
            "timestep": env_state.timestep,
        }

        assert set(info_dict.keys()) == set(
            ["reward", "timestep"] + list(self.log_vals.keys())
        ), "Info Dict Keys Mismatch -- this is necessary for logging wrapper to work properly"

        return info_dict

    @property
    def name(self) -> str:
        """IMPORTANT name of environment"""
        return "TransmonResetEnv"

    @property
    def num_actions(self, params: Optional[EnvParams] = EnvParams) -> int:
        """IMPORTANT number of actions"""
        return 2 * params.num_actions

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """IMPORTANT action space shape"""
        if params is None:
            params = self.default_params

        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(2 * params.num_actions, ),
            dtype=self.float_dtype,
        )

    def observation_space(self,
                          params: Optional[EnvParams] = None) -> spaces.Box:
        """IMPORTANT observation space shape"""
        return spaces.Box(-1.0, 1.0, shape=(1, ), dtype=self.float_dtype)


if __name__ == "__main__":
    rng = jax.random.PRNGKey(seed=30)

    ### INIT ENV ###

    transmon_reset_env_config = {
        "kappa": 5.,  # 4.25 initially
        "chi": -0.335,
        "delta": 1934,
        "anharm": -330.,
        "g_coupling": 67.,
        "gamma": 1 / 500.,
        "omega_max": 330.,
        "delta_max": 10.,
        "sim_t1": 0.20,
        "transmon_reset_coeff": 10.,
        "smoothness_coeff": 0.,
        "amp_pen_coeff": 0.,
        "deviation_coeff": 50.,
        "steps_pen_coeff": 20.,
        "max_grad": 43.,
        "k_factor": -0.001,
        "gauss_std":3.0,
        "max_deviation": 0.2,
        "max_steps": 4096,
        "noise": "ou",
    }

    env = TransmonResetEnv(**transmon_reset_env_config)

    import warnings
    warnings.filterwarnings('ignore')

    rng = jax.random.PRNGKey(10)

    zero_detuning = jnp.zeros_like(env.ts_action)
    one_drive = jnp.ones_like(env.ts_action) * jnp.heaviside(
        env.opt_time - env.ts_action, 1.)

    prepped_one_drive, prepped_one_detuning, one_stark_shift = env.prepare_trans_action(
        one_drive, zero_detuning)

    photon_one_pop, one_steps, mx_steps_reached, noise_a, noise_freq = env.calc_results(
        rng, prepped_one_drive, prepped_one_detuning)
    photon_perfect_pop, perfect_steps, mx_steps_reached, noise_a, noise_freq = env.calc_results(
        rng, one_drive * env._omega_max, jnp.zeros_like(one_drive))

    one_reward, one_state = env.calc_reward_and_state(rng, photon_one_pop,
                                                      one_steps,
                                                      one_stark_shift,
                                                      one_drive, zero_detuning)
    perfect_reward, perfect_state = env.calc_reward_and_state(
        rng, photon_perfect_pop, perfect_steps,
        jnp.zeros_like(one_stark_shift), one_drive, zero_detuning)

    one_env_state = EnvState(*one_state,
                             mx_steps_reached,
                             noise_a,
                             noise_freq,
                             action=jnp.concatenate(
                                 (one_drive, zero_detuning)),
                             timestep=0)
    perfect_env_state = EnvState(*perfect_state,
                                 mx_steps_reached,
                                 noise_a,
                                 noise_freq,
                                 action=jnp.concatenate(
                                     (one_drive, zero_detuning)),
                                 timestep=0)

    print(f"Theoretical Optimal Reset Duration: {env.opt_time}us")
    # For this set of params should be 0.163 us

    print(one_env_state)
    print(perfect_env_state)

    print(
        f"Theoretical Optimal Transmon Reset Val: {perfect_env_state.pulse_reset_transmon}"
    )
    print(
        f"Realistic Square Transmon Reset Val: {one_env_state.pulse_reset_transmon}"
    )
    # Should be 0.00036623 and 0.01099422 respectively

    # Seeding
    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    env_params = env.default_params
    init_obs, init_state = env.reset(_rng, env_params)
    input_action = jax.random.uniform(_rng, (2 * env.len_ts_action, ),
                                      minval=-1.0,
                                      maxval=1.0)
    jit_step = jax.jit(env.step)
    obs, state, reward, done, info = jit_step(_rng, init_state, input_action,
                                              env_params)
    start = time.time()
    obs, state, reward, done, info = block_until_ready(
        jit_step(_rng, init_state, input_action, env_params))
    end = time.time()
    print(
        f"Time for Single Jitted env.step: {jnp.round((end - start)/1e-6)}us")
    ### Testing vmapped evaluation
    vec_env = VecEnv(env)
    vec_env_params = env.default_params
    ### init env
    timings_noise_free = []
    for num_envs in [
            1, 64, 256, 512, 1024, 2048,
    ]:
        rng, _rng = jax.random.split(rng)
        rng_reset = jax.random.split(_rng, num_envs)
        _, vec_env_state = vec_env.reset(rng_reset, vec_env_params)
        # making the action
        rng, _rng, _rng_act = jax.random.split(rng, 3)
        rng_step = jax.random.split(_rng, num_envs)
        vectorised_action = jax.random.uniform(
            _rng_act, (num_envs, 2 * env.len_ts_action),
            minval=-1.0,
            maxval=1.0)
        jit_vmap_step = jax.jit(vec_env.step)
        returned_obsv, new_env_state, reward, done, info = jit_vmap_step(
            rng_step, vec_env_state, vectorised_action, vec_env_params)
        start = time.time()
        returned_ob, new_env_state, reward, done, info = block_until_ready(
            jit_vmap_step(rng_step, vec_env_state, vectorised_action,
                          vec_env_params))
        end = time.time()
        total_time = end - start
        print(
            f"Time taken for batch of {num_envs} steps in vmap: {total_time*1000}ms"
        )
        print(f"Time taken per env step: {total_time / num_envs *1000}ms")
        timings_noise_free.append(total_time / num_envs * 1000)
