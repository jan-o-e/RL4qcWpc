# Standard Imports
from typing import Tuple, Optional, List
from dataclasses import field
import matplotlib.pyplot as plt
from scipy.signal.windows import blackman
import time

# JAX Imports
import jax
import jax.numpy as jnp
from jax import lax, config, vmap, block_until_ready
import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.integrate import trapezoid
from jax.nn import relu
from gymnax.environments import spaces
from flax import struct
import chex
from diffrax import (backward_hermite_coefficients, CubicInterpolation,
                     LinearInterpolation, SaveAt, AbstractERK,
                     AbstractRungeKutta, Dopri5, Euler, AbstractSolver,
                     PIDController, ConstantStepSize, RESULTS
                     # AbastractAdaptiveSolver
                     )

# Custom Imports
from environment_template import SingleStepEnvironment
from utils.wrappers import VecEnv
from utils.noise_functions import generate_ou_noise_samples, generate_zero_noise_samples, generate_gaussian_noise_samples
from utils.signal_functions import gen_gauss_kernel, SignalToolClass

# Quantum Imports
from utils.shared_quantum_functions import (
    fidelity,
    statevector_to_rho,
    pure_state_fidelity,
    mixed_pure_state_fidelity,
)
from hamiltonian.stirap_hamiltonian import get_hamiltonian_operators

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
from qiskit_dynamics import Solver, DiscreteSignal, Signal
from utils.shared_quantum_functions import fidelity, statevector_to_rho, pure_state_fidelity, mixed_pure_state_fidelity


@struct.dataclass
class EnvState:
    """
    Flax Dataclass used to store Dynamic Environment State
    All relevant params that get updated each step should be stored here
    """

    reward: float
    fid: float
    x_state_pop_avg: float
    area_s: float
    area_p: float
    smoothness_s: float
    smoothness_p: float
    env_s: jnp.ndarray
    env_p: jnp.ndarray
    det_s: jnp.ndarray
    det_p: jnp.ndarray
    num_steps: int
    max_steps_reached: bool
    noise_amp: jnp.ndarray
    noise_freq: jnp.ndarray
    mu_amp_sampled: float
    action_pump_det: jnp.ndarray
    action_stokes_det: jnp.ndarray
    action_pump_env: jnp.ndarray
    action_stokes_env: jnp.ndarray
    timestep: int


@struct.dataclass
class EnvParams:
    """
    Flax Dataclass used to store Static Environment Params
    All static env params should be kept here, though they can be equally kept
    in the Jax class as well
    """

    args: List
    tf: float = 1.0
    min_action: Optional[float] = -1.0
    max_action: Optional[float] = 1.0


class MultiStirap(SingleStepEnvironment):

    def __init__(
            self,
            gamma: float,
            omega_0: float,
            delta_s: float,
            delta_p: float,
            mxstep: int = 4096,
            t_total: float = 1.0,
            x_penalty: float = 0.1,
            fid_factor: float = 1.0,
            log_fidelity: int = 1,
            pulse_area_penalty: float = 0.1,
            smoothness_calc_amp: str = "second_derivative",
            smoothness_calc_det: str = "second_derivative",
            convolution_std_amp: float = 1,
            convolution_std_freq: float = 1,
            smoothness_cutoff_freq: float = 5,
            smoothness_penalty_env: float = 0.1,
            smoothness_penalty_det: float = 0.1,
            mx_step_penalty: float = 0.0,
            fixed_endpoints: int = 0,
            gaussian_kernel_window_length=[15, 15],  # amplitude, frequency
            initial_state: jnp.ndarray = [1, 0],
            final_state: jnp.ndarray = [0, 1],
            n_action_steps: int = 32,
            dissipator: bool = True,
            dim: int = 4,
            x_detuning: float = 20.0,
            noise: str = "g",  #ou or g
            gaussian_noise_scaling: jnp.ndarray = [0.01,
                                                   0.01],  #freq, amplitude
            ou_noise_params: jnp.ndarray = [
                0, 5, 0.1, 0.1, 0, 0
            ],  #mu_freq, mu_amp, alpha_freq, alpha_amp, sigma_freq, sigma_amp
            n_sections=1,
            use_mu_beta=0):
        # Coupling strengths etc... in 2piMHz (angular frequency)
        self._gamma = gamma * 2 * jnp.pi
        self._omega_0 = omega_0 * 2 * jnp.pi
        self._delta_p = delta_p * 2 * jnp.pi
        self._delta_s = delta_s * 2 * jnp.pi
        self.x_detuning = x_detuning * 2 * jnp.pi
        self._resolution = 10
        self._n_steps = n_action_steps
        self.fixed_endpoints = fixed_endpoints
        self.log_fidelity = log_fidelity
        self.smoothness_method_amp = smoothness_calc_amp
        self.smoothness_method_det = smoothness_calc_det
        self.smoothness_cutoff_freq = smoothness_cutoff_freq
        self.dim = dim
        self.noise = noise
        self.gaussian_noise_scaling_amp = gaussian_noise_scaling[1]
        self.gaussian_noise_scaling_phase = gaussian_noise_scaling[0]
        self.ou_noise_params = ou_noise_params
        self.noise_over_sampling_rate = 4
        self.n_sections = n_sections
        self.use_mu_beta = use_mu_beta
        self.mxstep = mxstep
        self.convolution_std_amp = convolution_std_amp
        self.convolution_std_freq = convolution_std_freq
        self.mx_step_penalty = mx_step_penalty

        # Time in microseconds
        self._t_total = t_total

        # dict of values to log, along with their types
        self.log_vals = {
            "fid": 0,
            "x-state-avg": 0,
            "area-s": 0,
            "area-p": 0,
            "smoothness-s": 0,
            "smoothness-p": 0,
            "env-s": jnp.zeros(self._n_steps),
            "env-p": jnp.zeros(self._n_steps),
            "det-s": jnp.zeros(self._n_steps),
            "det-p": jnp.zeros(self._n_steps),
            "max-steps-reached": 0,
            "num-steps": 0,
            "noise-amp":
            jnp.zeros(self._n_steps * self.noise_over_sampling_rate),
            "noise-freq":
            jnp.zeros(self._n_steps * self.noise_over_sampling_rate)
        }

        # dtype
        if processor == "gpu":
            self.float_dtype = jnp.float32
            self.complex_dtype = jnp.complex64
        else:
            self.float_dtype = jnp.float64
            self.complex_dtype = jnp.complex128

        # self.masks = self._create_masks(self._n_steps, self.n_sections)
        self.masks = jnp.zeros((self._n_steps, self.n_sections),
                               dtype=self.float_dtype)
        self.masks = self._create_masks(self.masks, self._n_steps,
                                        self.n_sections, self.float_dtype)

        # sim steps finer than action steps
        self.sim_steps = self._n_steps * self._resolution
        self.ts_sim = jnp.linspace(0.0, self._t_total, self.sim_steps,
                                   self.float_dtype)
        self.ts_action = jnp.linspace(0.0, self._t_total, self._n_steps,
                                      self.float_dtype)
        self.dt = self._t_total / len(self.ts_sim - 1)

        # define Hamiltonian
        #load hamiltonian
        self.ham_ops, self.c_ops = get_hamiltonian_operators(
            self.dim, self._gamma, _dtype=self.complex_dtype)

        # define initial/final states
        if self.dim == 4:

            # define initial state (pure state in case of no dissipator)
            self.psi0 = (1 /
                         jnp.sqrt(initial_state[0]**2 + initial_state[1]**2) *
                         (jnp.array([initial_state[0], 0, initial_state[1], 0],
                                    dtype=self.complex_dtype)))

            # define initial state (mixed state in case of dissipator)
            self.rho0 = (
                1 / (initial_state[0]**2 + initial_state[1]**2) *
                (initial_state[0] * statevector_to_rho(
                    jnp.array([1, 0, 0, 0], dtype=self.complex_dtype)) +
                 initial_state[1] * statevector_to_rho(
                     jnp.array([0, 0, 1, 0], dtype=self.complex_dtype))))
            self.psi_des = (1 /
                            jnp.sqrt(final_state[0]**2 + final_state[1]**2) *
                            (jnp.array([final_state[0], 0, final_state[1], 0],
                                       dtype=self.complex_dtype)))
            self.H_dp, self.H_ds, self.H_p, self.H_s = self.ham_ops[:]

        elif self.dim == 5:

            # define desired final state
            self.psi_des = (
                1 / jnp.sqrt(final_state[0]**2 + final_state[1]**2) *
                (jnp.array([final_state[0], 0, final_state[1], 0, 0],
                           dtype=self.complex_dtype)))

            # define initial state (pure state in case of no dissipator)
            self.psi0 = (1 /
                         jnp.sqrt(initial_state[0]**2 + initial_state[1]**2) *
                         (jnp.array(
                             [initial_state[0], 0, initial_state[1], 0, 0],
                             dtype=self.complex_dtype,
                         )))

            # define initial state (mixed state in case of dissipator)
            self.rho0 = (
                1 / (initial_state[0]**2 + initial_state[1]**2) *
                (initial_state[0] * statevector_to_rho(
                    jnp.array([1, 0, 0, 0, 0], dtype=self.complex_dtype)) +
                 initial_state[1] * statevector_to_rho(
                     jnp.array([0, 0, 1, 0, 0], dtype=self.complex_dtype))))

            # define hamiltonian operators

            self.H_dp, self.H_ds, self.H_d_x, self.H_p, self.H_s, self.H_p_x, self.H_s_x = self.ham_ops[:]
        else:
            raise ValueError("State Space must be a 4 or 5 dimensional vector")

        if dissipator:
            self.solver = Solver(
                hamiltonian_operators=self.ham_ops,
                static_dissipators=self.c_ops,
                validate=False,
                array_library=processor_array_type,
                # vectorized=True,
            )
        else:
            self.solver = Solver(
                hamiltonian_operators=self.ham_ops,
                validate=False,
                array_library=processor_array_type,
                # vectorized=True,
            )

        #define initial state and desired final state
        self.dissipator = dissipator
        if self.dissipator:
            self.yi = self.rho0
        else:
            self.yi = self.psi0
        self.rho_des = statevector_to_rho(self.psi_des)

        # reward for achieving good fidelity
        self.fid_factor = fid_factor
        # penalty for inducing excited state population
        self.x_penalty = x_penalty
        # penalty for pulse area & smoothness
        self.pulse_area_penalty = pulse_area_penalty
        self.smoothness_penalty_env = smoothness_penalty_env
        self.smoothness_penalty_det = smoothness_penalty_det

        # define gaussian kernel for smoothing
        self.window_length_amp = gaussian_kernel_window_length[0]
        self.window_length_freq = gaussian_kernel_window_length[1]
        self.gauss_kernel_amp = gen_gauss_kernel(
            self.window_length_amp, gauss_std=self.convolution_std_amp)
        self.gauss_kernel_freq = gen_gauss_kernel(
            self.window_length_freq, gauss_std=self.convolution_std_freq)

        #def blackman waveform for baseline
        self.bm_waveform = self._omega_0 * jnp.array(
            jnp.abs(blackman(self._n_steps)), dtype=self.float_dtype)
        self.signal_tools = SignalToolClass(self.gauss_kernel_amp,
                                            self.gauss_kernel_freq,
                                            self._t_total, self.bm_waveform)
        self.baseline_area = trapezoid(self.bm_waveform, self.ts_action)
        self.baseline_smoothness_amp = self.signal_tools.get_baseline_smoothness(
            self.bm_waveform,
            method=self.smoothness_method_amp,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)
        #use the same function to calculate the baseline smoothness for the detuning but use the frequency kernel
        self.baseline_smoothness_freq = self.signal_tools.get_baseline_smoothness(
            self.bm_waveform,
            False,
            method=self.smoothness_method_det,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)

    @staticmethod
    def _create_masks(masks, n_steps, n_sections, _dtype=jnp.float64):
        assert n_steps % n_sections == 0

        def body_fun(i, masks):
            start = i * (n_steps // n_sections)
            length = (n_steps // n_sections)
            update = jnp.ones((length, 1), dtype=_dtype)
            masks = jax.lax.dynamic_update_slice(masks, update, (start, i))
            return masks

        masks = jax.lax.fori_loop(0, n_sections, body_fun, masks)
        return masks

    @property
    def default_params(self) -> EnvParams:
        """
        IMPORTANT Retrieving the Default Env Params
        """
        return EnvParams(
            args=jnp.array(
                [self._gamma, self._omega_0, self._delta_p, self._delta_s],
                dtype=self.float_dtype,
            ),
            tf=self._t_total,
        )

    def calc_step(self, key: chex.PRNGKey, action: chex.Array,
                  mu_amp_sampled) -> chex.Array:
        """input args:
        single_action_sample: tensor 'slice' of the full action tensor at a particular batch_index which corresponds to amplitudes and phases of the two pulses
        """
        rng, rng_noise = jax.random.split(key)

        delta_p = action[0:self._n_steps].T
        delta_s = action[self._n_steps:2 * self._n_steps].T
        omega_p = action[2 * self._n_steps:3 * self._n_steps].T
        omega_s = action[3 * self._n_steps:4 * self._n_steps].T

        # LINEAR INTERPOLATION
        # chirp_stokes = LinearInterpolation(ts=self.ts_action, ys=delta_s)
        # chirp_pump = LinearInterpolation(ts=self.ts_action, ys=delta_p)
        # env_stokes = LinearInterpolation(ts=self.ts_action, ys=omega_s)
        # env_pump = LinearInterpolation(ts=self.ts_action, ys=omega_p)

        # CUBIC INTERPOLATION
        chirp_stokes_coeff = backward_hermite_coefficients(
            self.ts_action, delta_s)
        chirp_pump_coeff = backward_hermite_coefficients(
            self.ts_action, delta_p)
        env_stokes_coeff = backward_hermite_coefficients(
            self.ts_action, omega_s)
        env_pump_coeff = backward_hermite_coefficients(self.ts_action, omega_p)

        chirp_stokes = CubicInterpolation(ts=self.ts_action,
                                          coeffs=chirp_stokes_coeff)
        chirp_pump = CubicInterpolation(ts=self.ts_action,
                                        coeffs=chirp_pump_coeff)
        env_stokes = CubicInterpolation(ts=self.ts_action,
                                        coeffs=env_stokes_coeff)
        env_pump = CubicInterpolation(ts=self.ts_action, coeffs=env_pump_coeff)

        # generate markovian noise
        if self.noise == "ou":
            (ou_noise_chirp_pump, ou_noise_chirp_stokes, ou_noise_env_pump,
             ou_noise_env_stokes, noise_samples_env_pump,
             noise_samples_chirp_pump) = generate_ou_noise_samples(
                 rng_noise,
                 self._n_steps,
                 self.noise_over_sampling_rate,
                 self.ou_noise_params,
                 self._t_total,
                 _dtype=self.float_dtype,
                 n_signals=2)
        if self.noise == "g":
            (g_noise_env_p, g_noise_det_p, g_noise_env_s, g_noise_det_s,
             noise_samples_env_pump,
             noise_samples_chirp_pump) = generate_gaussian_noise_samples(
                 rng_noise,
                 self._n_steps,
                 self._omega_0,
                 self._omega_0,
                 self._delta_p,
                 self._delta_s,
                 self.gaussian_noise_scaling_amp,
                 self.gaussian_noise_scaling_phase,
                 n_signals=2)
        else:
            (
                noise_samples_env_pump,
                noise_samples_chirp_pump,
            ) = generate_zero_noise_samples(
                self._n_steps,
                self.noise_over_sampling_rate,
                _dtype=self.float_dtype,
                n_signals=1,
            )

        def f_chirp_stokes(t):
            if self.noise == "g":
                return jnp.abs(chirp_stokes.evaluate(t) + g_noise_det_s)
            elif self.noise == "ou":
                return chirp_stokes.evaluate(
                    t) + ou_noise_chirp_stokes.evaluate(t)
            else:
                return chirp_stokes.evaluate(t)

        def f_chirp_pump(t):
            if self.noise == "g":
                return jnp.abs(chirp_pump.evaluate(t) + g_noise_det_p)
            elif self.noise == "ou":
                return chirp_pump.evaluate(t) + ou_noise_chirp_pump.evaluate(t)
            else:
                return chirp_pump.evaluate(t)

        def f_env_stokes(t):
            if self.noise == "g":
                return jnp.abs(env_stokes.evaluate(t) + g_noise_env_s)
            elif self.noise == "ou":
                return env_stokes.evaluate(t) + ou_noise_env_stokes.evaluate(t)
            else:
                return env_stokes.evaluate(t)

        def f_env_pump(t):
            if self.noise == "g":
                return jnp.abs(env_pump.evaluate(t) + g_noise_env_p)
            elif self.noise == "ou":
                return env_pump.evaluate(t) + ou_noise_env_pump.evaluate(t)
            else:
                return env_pump.evaluate(t)

        vec_delta_p = jnp.vectorize(f_chirp_pump)
        vec_delta_s = jnp.vectorize(f_chirp_stokes)
        vec_p_env = jnp.vectorize(f_env_pump)
        vec_s_env = jnp.vectorize(f_env_stokes)

        if self.dissipator:
            yi = self.rho0
        else:
            yi = self.psi0

        if self.dim == 4:
            # define signals
            signal_list = [
                Signal(vec_delta_p),
                Signal(vec_delta_s),
                Signal(vec_p_env),
                Signal(vec_s_env),
            ]

        elif self.dim == 5:

            def f_x_det(t):
                return chirp_pump.evaluate(t) + self.x_detuning

            vec_x_det = jnp.vectorize(f_x_det)

            # define signals
            signal_list = [
                Signal(vec_delta_p),
                Signal(vec_delta_s),
                Signal(vec_x_det),
                Signal(vec_p_env),
                Signal(vec_s_env),
                Signal(vec_p_env),
                Signal(vec_s_env),
            ]
        else:
            AssertionError("Invalid dimension number")

        stepsize_controller = PIDController(rtol=1e-8, atol=1e-6)

        sol = self.solver.solve(
            t_span=[self.ts_sim[0], self.ts_sim[-1]],
            signals=signal_list,
            y0=yi,
            t_eval=self.ts_action,
            convert_results=False,
            #method="jax_odeint",
            method=Dopri5(),
            # method=Euler(),
            # dt0=1/50,
            max_steps=self.mxstep,
            throw=False,
            stepsize_controller=stepsize_controller
            # max_dt = 1/350,
        )

        max_steps_reached = (sol.result == RESULTS.max_steps_reached)
        num_steps = sol.stats["num_steps"]

        rho_f = sol.y[-1]

        # calculate average excited state population
        if self.dissipator:
            x_state_avg = jnp.mean(jnp.abs(sol.y[:, 1, 1]))
        else:
            x_state_avg = jnp.mean(jnp.abs(sol.y[:, 1]))

        noise_samples_chirp_pump = noise_samples_chirp_pump.astype(
            self.float_dtype)
        noise_samples_env_pump = noise_samples_env_pump.astype(
            self.float_dtype)

        return rho_f, x_state_avg, noise_samples_env_pump, noise_samples_chirp_pump, num_steps, max_steps_reached

    def get_result_full_action(self, key: chex.PRNGKey, state: EnvState,
                               action_pump_det, action_stokes_det,
                               action_pump_env, action_stokes_env,
                               params: EnvParams):

        # Clip Amplitudes
        # While the amplitude and mean action values from the NN will vary
        # between -1 and 1, we want them to be positive values
        # So we translate and scale them to between 0 and 1
        action_pump_env = action_pump_env.at[:].set(action_pump_env[:] * 0.5 +
                                                    0.5)
        action_pump_env = action_pump_env.at[:].set(
            jnp.clip(action_pump_env[:], a_min=0.0,
                     a_max=1.0))  # Clipping Amplitudes
        action_stokes_env = action_stokes_env.at[:].set(action_stokes_env[:] *
                                                        0.5 + 0.5)
        action_stokes_env = action_stokes_env.at[:].set(
            jnp.clip(action_stokes_env[:], a_min=0.0,
                     a_max=1.0))  # Clipping Amplitudes

        stokes_env, pump_env = action_stokes_env.astype(
            self.float_dtype), action_pump_env.astype(self.float_dtype)

        stokes_det, pump_det = action_stokes_det.astype(
            self.float_dtype), action_pump_det.astype(self.float_dtype)

        stokes_env = self.signal_tools.drive_smoother(stokes_env)
        pump_env = self.signal_tools.drive_smoother(pump_env)
        stokes_det = self.signal_tools.drive_smoother(stokes_det, False)
        pump_det = self.signal_tools.drive_smoother(pump_det, False)

        normalizing_factor = self._omega_0
        stokes_env *= normalizing_factor
        pump_env *= normalizing_factor

        # Simulation and Obtaining Reward + Params for New State
        action_concatenated = jnp.concatenate(
            (pump_det, stokes_det, pump_env, stokes_env))

        rng, key = jax.random.split(key)

        single_result = self.calc_step(rng, action_concatenated,
                                       state.mu_amp_sampled)
        reward, updated_state_floats, noise_a, noise_freq, num_steps, mx_steps = self.calc_reward_and_state(
            single_result, action_concatenated)

        stokes_env = stokes_env.astype(self.float_dtype)
        pump_env = pump_env.astype(self.float_dtype)
        pump_det = pump_det.astype(self.float_dtype)
        stokes_det = stokes_det.astype(self.float_dtype)
        noise_a = noise_a.astype(self.float_dtype)
        noise_freq = noise_freq.astype(self.float_dtype)

        return updated_state_floats, stokes_env, pump_env, stokes_det, pump_det, noise_a, noise_freq, num_steps, mx_steps, reward

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

        # Setting New State with Updated State Array,
        # New Optimal Action (for logging),
        # and New Timestep

        assert self._n_steps % self.n_sections == 0
        section_length = int(self._n_steps / self.n_sections)

        action_pump_det_sub = action[0:section_length]
        action_stokes_det_sub = action[section_length:2 * section_length]
        action_pump_env_sub = action[2 * section_length:3 * section_length]
        action_stokes_env_sub = action[3 * section_length:4 * section_length]

        start_idx = (new_timestep - 1) * section_length
        end_idx = new_timestep * section_length

        action_pump_det = state.action_pump_det
        action_stokes_det = state.action_stokes_det
        action_pump_env = state.action_pump_env
        action_stokes_env = state.action_stokes_env

        action_pump_det = action_pump_det + jnp.repeat(
            action_pump_det_sub, self.n_sections) * self.masks[:,
                                                               state.timestep]
        action_stokes_det = action_stokes_det + jnp.repeat(
            action_stokes_det_sub,
            self.n_sections) * self.masks[:, state.timestep]
        action_pump_env = action_pump_env + jnp.repeat(
            action_pump_env_sub, self.n_sections) * self.masks[:,
                                                               state.timestep]
        action_stokes_env = action_stokes_env + jnp.repeat(
            action_stokes_env_sub,
            self.n_sections) * self.masks[:, state.timestep]

        def true_fn(args):
            key, state, action_pump_det, action_stokes_det, action_pump_env, action_stokes_env, params = args
            updated_state_floats, stokes_env, pump_env, stokes_det, pump_det, noise_a, noise_freq, num_steps, max_steps, reward = self.get_result_full_action(
                key, state, action_pump_det, action_stokes_det,
                action_pump_env, action_stokes_env, params)
            updated_state = EnvState(*updated_state_floats, stokes_env,
                                     pump_env, stokes_det, pump_det, num_steps,
                                     max_steps, noise_a, noise_freq,
                                     state.mu_amp_sampled, action_pump_det,
                                     action_stokes_det, action_pump_env,
                                     action_stokes_env, new_timestep)
            return updated_state, True, reward

        def false_fn(args):
            key, state, action_pump_det, action_stokes_det, action_pump_env, action_stokes_env, params = args
            updated_state = EnvState(
                state.reward, state.fid, state.x_state_pop_avg, state.area_s,
                state.area_p, state.smoothness_s, state.smoothness_p,
                state.env_s, state.env_p, state.det_s, state.det_p,
                state.num_steps, state.max_steps_reached, state.noise_amp,
                state.noise_freq, state.mu_amp_sampled, action_pump_det,
                action_stokes_det, action_pump_env, action_stokes_env,
                new_timestep)
            return updated_state, False, 0.0

        updated_state, done, reward = jax.lax.cond(
            new_timestep == self.n_sections, true_fn, false_fn,
            (key, state, action_pump_det, action_stokes_det, action_pump_env,
             action_stokes_env, params))

        return (
            lax.stop_gradient(self.get_obs(updated_state)),
            lax.stop_gradient(updated_state),
            reward,
            done,
            lax.stop_gradient(self.get_info(updated_state)),
        )

    def extract_values(self, results: chex.Array, actions: chex.Array):
        """Takes in values from simulation and returns fidelity and smoothness"""

        # Extracting Fidelity and Smoothness Values

        rho_f, xstate_avg = results
        if self.dissipator:
            max_fid = mixed_pure_state_fidelity(self.psi_des, rho_f)
        else:
            max_fid = pure_state_fidelity(self.psi_des, rho_f)

        det_p = actions[0:self._n_steps]
        det_s = actions[self._n_steps:2 * self._n_steps]
        env_p = actions[2 * self._n_steps:3 * self._n_steps]
        env_s = actions[3 * self._n_steps:4 * self._n_steps]

        smoothness_det_p = self.signal_tools.calculate_smoothness(
            det_p,
            method=self.smoothness_method_det,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)
        smoothness_det_s = self.signal_tools.calculate_smoothness(
            det_s,
            method=self.smoothness_method_det,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)
        # penalise the smoothness of the amplitude envelope
        smoothness_p = self.signal_tools.calculate_smoothness(
            env_p,
            method=self.smoothness_method_amp,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)
        smoothness_s = self.signal_tools.calculate_smoothness(
            env_s,
            method=self.smoothness_method_amp,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)

        area_s = self.signal_tools.area_of_pulse(env_s)
        area_p = self.signal_tools.area_of_pulse(env_p)

        return max_fid, xstate_avg, smoothness_s, smoothness_p, smoothness_det_p, smoothness_det_s, area_s, area_p

    def calc_reward_and_state(
        self,
        result: Tuple,
        drive: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Function holding Reward Calculation and State Param Calculations
        input args:
        results: batch of simulation results containing fid and smoothness
        drives: batch of actions"""

        (
            fid_state_transfer,
            excited_population,
            smoothness_s,
            smoothness_p,
            smoothness_det_s,
            smoothness_det_p,
            area_s,
            area_p,
        ) = self.extract_values(result[0:2], drive)

        if self.noise == "ou":
            noise_amp = result[2]
            noise_freq = result[3]
        else:
            noise_amp = jnp.zeros(
                self._n_steps * self.noise_over_sampling_rate,
                self.float_dtype)
            noise_freq = jnp.zeros(
                self._n_steps * self.noise_over_sampling_rate,
                self.float_dtype)

        num_steps = result[4]
        mx_steps_reached = result[5]

        if self.log_fidelity:
            fid_reward = jnp.log(1 / (1 - fid_state_transfer))
        else:
            fid_reward = fid_state_transfer

        def true_fun(_):
            mx_reward = 1
            fid_reward = 0.0
            fid_state_transfer = 0.0
            excited_population = 0.0
            return mx_reward, fid_reward, fid_state_transfer, excited_population

        # Modify the false_fun to keep fid_reward and excited_population unchanged
        def false_fun(_):
            mx_reward = 0
            # Since we do not modify fid_reward and excited_population in false_fun,
            # they will remain unchanged.
            return mx_reward, fid_reward, fid_state_transfer, excited_population

        # Use lax.cond to conditionally execute either true_fun or false_fun
        mx_reward, fid_reward, fid_state_transfer, excited_population = lax.cond(
            mx_steps_reached, true_fun, false_fun, operand=None)

        reward = (
            self.fid_factor * fid_state_transfer -
            self.x_penalty * excited_population - self.pulse_area_penalty *
            (area_s / self.baseline_area + area_p / self.baseline_area) / 2 -
            self.smoothness_penalty_env *
            (relu((smoothness_s + smoothness_p) /
                  (2 * self.baseline_smoothness_amp) - 1)) -
            self.smoothness_penalty_det *
            (relu((smoothness_det_s + smoothness_det_p) /
                  (2 * self.baseline_smoothness_freq) - 1)) -
            self.mx_step_penalty * mx_reward)

        state = jnp.array(
            [
                reward,
                fid_state_transfer,
                excited_population,
                area_s,
                area_p,
                smoothness_s,
                smoothness_p,
            ],
            dtype=self.float_dtype,
        )

        return reward, state, noise_amp, noise_freq, num_steps, mx_steps_reached

    def reset(self, key: chex.PRNGKey,
              params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """IMPORTANT Reset Environment, in this case nothing needs to be done
        so default obs and info are returned"""
        # self.precompile()

        key, _key = jax.random.split(key)

        if self.use_mu_beta:

            def true_fn(beta_dis_val):
                mu_amp_sampled = -1 * beta_dis_val * self.ou_noise_params[1]
                return mu_amp_sampled

            def false_fn(beta_dis_val):
                mu_amp_sampled = beta_dis_val * self.ou_noise_params[1]
                return mu_amp_sampled

            beta_dis_val = jax.random.beta(_key, 4, 0.7, shape=(1, ))[0]
            cond = jax.random.uniform(_key, shape=(1, ))[0] > 0.5
            mu_amp_sampled = lax.cond(cond,
                                      true_fn,
                                      false_fn,
                                      operand=beta_dis_val)

        else:
            mu_amp_sampled = (jax.random.uniform(_key, shape=(1, ))[0] * 2 -
                              1) * self.ou_noise_params[1]

        state = EnvState(
            reward=0.0,
            fid=0.0,
            x_state_pop_avg=0.0,
            area_s=0.0,
            area_p=0.0,
            smoothness_s=0.0,
            smoothness_p=0.0,
            env_s=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            env_p=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            det_s=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            det_p=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            num_steps=0,
            max_steps_reached=False,
            noise_amp=jnp.zeros(self._n_steps * self.noise_over_sampling_rate,
                                dtype=self.float_dtype),
            noise_freq=jnp.zeros(self._n_steps * self.noise_over_sampling_rate,
                                 dtype=self.float_dtype),
            mu_amp_sampled=mu_amp_sampled,
            action_pump_det=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            action_stokes_det=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            action_pump_env=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            action_stokes_env=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            timestep=0,
        )

        return self.get_obs(state), state

    def get_obs(self, env_state) -> chex.Array:
        """IMPORTANT Function to get observation at a given state, as this is a single-step
        episode environment, the observation can be left fixed"""

        return jax.lax.cond(env_state.timestep == 0,
                            lambda _: jnp.array((0.0, 0.0)),
                            lambda _: jnp.array(
                                (1.0, env_state.mu_amp_sampled)),
                            operand=None)

    def get_info(self, env_state: EnvState) -> dict:
        """IMPORTANT Function to get info for a given input state"""
        info_dict = {
            "reward": env_state.reward,
            "fid": env_state.fid,
            "x-state-avg": env_state.x_state_pop_avg,
            "area-s": env_state.area_s,
            "area-p": env_state.area_p,
            "smoothness-s": env_state.smoothness_s,
            "smoothness-p": env_state.smoothness_p,
            "env-s": env_state.env_s,
            "env-p": env_state.env_p,
            "det-s": env_state.det_s,
            "det-p": env_state.det_p,
            "num-steps": env_state.num_steps,
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
        return "Amplitude & Phase Modulated 4lvl STIRAP"

    @property
    def num_actions(self, params: Optional[EnvParams] = EnvParams) -> int:
        """IMPORTANT number of actions"""
        return int(4 * self._n_steps)

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """IMPORTANT action space shape"""
        if params is None:
            params = self.default_params

        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(int(4 * self._n_steps / self.n_sections), ),
            dtype=self.float_dtype,
        )

    def observation_space(self,
                          params: Optional[EnvParams] = None) -> spaces.Box:
        """IMPORTANT observation space shape"""
        # TODO update this
        return spaces.Box(-1000, 1000, shape=(2, ), dtype=self.float_dtype)


# Test the environment
if __name__ == "__main__":

    # Seeding
    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)

    env = MultiStirap(gamma=3.0,
                      omega_0=10.0,
                      delta_s=-10.0,
                      delta_p=0.0,
                      dissipator=True,
                      dim=4,
                      n_sections=4,
                      n_action_steps=32,
                      noise="ou")

    input_action = jax.random.normal(rng,
                                     shape=(int(4 * env._n_steps /
                                                env.n_sections), ))

    # env_params = env.default_params
    # obs, state = env.reset(_rng, env_params)

    # # Testing single step evaluation
    # start_time = time.time()
    # print(obs)
    # for _ in range(5):
    #     input_action = jax.random.normal(rng, shape=(int(4*env._n_steps/env.n_sections), ))
    #     obs, state, reward, done, info= env.step(_rng, state, input_action, env_params)
    #     print(obs)

    ### Testing vmapped evaluation
    vec_env = VecEnv(env)
    vec_env_params = env.default_params

    ### init env
    num_envs = 8
    rng, _rng = jax.random.split(rng)
    rng_reset = jax.random.split(_rng, num_envs)
    vec_obs, vec_state = vec_env.reset(rng_reset, vec_env_params)

    # making the action
    vectorised_action = jax.random.normal(
        rng, shape=(num_envs, int(4 * env._n_steps / env.n_sections)))

    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, num_envs)

    jit_vmap_step = jax.jit(vec_env.step)

    for _ in range(20):
        start_time = time.time()

        jax.debug.print("{x}", x=vec_obs)
        vec_obs, vec_state, reward, done, info = jit_vmap_step(
            rng_step, vec_state, vectorised_action, vec_env_params)
        jax.debug.print("{x}", x=reward)

        if jnp.sum(done).item() > 0:
            print("resetting")
            vec_obs, vec_state = vec_env.reset(rng_step, vec_env_params)

        print(f" time for step: {time.time()-start_time}")

    # start = time.time()
    # returned_obsv, new_env_state, reward, done, info = block_until_ready(
    #     jit_vmap_step(rng_step, vec_env_state, vectorised_action, vec_env_params)
    # )
    # end = time.time()

    # print(
    #     f"Time taken for batch of {num_steps} steps in vmap: {jnp.round(end - start, 3)}s"
    # )
    # print(
    #     f"Time taken per vmapped step: {jnp.round((end - start) / num_steps /1e-6)}us"
    # )
