from typing import Tuple, Optional, List
from dataclasses import field
from scipy.signal.windows import blackman
import time
import os
import matplotlib.pyplot as plt

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
                     LinearInterpolation, Dopri8, PIDController,
                     ConstantStepSize, RESULTS
                     # AbastractAdaptiveSolver
                     )
from environment_template import SingleStepEnvironment
from utils.wrappers import VecEnv
from utils.noise_functions import generate_gaussian_noise_samples, generate_ou_noise_samples, generate_zero_noise_samples
from utils.signal_functions import gen_gauss_kernel, SignalToolClass

# Qiskit Imports
from utils.shared_quantum_functions import (statevector_to_rho,
                                            mixed_pure_state_fidelity,
                                            pure_state_fidelity)
from qiskit_dynamics import Solver, Signal

from hamiltonian.ryd_two_hamiltonian import generate_hamiltonian_and_c_ops

# Check if GPU is available
if "cuda" in str(jax.devices()):
    print("Connected to a GPU")
    processor_array_type = "jax"
    processor = "gpu"

else:
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)
    print("Not connected to a GPU")
    processor = "cpu"
    processor_array_type = "jax_sparse"


@struct.dataclass
class EnvState:
    """
    Flax Dataclass used to store Dynamic Environment State
    All relevant params that get updated each step should be stored here
    """

    reward: float
    fid: float
    x_e: float
    x_r: float
    area_p: float
    area_s: float
    smoothness_p: float
    smoothness_s: float
    smoothness_det_p: float
    smoothness_det_s: float
    env_p: jnp.ndarray
    env_s: jnp.ndarray
    det_p: jnp.ndarray
    det_s: jnp.ndarray
    max_steps_reached: bool
    num_steps: int
    noise_amp: jnp.ndarray
    noise_freq: jnp.ndarray
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


class RydbergTwoEnv(SingleStepEnvironment):

    def __init__(
            self,
            gamma: float,
            blockade_strength: float,
            omega_r: float,
            omega_e: float,
            delta_r: float,
            delta_e: float,
            mxstep: int,
            t_total: float = 1.0,
            fid_reward: float = 4.0,
            x_penalty: float = 1.0,
            log_fidelity: int = 0,
            pulse_area_penalty: float = 0.0,
            convolution_std_amp: float = 1,
            convolution_std_freq: float = 1,
            smoothness_calc_amp: str = "second_derivative",
            smoothness_calc_det: str = "second_derivative",
            smoothness_cutoff_freq: float = 5,
            smoothness_penalty_env: float = 0.001,
            smoothness_penalty_det: float = 0.1,
            mx_step_penalty: float = 0.0,
            fixed_endpoints: int = 0,
            gaussian_kernel_window_length=[50, 25],  # amplitude, frequency
            n_action_steps: int = 50,
            dissipator: int = 1,
            const_freq_pump: int = 0,
            const_amp_stokes: int = 0,
            e_dissipation: float = 1,
            r_dissipation: float = 0.001,
            dim: int = 2,  # num qubits
            noise: str = "None",  # ou or g
            gaussian_noise_scaling: jnp.ndarray = [0.01,
                                                   0.01],  # phase, amplitude
            ou_noise_params: jnp.ndarray = [
                0,
                0,
                0.1,
                0.1,
                0.5,
                0.5,
            ],  # mu_freq, mu_amp alpha_freq, alpha_amp, sigma_freq, sigma_amp
    ):
        # Coupling strengths etc... in 2piMHz (angular frequency)
        self._gamma = gamma * 2 * jnp.pi
        self._omega_r = omega_r * 2 * jnp.pi
        self._omega_e = jnp.sqrt(
            jnp.abs(delta_e) * omega_e
        ) * 2 * jnp.pi  #convert from effective to single photon rabi frequency
        self._delta_r = delta_r * 2 * jnp.pi
        self._delta_e = delta_e * 2 * jnp.pi
        self._n_steps = n_action_steps
        self.blockade_strength = blockade_strength * 2 * jnp.pi
        self.dim = dim
        self.noise = noise
        self.gaussian_noise_scaling_amp = gaussian_noise_scaling[1]
        self.gaussian_noise_scaling_phase = gaussian_noise_scaling[0]
        self.ou_noise_params = ou_noise_params
        self.noise_over_sampling_rate = 4
        self.e_dissipation = e_dissipation
        self.r_dissipation = r_dissipation
        self.const_freq_pump = const_freq_pump
        self.const_amp_stokes = const_amp_stokes
        self.fixed_endpoints = fixed_endpoints
        self.log_fidelity = log_fidelity
        self.smoothness_method_amp = smoothness_calc_amp
        self.smoothness_method_det = smoothness_calc_det
        self.smoothness_cutoff_freq = smoothness_cutoff_freq * 2 * jnp.pi
        self.convolution_std_amp = convolution_std_amp
        self.convolution_std_freq = convolution_std_freq

        self.mxstep = mxstep
        assert (mxstep > 0), "mxstep must be an int greater than 0"
        self.mx_step_penalty = mx_step_penalty

        assert noise in ["ou", "g", "None"], "Noise must be either 'ou' or 'g'"

        # Time in microseconds
        self._t_total = t_total

        # dict of values to log, along with their types
        self.log_vals = {
            "fid": 0,
            "e-state-pop-avg": 0,
            "r-state-pop-avg": 0,
            "area-p": 0,
            "area-s": 0,
            "smoothness-p": 0,
            "smoothness-s": 0,
            "smoothness-det-p": 0,
            "smoothness-det-s": 0,
            "env-p": jnp.zeros(self._n_steps),
            "env-s": jnp.zeros(self._n_steps),
            "det-p": jnp.zeros(self._n_steps),
            "det-s": jnp.zeros(self._n_steps),
            "max-steps-reached": 0,
            "num-steps": 0,
            "noise-amp":
            jnp.zeros(self._n_steps * self.noise_over_sampling_rate),
            "noise-freq":
            jnp.zeros(self._n_steps * self.noise_over_sampling_rate),
        }

        # dtype
        if processor == "gpu":
            self.float_dtype = jnp.float32
            self.complex_dtype = jnp.complex64
        else:
            self.float_dtype = jnp.float64
            self.complex_dtype = jnp.complex128

        # sim steps finer than action steps
        self._resolution = 10
        self.sim_steps = self._n_steps * self._resolution
        self.ts_action = jnp.linspace(0.0, self._t_total, self._n_steps)
        self.ts_sim = jnp.linspace(0.0, self._t_total, self.sim_steps)

        # define Hamiltonian
        if self.dim == 2:

            # define initial pure state (pure state in case of no dissipator)
            self.psi0 = (
                1 / jnp.sqrt(4) *
                (jnp.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            dtype=self.complex_dtype)))

            # define initial state (mixed state in case of dissipator)
            self.rho0 = statevector_to_rho(self.psi0)
            self.psi_des = (
                1 / jnp.sqrt(4) *
                (jnp.array([1, 1, 1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            dtype=self.complex_dtype)))

            self.h_ops, self.c_ops = generate_hamiltonian_and_c_ops(
                reduced_hspace=False,
                complex_dtype=self.complex_dtype,
                gamma=self._gamma,
                e_dissipation=self.e_dissipation,
                r_dissipation=self.r_dissipation,
            )

            self.H_omega_p, self.H_omega_s, self.H_delta_p, self.H_delta_s, self.H_delta_sp, self.H_v = self.h_ops[:]

            if dissipator:
                self.solver = Solver(
                    hamiltonian_operators=self.h_ops,
                    static_dissipators=self.c_ops,
                    validate=False,
                    array_library=processor_array_type,
                    # vectorized=True,
                )
            else:
                self.solver = Solver(
                    hamiltonian_operators=self.h_ops,
                    validate=False,
                    array_library=processor_array_type,
                    # vectorized=True,
                )

        else:
            raise ValueError("State Space must be 2 qubits")

        # define initial state and desired final state
        self.dissipator = dissipator
        if self.dissipator:
            self.yi = self.rho0
        else:
            self.yi = self.psi0
        self.rho_des = statevector_to_rho(self.psi_des)

        # reward for achieving good fidelity
        self.fid_factor = fid_reward
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
        self.bm_waveform_e = self._omega_e * jnp.array(
            jnp.abs(blackman(self._n_steps)), dtype=self.float_dtype)
        self.bm_waveform_r = self._omega_r * jnp.array(
            jnp.abs(blackman(self._n_steps)), dtype=self.float_dtype)
        self.baseline_area_e = trapezoid(self.bm_waveform_e, self.ts_action)
        self.baseline_area_r = trapezoid(self.bm_waveform_r, self.ts_action)
        self.signal_tools = SignalToolClass(
            self.gauss_kernel_amp,
            self.gauss_kernel_freq,
            self._t_total,
            self.bm_waveform_e,
        )
        self.baseline_smoothness_amp_e = self.signal_tools.get_baseline_smoothness(
            self.bm_waveform_e,
            True,
            method=self.smoothness_method_amp,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)
        #use the same function to calculate the baseline smoothness for the detuning but use the frequency kernel
        self.baseline_smoothness_freq_e = self.signal_tools.get_baseline_smoothness(
            self.bm_waveform_e,
            False,
            method=self.smoothness_method_det,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)

        self.baseline_smoothness_amp_r = self.signal_tools.get_baseline_smoothness(
            self.bm_waveform_r,
            True,
            method=self.smoothness_method_amp,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)
        #use the same function to calculate the baseline smoothness for the detuning but use the frequency kernel
        self.baseline_smoothness_freq_r = self.signal_tools.get_baseline_smoothness(
            self.bm_waveform_r,
            False,
            method=self.smoothness_method_det,
            _f_sampling=self._n_steps,
            _f_cutoff=self.smoothness_cutoff_freq)

    @property
    def default_params(self) -> EnvParams:
        """
        IMPORTANT Retrieving the Default Env Params
        """
        return EnvParams(
            args=jnp.array(
                [
                    self._gamma, self._omega_e, self._omega_r, self._delta_e,
                    self._delta_r
                ],
                dtype=self.float_dtype,
            ),
            tf=self._t_total,
        )

    def calc_step(self, key: chex.PRNGKey, action: chex.Array) -> chex.Array:
        """input args:
        single_action_sample: tensor 'slice' of the full action tensor at a particular batch_index which corresponds to amplitudes and phases of the two pulses
        """
        # params = self.default_params
        # Carries out a full episode of system dynamics for a single action sample within a batch
        rng_noise, _rng = jax.random.split(key)

        if self.const_freq_pump == 0 and self.const_amp_stokes == 0:
            delta_p = action[0:self._n_steps].T
            delta_s = action[self._n_steps:2 * self._n_steps].T
            omega_p = action[2 * self._n_steps:3 * self._n_steps].T
            omega_s = action[3 * self._n_steps:4 * self._n_steps].T

        elif self.const_freq_pump == 1 and self.const_amp_stokes == 0:
            delta_p = action[0] * jnp.ones(self._n_steps,
                                           dtype=self.float_dtype).T
            delta_s = action[1:self._n_steps + 1].T
            omega_p = action[self._n_steps + 1:2 * self._n_steps + 1].T
            omega_s = action[2 * self._n_steps + 1:3 * self._n_steps + 1].T

        elif self.const_freq_pump == 0 and self.const_amp_stokes == 1:
            delta_p = action[0:self._n_steps].T
            delta_s = action[self._n_steps:2 * self._n_steps].T
            omega_p = action[2 * self._n_steps:3 * self._n_steps].T
            omega_s = action[3 * self._n_steps] * jnp.ones(
                self._n_steps, dtype=self.float_dtype).T

        elif self.const_freq_pump == 1 and self.const_amp_stokes == 1:
            delta_p = action[0] * jnp.ones(self._n_steps,
                                           dtype=self.float_dtype).T
            delta_s = action[1:self._n_steps + 1].T
            omega_p = action[self._n_steps + 1:2 * self._n_steps + 1].T
            omega_s = action[2 * self._n_steps + 1] * jnp.ones(
                self._n_steps, dtype=self.float_dtype).T

        # LINEAR INTERPOLATION
        # chirp_pump = LinearInterpolation(ts=self.ts_action, ys=delta_p)
        # env_pump = LinearInterpolation(ts=self.ts_action, ys=omega_p)

        # CUBIC INTERPOLATION
        chirp_pump_coeff = backward_hermite_coefficients(
            self.ts_action, delta_p)
        chirp_stokes_coeff = backward_hermite_coefficients(
            self.ts_action, delta_s)
        env_pump_coeff = backward_hermite_coefficients(self.ts_action, omega_p)
        env_stokes_coeff = backward_hermite_coefficients(
            self.ts_action, omega_s)

        chirp_pump = CubicInterpolation(ts=self.ts_action,
                                        coeffs=chirp_pump_coeff)
        chirp_stokes = CubicInterpolation(ts=self.ts_action,
                                          coeffs=chirp_stokes_coeff)
        env_pump = CubicInterpolation(ts=self.ts_action, coeffs=env_pump_coeff)
        env_stokes = CubicInterpolation(ts=self.ts_action,
                                        coeffs=env_stokes_coeff)

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
                 self._omega_r,
                 self._omega_e,
                 self._delta_r,
                 self._delta_e,
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

        def f_chirp_pump(t):
            if self.noise == "g":
                return jnp.abs(chirp_pump.evaluate(t) + g_noise_det_p)
            elif self.noise == "ou":
                return chirp_pump.evaluate(t) + ou_noise_chirp_pump.evaluate(t)
            else:
                return chirp_pump.evaluate(t)

        def f_chirp_stokes(t):
            if self.noise == "g":
                return jnp.abs(chirp_stokes.evaluate(t) + g_noise_det_s)
            elif self.noise == "ou":
                return chirp_stokes.evaluate(
                    t) + ou_noise_chirp_stokes.evaluate(t)
            else:
                return chirp_stokes.evaluate(t)

        def f_env_pump(t):
            if self.noise == "g":
                return jnp.abs(env_pump.evaluate(t) + g_noise_env_p)
            elif self.noise == "ou":
                return env_pump.evaluate(t) + ou_noise_env_pump.evaluate(t)
            else:
                return env_pump.evaluate(t)

        def f_env_stokes(t):
            if self.noise == "g":
                return jnp.abs(env_stokes.evaluate(t) + g_noise_env_s)
            elif self.noise == "ou":
                return env_stokes.evaluate(t) + ou_noise_env_stokes.evaluate(t)
            else:
                return env_stokes.evaluate(t)

        def f_blockade(t):
            if self.noise == "g":
                #TODO: reconsider noise scaling for blockade strength
                return jnp.abs(self.blockade_strength +
                               self.gaussian_noise_scaling_phase *
                               jax.random.normal(rng_noise, shape=(1, ))[0])
            elif self.noise == "ou":
                return self.blockade_strength + ou_noise_chirp_pump.evaluate(t)
            return self.blockade_strength

        def f_chirp_both(t):
            return f_chirp_pump(t) + f_chirp_stokes(t)

        if self.dissipator:
            yi = self.rho0
        else:
            yi = self.psi0

        vec_delta_p = jnp.vectorize(f_chirp_pump)
        vec_delta_s = jnp.vectorize(f_chirp_stokes)
        vec_delta_both = jnp.vectorize(f_chirp_both)
        vec_p_env = jnp.vectorize(f_env_pump)
        vec_s_env = jnp.vectorize(f_env_stokes)
        vec_blockade = jnp.vectorize(f_blockade)

        if self.dim == 2:
            # define signals
            signal_list = [
                Signal(vec_p_env),
                Signal(vec_s_env),
                Signal(vec_delta_p),
                Signal(vec_delta_s),
                Signal(vec_delta_both),
                Signal(vec_blockade),
            ]

            stepsize_controller = PIDController(rtol=1e-6, atol=1e-8)

            sol = self.solver.solve(
                t_span=[self.ts_sim[0], self.ts_sim[-1]],
                signals=signal_list,
                y0=yi,
                t_eval=self.ts_action,
                convert_results=False,
                #method="jax_odeint",
                #mxstep=self.mxstep,
                method=Dopri8(),
                max_steps=self.mxstep,
                throw=False,
                # method=Euler(),
                # dt0=1/50,
                stepsize_controller=stepsize_controller
                # max_dt = 1/350,
            )

            # NOTE if we want to return the full state history JAX can't jit the function
            max_steps_reached = (sol.result == RESULTS.max_steps_reached)
            num_steps = sol.stats["num_steps"]

            rho_f = sol.y[-1]
            rho_states = sol.y

        else:
            raise ValueError(
                "State Space must be 2 qubits, other qubit spaces not implemented"
            )

        # calculate average excited state population
        if self.dissipator:
    
            x_state_e = (jnp.mean(jnp.abs(rho_states[:, 4, 4])) +
                            jnp.mean(jnp.abs(rho_states[:, 5, 5])) +
                            jnp.mean(jnp.abs(rho_states[:, 8, 8]))) / 3
            x_state_r = (jnp.mean(jnp.abs(rho_states[:, 9, 9])) +
                            jnp.mean(jnp.abs(rho_states[:, 6, 6])) +
                            jnp.mean(jnp.abs(rho_states[:, 10, 10])) +
                            jnp.mean(jnp.abs(rho_states[:, 11, 11])) +
                            jnp.mean(jnp.abs(rho_states[:, 7, 7]))) / 5
        else:
        
            x_state_e = (jnp.mean(jnp.abs(rho_states[:, 4])) +
                            jnp.mean(jnp.abs(rho_states[:, 5])) +
                            jnp.mean(jnp.abs(rho_states[:, 8]))) / 3
            x_state_r = (jnp.mean(jnp.abs(rho_states[:, 7])) +
                            jnp.mean(jnp.abs(rho_states[:, 6])) +
                            jnp.mean(jnp.abs(rho_states[:, 9])) +
                            jnp.mean(jnp.abs(rho_states[:, 10])) +
                            jnp.mean(jnp.abs(rho_states[:, 11]))) / 8

        return (
            rho_f,
            x_state_e,
            x_state_r,
            max_steps_reached,
            num_steps,
            noise_samples_env_pump,
            noise_samples_chirp_pump,
        )

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

        # Preparing Action for Simulation
        if self.const_freq_pump == 0 and self.const_amp_stokes == 0:
            action_pump_det = action[0:self._n_steps].T
            action_stokes_det = action[self._n_steps:2 * self._n_steps].T
            action_pump_env = action[2 * self._n_steps:3 * self._n_steps].T
            action_stokes_env = action[3 * self._n_steps:4 * self._n_steps].T

        elif self.const_freq_pump == 1 and self.const_amp_stokes == 0:
            action_pump_det = action[0] * jnp.ones(self._n_steps,
                                                   dtype=self.float_dtype).T
            action_stokes_det = action[1:self._n_steps + 1].T
            action_pump_env = action[self._n_steps + 1:2 * self._n_steps + 1].T
            action_stokes_env = action[2 * self._n_steps +
                                       1:3 * self._n_steps + 1].T

        elif self.const_freq_pump == 0 and self.const_amp_stokes == 1:
            action_pump_det = action[0:self._n_steps].T
            action_stokes_det = action[self._n_steps:2 * self._n_steps].T
            action_pump_env = action[2 * self._n_steps:3 * self._n_steps].T
            action_stokes_env = action[3 * self._n_steps] * jnp.ones(
                self._n_steps, dtype=self.float_dtype).T

        elif self.const_freq_pump == 1 and self.const_amp_stokes == 1:
            action_pump_det = action[0] * jnp.ones(self._n_steps,
                                                   dtype=self.float_dtype).T
            action_stokes_det = action[1:self._n_steps + 1].T
            action_pump_env = action[self._n_steps + 1:2 * self._n_steps + 1].T
            action_stokes_env = action[2 * self._n_steps + 1] * jnp.ones(
                self._n_steps, dtype=self.float_dtype).T

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
            jnp.clip(action_stokes_env[:], a_min=0.0, a_max=1.0))

        pump_env = action_pump_env.astype(self.float_dtype)
        stokes_env = action_stokes_env.astype(self.float_dtype)

        pump_det = action_pump_det.astype(self.float_dtype)
        stokes_det = action_stokes_det.astype(self.float_dtype)

        # smoothen amplitude
        pump_env = self.signal_tools.drive_smoother(pump_env)
        if self.const_amp_stokes == 0:
            stokes_env = self.signal_tools.drive_smoother(stokes_env)

        # smoothen detuning
        if self.const_freq_pump == 0:
            pump_det = self.signal_tools.drive_smoother(pump_det, False)
        stokes_det = self.signal_tools.drive_smoother(stokes_det, False)

        if self.fixed_endpoints:
            stokes_env = stokes_env.at[0].set(0)
            stokes_env = stokes_env.at[-1].set(0)
            pump_env = pump_env.at[0].set(0)
            pump_env = pump_env.at[-1].set(0)

        #scale pulses
        stokes_env *= self._omega_r
        pump_env *= self._omega_e
        pump_det *= self._delta_e
        stokes_det *= self._delta_r

        stokes_env = stokes_env.astype(self.float_dtype)
        pump_env = pump_env.astype(self.float_dtype)
        pump_det = pump_det.astype(self.float_dtype)
        stokes_det = stokes_det.astype(self.float_dtype)

        # Simulation and Obtaining Reward + Params for New State
        action_concatenated = jnp.concatenate(
            (pump_det, stokes_det, pump_env, stokes_env),
            dtype=self.float_dtype)
        single_result = self.calc_step(key, action_concatenated)
        reward, updated_state_floats, noise_a, noise_freq, mx_steps, num_steps = self.calc_reward_and_state(
            single_result, action_concatenated)

        noise_a = noise_a.astype(self.float_dtype)
        noise_freq = noise_freq.astype(self.float_dtype)

        # Setting New State with Updated State Array,
        # New Optimal Action (for logging),
        # and New Timestep
        updated_state = EnvState(
            *updated_state_floats,
            pump_env,
            stokes_env,
            pump_det,
            stokes_det,
            mx_steps,
            num_steps,
            noise_a,
            noise_freq,
            new_timestep,
        )

        done = True
        return (
            lax.stop_gradient(self.get_obs()),
            lax.stop_gradient(updated_state),
            reward,
            done,
            lax.stop_gradient(self.get_info(updated_state)),
        )

    def extract_values(self, results: chex.Array, actions: chex.Array):
        """Takes in values from simulation and returns fidelity and smoothness
        input args:
        results: density matrix of final timestep from simulation
        actions: action tensor"""

        # Extracting Fidelity and Smoothness Values

        rho_f, x_e, x_r = results

       
        if self.dissipator:
            max_fid = mixed_pure_state_fidelity(self.psi_des, rho_f)
            #psi_angles = [
            #    jnp.angle(rho_f[0, 0]),
            #    jnp.angle(rho_f[0, 1]),
            #    jnp.angle(rho_f[0, 2]),
            #    jnp.angle(rho_f[0, 3])
            #]
            #psi_des_angles = [0, 0, 0, jnp.pi]
            #max_fid = jnp.abs(
            #    rho_f[0, 0] +
            #    jnp.exp(-1j * (psi_des_angles[1] + psi_angles[1])) *
            #    rho_f[0, 1] +
            #    jnp.exp(-1j * (psi_des_angles[2] + psi_angles[2])) *
            #    rho_f[0, 1] +
            #    jnp.exp(-1j * (psi_des_angles[3] + 2 * psi_angles[3])) *
            #    rho_f[0, 3] +
            #    )**2
            #
            #max_fid = jnp.abs(rho_f[0, 0] +
            #                  jnp.exp(-1j *
            #                          (psi_des_angles[1])) * rho_f[1, 0] +
            #                  jnp.exp(-1j *
            #                          (psi_des_angles[2])) * rho_f[2, 0] +
            #                  jnp.exp(-1j * (psi_des_angles[3])) *
            #                  rho_f[3, 0])**2

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
        area_p = self.signal_tools.area_of_pulse(env_p)
        area_s = self.signal_tools.area_of_pulse(env_s)

        return max_fid, x_e, x_r, smoothness_p, smoothness_s, smoothness_det_p, smoothness_det_s, area_p, area_s

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
            x_e,
            x_r,
            smoothness_p,
            smoothness_s,
            smoothness_det_p,
            smoothness_det_s,
            area_p,
            area_s,
        ) = self.extract_values(result[0:3], drive)

        mx_steps_reached = result[3]
        num_steps = result[4]

        if self.noise == "ou":
            noise_amp = result[5]
            noise_freq = result[6]
        else:
            noise_amp = jnp.zeros(self._n_steps *
                                  self.noise_over_sampling_rate)
            noise_freq = jnp.zeros(self._n_steps *
                                   self.noise_over_sampling_rate)

        if self.log_fidelity:
            fid_reward = jnp.log(1 / (1 - fid_state_transfer))
        else:
            fid_reward = fid_state_transfer

        def true_fun(_):
            mx_reward = 1
            fid_reward = 0.0
            fid_state_transfer = 0.0
            x_e = 0.0
            x_r = 0.0
            return mx_reward, fid_reward, fid_state_transfer, x_e, x_r

        # Modify the false_fun to keep fid_reward and excited_population unchanged
        def false_fun(_):
            mx_reward = 0
            # Since we do not modify fid_reward and excited_population in false_fun,
            # they will remain unchanged.
            return mx_reward, fid_reward, fid_state_transfer, x_e, x_r

        # Use lax.cond to conditionally execute either true_fun or false_fun
        mx_reward, fid_reward, fid_state_transfer, x_e, x_r = lax.cond(
            mx_steps_reached, true_fun, false_fun, operand=None)

        reward = (
            self.fid_factor * fid_reward - self.x_penalty *
            (self.e_dissipation * x_e + self.r_dissipation * x_r) -
            self.pulse_area_penalty *
            (area_p / self.baseline_area_r + area_s / self.baseline_area_e) / 2
            - self.smoothness_penalty_env *
            (relu((smoothness_s) /
                  (self.baseline_smoothness_amp_e) - 1) + relu(
                      (smoothness_p) / (self.baseline_smoothness_amp_r) - 1)) -
            self.smoothness_penalty_det *
            (relu((smoothness_det_s) /
                  (self.baseline_smoothness_freq_e) - 1) + relu(
                      (smoothness_det_p) /
                      (self.baseline_smoothness_freq_r) - 1)) -
            self.mx_step_penalty * mx_reward)

        state = jnp.array(
            [
                reward,
                fid_state_transfer,
                x_e,
                x_r,
                area_p,
                area_s,
                smoothness_p,
                smoothness_s,
                smoothness_det_p,
                smoothness_det_s,
            ],
            dtype=self.float_dtype,
        )

        return reward, state, noise_amp, noise_freq, mx_steps_reached, num_steps

    def reset(self, key: chex.PRNGKey,
              params: EnvParams) -> Tuple[chex.Array, EnvState]:
        """IMPORTANT Reset Environment, in this case nothing needs to be done
        so default obs and info are returned"""
        # self.precompile()
        state = EnvState(
            reward=0.0,
            fid=0.0,
            x_e=0.0,
            x_r=0.0,
            area_p=0.0,
            area_s=0.0,
            smoothness_p=0.0,
            smoothness_s=0.0,
            smoothness_det_p=0.0,
            smoothness_det_s=0.0,
            env_p=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            env_s=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            det_p=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            det_s=jnp.zeros(self._n_steps, dtype=self.float_dtype),
            max_steps_reached=False,
            num_steps=0,
            noise_amp=jnp.zeros(self._n_steps * self.noise_over_sampling_rate,
                                dtype=self.float_dtype),
            noise_freq=jnp.zeros(self._n_steps * self.noise_over_sampling_rate,
                                 dtype=self.float_dtype),
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
            "fid": env_state.fid,
            "e-state-pop-avg": env_state.x_e,
            "r-state-pop-avg": env_state.x_r,
            "area-p": env_state.area_p,
            "area-s": env_state.area_s,
            "smoothness-p": env_state.smoothness_p,
            "smoothness-s": env_state.smoothness_s,
            "smoothness-det-p": env_state.smoothness_det_p,
            "smoothness-det-s": env_state.smoothness_det_s,
            "env-p": env_state.env_p,
            "env-s": env_state.env_s,
            "det-p": env_state.det_p,
            "det-s": env_state.det_s,
            "max-steps-reached": env_state.max_steps_reached,
            "num-steps": env_state.num_steps,
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
        return "Rydberg Gate for two photon resonance"

    @property
    def num_actions(self, params: Optional[EnvParams] = EnvParams) -> int:
        """IMPORTANT number of actions"""
        if self.const_amp_stokes == 0 and self.const_freq_pump == 0:
            return self._n_steps * 4
        elif self.const_amp_stokes == 1 and self.const_freq_pump == 0:
            return 3 * self._n_steps + 1
        elif self.const_amp_stokes == 0 and self.const_freq_pump == 1:
            return 3 * self._n_steps + 1
        elif self.const_amp_stokes == 1 and self.const_freq_pump == 1:
            return 2 * self._n_steps + 2
        else:
            raise ValueError("Invalid Configuration of Constants")

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Box:
        """IMPORTANT action space shape"""
        if params is None:
            params = self.default_params

        if self.const_amp_stokes == 0 and self.const_freq_pump == 0:
            _shape = 4 * self._n_steps
        elif self.const_amp_stokes == 1 and self.const_freq_pump == 0:
            _shape = 3 * self._n_steps + 1
        elif self.const_amp_stokes == 0 and self.const_freq_pump == 1:
            _shape = 3 * self._n_steps + 1
        elif self.const_amp_stokes == 1 and self.const_freq_pump == 1:
            _shape = 2 * self._n_steps + 2
        else:
            raise ValueError("Invalid Configuration of Constants")

        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(_shape, ),
            dtype=self.float_dtype,
        )

    def observation_space(self,
                          params: Optional[EnvParams] = None) -> spaces.Box:
        """IMPORTANT observation space shape"""
        return spaces.Box(0, 1.0, shape=(1, ), dtype=self.float_dtype)


# Test the environment
if __name__ == "__main__":
    env = RydbergTwoEnv(gamma=1,
                        blockade_strength=500,
                        omega_e=50,
                        omega_r=50,
                        delta_r=50,
                        delta_e=2500,
                        mxstep=4096,
                        noise="None",
                        log_fidelity=1,
                        dissipator=True,
                        const_freq_pump=1)

    # Seeding
    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)
    init_obs, init_state = env.reset(_rng, env.default_params)
    input_action = jax.random.uniform(_rng, (4 * env._n_steps, ),
                                      minval=-1.0,
                                      maxval=1.0)
    env_params = env.default_params

    # Testing single step
    jit_step = jax.jit(env.step)
    obs, state, reward, done, info = jit_step(_rng, init_state, input_action,
                                              env_params)

    start = time.time()
    obs, state, reward, done, info = block_until_ready(
        jit_step(_rng, init_state, input_action, env_params))
    end = time.time()
    #print(f"Fidelity: {state.fid}")
    print(f"state: {state}")
    print(
        f"Time for Single Jitted env.step: {jnp.round((end - start)/1e-6)}us")

    ### Testing vmapped evaluation
    vec_env = VecEnv(env)
    vec_env_params = env.default_params

    ### init env
    timings_noise_free = []

    for num_envs in [
            1, 64, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2, 4 * 8192
    ]:
        rng, _rng = jax.random.split(rng)
        rng_reset = jax.random.split(_rng, num_envs)
        _, vec_env_state = vec_env.reset(rng_reset, vec_env_params)

        # making the action
        rng, _rng, _rng_act = jax.random.split(rng, 3)
        rng_step = jax.random.split(_rng, num_envs)

        vectorised_action = jax.random.uniform(_rng_act,
                                               (num_envs, 4 * env._n_steps),
                                               minval=-1.0,
                                               maxval=1.0)

        jit_vmap_step = jax.jit(vec_env.step)
        returned_obsv, new_env_state, reward, done, info = jit_vmap_step(
            rng_step, vec_env_state, vectorised_action, vec_env_params)

        start = time.time()
        returned_obsv, new_env_state, reward, done, info = block_until_ready(
            jit_vmap_step(rng_step, vec_env_state, vectorised_action,
                          vec_env_params))
        end = time.time()

        total_time = end - start

        print(
            f"Time taken for batch of {num_envs} steps in vmap: {total_time*1000}ms"
        )
        print(f"Time taken per env step: {total_time / num_envs *1000}ms")
        timings_noise_free.append(total_time / num_envs * 1000)

    print(timings_noise_free)

    #test with ou noise
    env = RydbergTwoEnv(gamma=1,
                        blockade_strength=100,
                        omega_e=20,
                        omega_r=20,
                        delta_r=20,
                        delta_e=100,
                        mxstep=4096,
                        noise="ou",
                        dissipator=True,
                        const_freq_pump=0)

    # Test the step function
    # amplitudes (first actions) are between -1 and 1, and scaled within the environment
    pump = jnp.array(blackman(50), dtype=env.float_dtype)
    stokes = jnp.array(blackman(50), dtype=env.float_dtype)
    stokes = stokes.at[:].set(1 - stokes[:])
    det_stokes = jnp.array(-1 * blackman(50), dtype=env.float_dtype)
    det_pump = -500 * jnp.ones(50, dtype=env.float_dtype)

    nested_sample = jnp.array([det_pump, det_stokes, pump, stokes],
                              dtype=env.float_dtype)
    input_action = jnp.ravel(nested_sample)

    # Seeding
    seed = 0
    rng = jax.random.PRNGKey(seed)
    rng, _rng = jax.random.split(rng)

    env_params = env.default_params
    init_obs, init_state = env.reset(_rng, env_params)

    # Testing single step
    jit_step = jax.jit(env.step)
    obs, state, reward, done, info = jit_step(_rng, init_state, input_action,
                                              env_params)

    start = time.time()
    obs, state, reward, done, info = block_until_ready(
        jit_step(_rng, init_state, input_action, env_params))
    end = time.time()
    #print(f"Fidelity: {state.fid}")
    print(
        f"Time for Single Jitted env.step: {jnp.round((end - start)/1e-6)}us")

    ### Testing vmapped evaluation
    vec_env = VecEnv(env)
    vec_env_params = env.default_params

    ### init env
    timings_noise = []

    for num_envs in [
            1, 64, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2, 4 * 8192
    ]:
        rng, _rng = jax.random.split(rng)
        rng_reset = jax.random.split(_rng, num_envs)
        _, vec_env_state = vec_env.reset(rng_reset, vec_env_params)

        # making the action
        rng, _rng, _rng_act = jax.random.split(rng, 3)
        rng_step = jax.random.split(_rng, num_envs)

        vectorised_action = jax.random.uniform(_rng_act,
                                               (num_envs, 4 * env._n_steps),
                                               minval=-1.0,
                                               maxval=1.0)

        jit_vmap_step = jax.jit(vec_env.step)
        returned_obsv, new_env_state, reward, done, info = jit_vmap_step(
            rng_step, vec_env_state, vectorised_action, vec_env_params)

        start = time.time()
        returned_obsv, new_env_state, reward, done, info = block_until_ready(
            jit_vmap_step(rng_step, vec_env_state, vectorised_action,
                          vec_env_params))
        end = time.time()

        total_time = end - start

        print(
            f"Time taken for batch of {num_envs} steps in vmap: {total_time*1000}ms"
        )
        print(f"Time taken per env step: {total_time / num_envs *1000}ms")
        timings_noise.append(total_time / num_envs * 1000)

    print(timings_noise)
