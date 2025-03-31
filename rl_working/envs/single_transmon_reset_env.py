# Standard Imports
from typing import Tuple, Optional, List
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal.windows import blackman
import time

# JAX Imports
import jax
import jax.numpy as jnp
from jax import jit, lax, config, vmap, block_until_ready
import jax.numpy as jnp
from jax.scipy.special import erf
from jax.scipy.integrate import trapezoid
from jax.nn import relu
from gymnax.environments import spaces
from flax import struct
import chex
from rl_working.envs.environment_template import SingleStepEnvironment

# Check if GPU is available
if "cuda" in str(jax.devices()):
    print("Connected to a GPU")
    processor_array_type = "jax"
    processor = "gpu"
    float_dtype = jnp.float32
    complex_dtype = jnp.complex64
else:
    jax.config.update("jax_platform_name", "cpu")
    print("Not connected to a GPU")
    processor_array_type = "jax_sparse"
    jax.config.update("jax_enable_x64", True)
    processor = "cpu"
    float_dtype = jnp.float64
    complex_dtype = jnp.complex128

# Qiskit Imports
from qiskit_dynamics import Solver, Signal

from diffrax import LinearInterpolation, PIDController, Tsit5, Dopri8, Heun

@struct.dataclass
class EnvStateDetuning:
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
    steps: float
    max_steps_pen: float
    mean_deviation: float
    deviation_reward: float
    action: chex.Array
    timestep: int

@struct.dataclass
class EnvParamsDetuning:
    """
    Flax Dataclass used to store Static Environment Params
    All static env params should be kept here, though they can be equally kept
    in the Jax class as well
    """

    t1: float

    window_length: Optional[int] = 8
    kernel: Optional[chex.Array] = jnp.ones(window_length) / window_length
    gauss_mean: Optional[int] = 0.0
    gauss_std: Optional[int] = 1.0
    small_window: Optional[chex.Array] = jnp.linspace(
        -0.5 * (window_length - 1), 0.5 * (window_length - 1), window_length
    )
    gauss_kernel: Optional[chex.Array] = (
        1
        / (jnp.sqrt(2 * jnp.pi) * gauss_std)
        * jnp.exp(-((small_window - gauss_mean) ** 2) / (2 * gauss_std**2))
    )

    smearing_window_length: Optional[int] = 30
    smearing_max: Optional[float] = 3.
    smearing_std: Optional[float] = 4.
    smearing_gauss_waveform: Optional[chex.Array] = (
        jnp.exp(-jnp.linspace(-smearing_max, smearing_max, smearing_window_length)**2 / (2 * smearing_std**2))
    ).at[int(0.5 * (smearing_window_length - 1))].set(0.)
    smearing_gauss_kernel: Optional[chex.Array] = smearing_gauss_waveform / jnp.sum(smearing_gauss_waveform)

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
    Jax Compatible Environment for Finding Optimal Transmon
    and Resonator Reset Environments
    """
    def __init__(
        self,
        kappa,
        chi,
        delta,
        anharm,
        g_coupling,
        gamma,
        omega_max,
        delta_max,
        sim_t1,
        transmon_reset_coeff,
        deviation_coeff,
        smoothness_coeff,
        amp_pen_coeff,
        steps_pen_coeff,
        max_grad,
        k_factor,
        max_deviation,
        max_steps
    ):
        super().__init__()
        self.int_dtype = jnp.int16
        self.float_dtype = float_dtype
        self.complex_dtype = complex_dtype

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

        self._g_bar_factor = 1. / jnp.sqrt(2) * self._g_coupling * (self._anharm) / (self._delta * (self._delta + self._anharm))

        self._n_trans = 3
        self._n_res = 2

        self.photon_limit = 1.

        self._t1 = sim_t1

        params = self.default_params

        self.len_ts_sim = params.num_sim
        self.len_ts_sim_debug = params.num_sim_debug
        self.len_ts_action = params.num_actions

        self.ts_sim = jnp.linspace(0.0, self._t1, self.len_ts_sim, dtype=self.float_dtype)
        self.ts_sim_debug = jnp.linspace(0.0, self._t1, self.len_ts_sim_debug, dtype=self.float_dtype)
        self.ts_action = jnp.linspace(0.0, self._t1, self.len_ts_action, dtype=self.float_dtype)

        self.ode_solver = Dopri8()
        self.pid_controller = PIDController(
            pcoeff=0.4,
            icoeff=0.3,
            dcoeff=0.,
            rtol=1e-3,
            atol=1e-4,
            jump_ts=self.ts_action
        )
        self.max_steps = max_steps

        a = jnp.diag(jnp.sqrt(jnp.arange(1, self._n_res)), 1)
        adag = jnp.diag(jnp.sqrt(jnp.arange(1, self._n_res)), -1)
        q = jnp.diag(jnp.sqrt(jnp.arange(1, self._n_trans)), 1)
        qdag = jnp.diag(jnp.sqrt(jnp.arange(1, self._n_trans)), -1)

        trans_ident = jnp.eye(self._n_trans, dtype=self.complex_dtype)
        res_ident = jnp.eye(self._n_res, dtype=self.complex_dtype)

        self.a_op = jnp.kron(trans_ident, a)
        self.ad_op = jnp.kron(trans_ident, adag)
        self.q_op = jnp.kron(q, res_ident)
        self.qd_op = jnp.kron(qdag, res_ident)

        self.H_STATIC = 0.5 * self._chi * self.qd_op @ self.q_op @ self.ad_op @ self.a_op
        self.H_DISSIPATE = [
            jnp.sqrt(self._kappa) * self.a_op, 
            jnp.sqrt(self._gamma) * self.q_op
        ]
        self.H_DRIVE = [
            self._g_bar_factor * (self.qd_op @ self.qd_op @ self.a_op + self.ad_op @ self.q_op @ self.q_op) / jnp.sqrt(2),
            # self._k_factor * self.qd_op @ self.q_op
            self.qd_op @ self.q_op,
        ]

        # initial state is |f0>
        f_ket = jnp.zeros(self._n_trans)
        f_ket = f_ket.at[2].set(1.)

        res_ket = jnp.zeros(self._n_res)
        res_ket = res_ket.at[0].set(1.)

        f0_ket = jnp.kron(f_ket, res_ket)
        self.f0_dm = jnp.outer(f0_ket, f0_ket)

        self.transmon_reset_coeff = transmon_reset_coeff
        self.deviation_coeff = deviation_coeff
        self.smoothness_coeff = smoothness_coeff
        self.amp_pen_coeff = amp_pen_coeff
        self.steps_pen_coeff = steps_pen_coeff

        # self._pulse_complete_ind = jnp.argmin(jnp.abs(self.ts_sim - first_min_time))
        # not needed for transmon, starts from max goes to min

        self.kernel = params.gauss_kernel
        self.smearing_kernel = params.smearing_gauss_kernel

        self.pulse_dt = self.ts_action[1] - self.ts_action[0]
        self.max_grad = max_grad

        self.solver = Solver(
            static_hamiltonian=self.H_STATIC,
            hamiltonian_operators=self.H_DRIVE,
            static_dissipators=self.H_DISSIPATE,
            rotating_frame=self.H_STATIC
            # validate=True,
        )

        self.opt_time = self.determine_theoretical_duration()
        self.ref_waveform = jnp.heaviside(self.opt_time - self.ts_action, 1.)

        self.max_deviation = max_deviation

        dummy_key = jax.random.PRNGKey(seed=0)
        _, self._state = self.reset_env(key=dummy_key, params=None)
        self.log_vals = self.get_info(self._state)

    def determine_theoretical_duration(self):
        max_amp_waveform = jnp.heaviside(0.4 - self.ts_action, 1.) * self._omega_max
        detuning_waveform = jnp.zeros_like(max_amp_waveform) * self._delta_max

        old_k_factor = self._k_factor

        # set k value to zero
        self._k_factor = 0.

        results, steps = self.calc_results_debug(max_amp_waveform, detuning_waveform)
        transmon_pop = jnp.trace(results @ self.qd_op @ self.q_op, axis1=1, axis2=2).real
        time_of_min = self.ts_sim_debug[jnp.argmin(transmon_pop)]
        self._k_factor = old_k_factor
        return time_of_min

    @property
    def default_params(self) -> EnvParamsDetuning:
        """
        IMPORTANT Retrieving the Default Env Params
        """
        return EnvParamsDetuning(
            t1=self._t1,
            min_action=-1.,
            max_action=1.,
        )
    
    def drive_smoother(self, res_drive: chex.Array):
        """Physics Specific Function"""
        conv_result = jnp.convolve(res_drive, self.kernel, mode="same")
        return conv_result
    
    def drive_smearer(self, res_drive):
        return jnp.convolve(res_drive, self.smearing_kernel, mode='same')
    
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
        max_grad = self.max_grad # for t1=0.8 using a gaussian square as used in IBM

        output_waveform = jnp.zeros_like(ref_waveform)

        def update_array_at_index(ind, waveform):
            diff = ref_waveform[ind] - waveform[ind - 1]
            diff_clipped = jnp.clip(diff, a_min=-pulse_dt * max_grad, a_max=pulse_dt * max_grad)
            waveform = waveform.at[ind].set(waveform[ind - 1] + diff_clipped)
            return waveform
        
        output_waveform = jax.lax.fori_loop(
            lower=1, 
            upper=len(ref_waveform),
            body_fun=update_array_at_index,
            init_val=output_waveform
        )

        return output_waveform

    def prepare_trans_action(self, trans_amp, trans_detuning):
        trans_drive = trans_amp.astype(self.float_dtype)  # Scale up Action
        trans_normed_drive = self.normalize_pulse(trans_drive)
        trans_clipped_drive = jnp.clip(trans_normed_drive, a_min=0., a_max=1.)
        trans_drive = self.drive_smoother(trans_clipped_drive)  # Apply Smoother
        # res_drive = self.drive_smearer(res_drive) # Apply Drive Smearing
        # res_drive = self.limit_gradient(res_drive) # Apply Instantaneous Gradient Limits
        
        trans_detuning = trans_detuning.astype(self.float_dtype)
        trans_normed_detuning = self.normalize_pulse(trans_detuning)
        trans_detuning = self.drive_smoother(trans_normed_detuning)

        # Calculate Predicted Stark Shift
        mean_deviation = jnp.mean(jnp.abs(trans_drive - self.ref_waveform))
        trans_drive *= jnp.heaviside(self.max_deviation - mean_deviation, 1.)

        return self._omega_max * trans_drive, self._delta_max * trans_detuning, mean_deviation

    def step_env(
        self, key: chex.PRNGKey, state: EnvStateDetuning, action: chex.Array, params: EnvParamsDetuning
    ) -> Tuple[chex.Array, EnvStateDetuning, chex.Array, bool, dict]:
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
        trans_drive, trans_detuning, mean_deviation = self.prepare_trans_action(action_amp, action_detuning)

        # Simulation and Obtaining Reward + Params for New State
        result, steps = self.calc_results(trans_drive, trans_detuning)
        reward, updated_state_array = self.calc_reward_and_state(
            key,
            result.astype(self.float_dtype),
            steps,
            mean_deviation,
            action_amp,
            action_detuning,
        )

        env_state = EnvStateDetuning(*updated_state_array, action, new_timestep)

        self._state = env_state
        self.log_vals = self.get_info(self._state)

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

        transmon_pop = jnp.trace(results @ self.qd_op @ self.q_op, axis1=1, axis2=2).real

        # End Transmon Pop
        pulse_reset_transmon = transmon_pop[-1] # Transmon Pop at end of sim

        # Processing Raw Action without Smearing
        raw_action_amp = self.normalize_pulse(action_amp)
        clipped_action_amp = jnp.clip(raw_action_amp, a_min=0., a_max=1.)
        smooth_action_amp = self.drive_smoother(clipped_action_amp)
        
        raw_action_detuning = self.normalize_pulse(action_detuning)
        smooth_action_detuning = self.drive_smoother(raw_action_detuning)

        pulse_reset_val = jnp.abs(
            smooth_action_amp[-1]
        ) + jnp.abs(smooth_action_amp[0])

        mean_smooth_waveform_difference = 0.5 * jnp.sum(jnp.abs(smooth_action_amp - raw_action_amp)) / self._t1
        mean_smooth_waveform_difference += 0.5 * jnp.sum(jnp.abs(smooth_action_detuning - raw_action_detuning)) / self._t1

        def valid_steps():
            return jnp.array([
                mean_smooth_waveform_difference,
                pulse_reset_val,
                pulse_reset_transmon,
                0.
            ])
        
        def invalid_steps():
            return jnp.array([
                100.,
                2.,
                2.,
                1.
            ])
        
        reward_calc_vals = jax.lax.select(
            (steps < self.max_steps) * (mean_deviation < self.max_deviation),
            valid_steps(),
            invalid_steps()
        )

        return reward_calc_vals

    def calc_results(
        self, trans_drive: chex.Array, trans_detuning: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Physics Specific Function, Function used for ODE Simulation"""
        params = self.default_params

        f0g1_control = LinearInterpolation(ts=self.ts_action, ys=trans_drive)
        stark_control = LinearInterpolation(
            ts=self.ts_action,
            ys=self._k_factor * (trans_drive**2 - self._omega_max**2) + trans_detuning
        )

        def get_f0g1(t):
            return f0g1_control.evaluate(t)
        def get_stark(t):
            return stark_control.evaluate(t)
        
        vec_get_f0g1 = jnp.vectorize(get_f0g1)
        vec_get_stark = jnp.vectorize(get_stark)

        signals = [
            Signal(vec_get_f0g1),
            Signal(vec_get_stark),
        ]

        sol = self.solver.solve(
            t_span=[0., self.ts_sim[-1]],
            signals=signals,
            y0=self.f0_dm,
            t_eval=self.ts_sim,
            # convert_results=True,
            method=self.ode_solver,
            stepsize_controller=self.pid_controller,
            max_steps=self.max_steps,
            throw=False
        )

        return sol.y, sol.stats["num_steps"]

    def calc_results_debug(
        self, trans_drive: chex.Array, trans_detuning: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, chex.Array]:
        """Physics Specific Function, Function used for ODE Simulation"""
        params = self.default_params

        f0g1_control = LinearInterpolation(ts=self.ts_action, ys=trans_drive)
        stark_control = LinearInterpolation(
            ts=self.ts_action,
            ys=self._k_factor * (trans_drive**2 - self._omega_max**2) + trans_detuning
        )

        def get_f0g1(t):
            return f0g1_control.evaluate(t)
        def get_stark(t):
            return stark_control.evaluate(t)
        
        vec_get_f0g1 = jnp.vectorize(get_f0g1)
        vec_get_stark = jnp.vectorize(get_stark)

        signals = [
            Signal(vec_get_f0g1),
            Signal(vec_get_stark),
        ]

        sol = self.solver.solve(
            t_span=[0., self.ts_sim[-1]],
            signals=signals,
            y0=self.f0_dm,
            t_eval=self.ts_sim_debug,
            # convert_results=True,
            method=self.ode_solver,
            stepsize_controller=self.pid_controller,
            max_steps=self.max_steps,
            throw=False
        )

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
        ) = self.extract_values(_rng, result, steps, mean_deviation, action, detuning)
        # The above function holds physics-specific details

        reward = (
            + self.transmon_reward(pulse_reset_transmon)
            + self.smoothness_reward(mean_smooth_waveform_difference)
            + self.amp_reward(pulse_reset_val)
            + self.deviation_reward(mean_deviation)
            - self.max_steps_pen(max_steps_flag)
        )

        state = jnp.array(
            [
                reward,
                pulse_reset_transmon,
                self.transmon_reward(pulse_reset_transmon),
                mean_smooth_waveform_difference,
                self.smoothness_reward(mean_smooth_waveform_difference),
                pulse_reset_val,
                self.amp_reward(pulse_reset_val),
                steps,
                -self.max_steps_pen(max_steps_flag),
                mean_deviation,
                self.deviation_reward(mean_deviation)
            ],
            dtype=self.float_dtype,
        )

        return (reward, state)

    # def time_reward(self, pulse_end_time):
    #     return -self.time_coeff * pulse_end_time

    def transmon_reward(self, pulse_reset_transmon):
        return -self.transmon_reset_coeff * jnp.log10(pulse_reset_transmon)

    def smoothness_reward(self, mean_smooth_waveform_difference):
        return -self.smoothness_coeff * mean_smooth_waveform_difference

    def amp_reward(self, pulse_reset_val):
        return -self.amp_pen_coeff * pulse_reset_val
    
    def max_steps_pen(self, max_steps_flag):
        return self.steps_pen_coeff * max_steps_flag

    def deviation_reward(self, mean_deviation):
        return -self.deviation_coeff * relu(mean_deviation - self.max_deviation)

    def rollout_action(
        self,
        key: chex.PRNGKey,
        action: chex.Array,
        time_below_photon_val: Optional[float] = 0.1,
        photon_log_scale: Optional[bool] = False,
        bound_plots: Optional[bool] = True,
    ):
        rng, _rng = jax.random.split(key)
        ts_sim = self.ts_sim
        ts_action = self.ts_action

        fig, ax = plt.subplots(3, figsize=(8.0, 16.0))

        # Obtaining Raw Action vs Smooth Action
        raw_action = action * self.a0
        normalizing_factor = jnp.clip(
            self.mu * self.a0 / jnp.absolute(raw_action),
            0.0,
            1.0,
        )
        raw_action *= normalizing_factor
        smooth_action = self.prepare_action(action)

        # Defining Default Gaussian Square Readout
        # Gaussian Edge with sigma and duration
        # Constant Amplitude for certain duration
        total_default_duration = self.tau_0
        dt = 0.00045
        gaussian_edge_sigma = 64 * dt
        gaussian_edge_duration = 2.0 * gaussian_edge_sigma
        gaussian_square_readout = self.a0 * jnp.heaviside(
            ts_action - gaussian_edge_duration, 0.0
        )
        gaussian_square_readout *= jnp.heaviside(
            total_default_duration - gaussian_edge_duration - ts_action, 1.0
        )
        gaussian_square_readout += (
            self.a0
            * jnp.exp(
                -((ts_action - gaussian_edge_duration) ** 2)
                / (2 * gaussian_edge_sigma**2)
            )
            * jnp.heaviside(gaussian_edge_duration - ts_action, 1.0)
        )
        gaussian_square_readout += (
            self.a0
            * jnp.exp(
                -((ts_action - (total_default_duration - gaussian_edge_duration)) ** 2)
                / (2 * gaussian_edge_sigma**2)
            )
            * jnp.heaviside(gaussian_edge_duration - ts_action, 1.0)
        )

        # Obtaining Results for Smooth Action and Gaussian Square
        rng, _rng = jax.random.split(rng)
        smooth_results = self.calc_results(smooth_action)
        (
            s_max_pf,
            s_max_photon,
            s_photon_reset_time,
            s_real_photon_reset_time,
            s_pulse_end_time,
            s_max_pf_time,
            s_smoothness,
            s_bandwidth,
            s_pF,
            s_higher_photons,
            s_pulse_reset_val,
        ) = self.extract_values(_rng, smooth_results, smooth_action, action)

        s_photon_reset_time = jnp.round(s_photon_reset_time, 3)
        s_pulse_end_time = jnp.round(s_pulse_end_time, 3)
        s_max_pf_time = jnp.round(s_max_pf_time, 3)

        rng, _rng = jax.random.split(rng)
        gaussian_results = self.calc_results(gaussian_square_readout)
        (
            g_max_pf,
            g_max_photon,
            g_photon_reset_time,
            g_real_photon_reset_time,
            g_pulse_end_time,
            g_max_pf_time,
            g_smoothness,
            g_bandwidth,
            g_pF,
            g_higher_photons,
            g_pulse_reset_val,
        ) = self.extract_values(
            _rng,
            gaussian_results,
            gaussian_square_readout,
            gaussian_square_readout / self.a0,
        )

        g_photon_reset_time = jnp.round(g_photon_reset_time, 3)
        g_pulse_end_time = jnp.round(g_pulse_end_time, 3)
        g_max_pf_time = jnp.round(g_max_pf_time, 3)

        smooth_g = smooth_results[:, 0] + 1.0j * smooth_results[:, 1]
        smooth_e = smooth_results[:, 2] + 1.0j * smooth_results[:, 3]
        gaussian_g = gaussian_results[:, 0] + 1.0j * gaussian_results[:, 1]
        gaussian_e = gaussian_results[:, 2] + 1.0j * gaussian_results[:, 3]

        s_higher_photons = jnp.abs(smooth_g) ** 2
        g_higher_photons = jnp.abs(gaussian_g) ** 2

        if bound_plots:
            ax[0].set_xlim(left=None, right=self._t1)
            ax[1].set_xlim(left=None, right=self._t1)
            ax[2].set_xlim(left=None, right=self._t1)

        ax[0].plot(ts_action, raw_action, label="Raw Action", alpha=0.5)
        ax[0].plot(ts_action, smooth_action, label="Smooth Action")
        ax[0].plot(ts_action, gaussian_square_readout, label="Default Gaussian Square")
        ax[0].axvline(
            x=s_max_pf_time,
            color="green",
            linestyle="dashed",
            label=f"Smooth Max pF Time: {s_max_pf_time}us",
        )
        ax[0].axvline(
            x=s_photon_reset_time,
            color="red",
            linestyle="dashed",
            label=f"Smooth Reset Time: {s_photon_reset_time}us",
        )
        ax[0].set_xlabel("Time (us)")
        ax[0].set_ylabel("Amp (A.U.)")
        ax[0].set_title("Pulse Waveforms")
        ax[0].legend()

        ax[1].plot(ts_sim, s_pF, label=f"Smooth pF max: {jnp.round(s_max_pf, 3)}")
        ax[1].plot(ts_sim, g_pF, label=f"Gaussian Square pF: {jnp.round(g_max_pf, 3)}")
        ax[1].axvline(
            x=s_max_pf_time,
            color="green",
            linestyle="dashed",
            label=f"Smooth Max pF Time: {s_max_pf_time}us",
        )
        ax[1].axvline(
            x=g_max_pf_time,
            color="purple",
            linestyle="dashed",
            label=f"Gaussian Max pF Time: {g_max_pf_time}us",
        )
        ax[1].axvline(
            x=s_photon_reset_time,
            color="red",
            linestyle="dashed",
            label=f"Smooth Reset Time: {s_photon_reset_time}us",
        )
        ax[1].set_xlabel("Time (us)")
        ax[1].set_ylabel("Negative Log Error (pF)")
        ax[1].set_title("pF vs Time")
        ax[1].legend()

        # Find first time 0.1 photons are reached right before reset
        s_photon_val_index = jnp.where(
            jnp.flip(s_higher_photons) > time_below_photon_val, size=1
        )[0][0]
        s_photon_val_time = jnp.round(ts_sim[-s_photon_val_index], 3)

        ax[2].plot(ts_sim, s_higher_photons, label="Smooth Photons")
        ax[2].plot(ts_sim, g_higher_photons, label="Gaussian Square Photons")
        ax[2].axvline(
            x=s_max_pf_time,
            color="green",
            linestyle="dashed",
            label=f"Smooth Max pF Time: {s_max_pf_time}us",
        )
        ax[2].axvline(
            x=g_max_pf_time,
            color="purple",
            linestyle="dashed",
            label=f"Gaussian Max pF Time: {g_max_pf_time}us",
        )
        ax[2].axvline(
            x=s_photon_val_time,
            color="orange",
            linestyle="dashed",
            label=f"Smooth Time to {time_below_photon_val} Photons: {s_photon_val_time}us",
        )
        ax[2].axvline(
            x=s_photon_reset_time,
            color="red",
            linestyle="dashed",
            label=f"Smooth Reset Time: {s_photon_reset_time}us",
        )
        ax[2].set_xlabel("Time (us)")
        ax[2].set_ylabel("Photons")
        ax[2].set_title("Photons vs Time")
        if photon_log_scale:
            ax[2].set_yscale("log")
        ax[2].set_ylim(bottom=1e-3)
        ax[2].legend()

        plt.figure(figsize=(8.0, 5.0))

        s_max_pf_index = jnp.argmin(jnp.abs(ts_sim - s_max_pf_time))
        g_max_pf_index = jnp.argmin(jnp.abs(ts_sim - g_max_pf_time))
        s_pulse_end_index = jnp.argmin(jnp.abs(ts_sim - s_pulse_end_time))

        plt.plot(smooth_g.real, smooth_g.imag, label="Smooth G")
        plt.plot(smooth_e.real, smooth_e.imag, label="Smooth E")
        plt.plot(gaussian_g.real, gaussian_g.imag, label="Gaussian G")
        plt.plot(gaussian_e.real, gaussian_e.imag, label="Gaussian E")
        plt.scatter(
            smooth_g[s_max_pf_index].real,
            smooth_g[s_max_pf_index].imag,
            label=f"Smooth G Max Pf time: {s_max_pf_time}us",
        )
        plt.scatter(
            smooth_e[s_max_pf_index].real,
            smooth_e[s_max_pf_index].imag,
            label=f"Smooth E Max Pf time: {s_max_pf_time}us",
        )
        plt.scatter(
            smooth_g[s_pulse_end_index].real,
            smooth_g[s_pulse_end_index].imag,
            label=f"Smooth G End time: {s_pulse_end_time}us",
        )
        plt.scatter(
            smooth_e[s_pulse_end_index].real,
            smooth_e[s_pulse_end_index].imag,
            label=f"Smooth E End time: {s_pulse_end_time}us",
        )
        plt.scatter(
            gaussian_g[g_max_pf_index].real,
            gaussian_g[g_max_pf_index].imag,
            label=f"Gaussian G Max Pf time: {g_max_pf_time}us",
        )
        plt.scatter(
            gaussian_e[g_max_pf_index].real,
            gaussian_e[g_max_pf_index].imag,
            label=f"Gaussian E Max Pf time: {g_max_pf_time}us",
        )
        plt.xlabel("I Quadrature (A.U.)")
        plt.ylabel("Q Quadrature (A.U.)")
        plt.title("IQ Phase Space Trajectories")
        plt.legend(bbox_to_anchor=(1.0, 1.0))

        plt.show()

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParamsDetuning
    ) -> Tuple[chex.Array, EnvStateDetuning]:
        """IMPORTANT Reset Environment, in this case nothing needs to be done
        so default obs and info are returned"""
        # self.precompile()
        state = EnvStateDetuning(
            reward=0.,
            pulse_reset_transmon=0.,
            transmon_reset_reward=0.,
            mean_smooth_waveform_difference=0.,
            smoothness_reward=0.,
            pulse_reset_val=0.,
            amp_reward=0.,
            steps=0.,
            max_steps_pen=0.,
            mean_deviation=0.,
            deviation_reward=0.,
            action=jnp.zeros(2 * self.len_ts_action),
            timestep=0,
        )
        return self.get_obs(params), state

    def get_obs(self, params: Optional[EnvParamsDetuning] = EnvParamsDetuning) -> chex.Array:
        """IMPORTANT Function to get observation at a given state, as this is a single-step
        episode environment, the observation can be left fixed"""
        return jnp.zeros((1,), dtype=self.float_dtype)

    def get_info(self, env_state: EnvStateDetuning) -> dict:
        """IMPORTANT Function to get info for a given input state"""

        return {
            "reward": env_state.reward,
            # "pulse reset transmon": env_state.pulse_reset_transmon,
            #we want to return the fidelity not the pulse reset depopulation
            "fid": 1-env_state.pulse_reset_transmon/2,
            "transmon reset reward": env_state.transmon_reset_reward,
            "mean smooth waveform difference": env_state.mean_smooth_waveform_difference,
            "smoothness reward": env_state.smoothness_reward,
            "pulse reset val": env_state.pulse_reset_val,
            "amp reward": env_state.amp_reward,
            "action": env_state.action,
            "steps": env_state.steps,
            "max steps pen": env_state.max_steps_pen,
            "mean deviation": env_state.mean_deviation,
            "deviation reward": env_state.deviation_reward,
            "timestep": env_state.timestep,
        }

    @property
    def name(self) -> str:
        """IMPORTANT name of environment"""
        return "TransmonResetAmpDriveEnv"

    @property
    def num_actions(self, params: Optional[EnvParamsDetuning] = EnvParamsDetuning) -> int:
        """IMPORTANT number of actions"""
        return 2 * params.num_actions

    def action_space(self, params: Optional[EnvParamsDetuning] = None) -> spaces.Box:
        """IMPORTANT action space shape"""
        if params is None:
            params = self.default_params

        return spaces.Box(
            low=params.min_action,
            high=params.max_action,
            shape=(2 * params.num_actions,),
            dtype=self.float_dtype,
        )

    def observation_space(self, params: Optional[EnvParamsDetuning] = None) -> spaces.Box:
        """IMPORTANT observation space shape"""
        return spaces.Box(-1.0, 1.0, shape=(1,), dtype=self.float_dtype)