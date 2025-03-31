import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from diffrax import (
    backward_hermite_coefficients,
    CubicInterpolation,
    LinearInterpolation,
)


def ou_process(key, steps, alpha, mu, sigma):
    """ Generate an Ornstein-Uhlenbeck process sample. 
    input arguments:
    key: PRNGKey
    steps: int, number of steps
    alpha: float, the rate of change between timesteps
    mu: float, the mean of the process in 2pi MHZ
    sigma: float, the sd of the process in 2pi MHZ"""
    ou_init = jnp.zeros((steps + 1, ))
    noise = jax.random.normal(key, (steps, ))

    def ou_step(t, val):
        dx = (-(val[t - 1] - mu) * alpha**2 +
              sigma * jnp.sqrt(2) * noise[t] * alpha)
        val = val.at[t].set(val[t - 1] + dx)
        return val

    return jax.lax.fori_loop(1, steps + 1, ou_step, ou_init)[1:]


def rtn(key, N, _scale, bi_sd=0):
    """
    Generate a random telegraph noise sample.
    input arguments:
    N: int, number of samples
    scale: float, the characteristic random switching time in sample units
    bi_sd: float, the standard deviation of the white noise added to the bi-levels"""
    rng, rng_norm = jax.random.split(key)
    Y = jax.random.weibull_min(rng, scale=_scale, concentration=1, shape=(N, ))
    X = jnp.round(Y)

    L = jnp.zeros(N)
    bi = -1  # set binary states
    j = 0
    index = 0
    while index < N and j < len(X):
        if X[j] > 0:
            count = int(X[j])
            for _ in range(count):
                if index < N:
                    L = L.at[index].set(bi)
                    index += 1
            bi = 1 if bi == -1 else -1
        j += 1
    if bi_sd != 0:
        L = L + bi_sd * jax.random.normal(
            rng_norm, shape=(N, ))  # add white noise to bi-levels
    return L


def sinusoidal_coupling(N, t_max, omega_t, phase_0):
    '''generate sinusoidal modulation
    N is number of samples
    t_max is total time of the moudlation
    omega_t is frequency at timescale of t_max
    phase_0 is the initial phase of the modulation'''
    gen_time = jnp.linspace(0, t_max, N)
    return_samples = jnp.array(
        [jnp.sin(omega_t * 2 * jnp.pi * t + phase_0) for t in gen_time])
    return return_samples


def modulated_rtn(_key, _N, _t_max, _omega_t, _phase_0, _scale, _sd=0):
    """Generate a modulated RTN sample.
    input arguments:
    _N: int, number of samples
    _t_max: float, the total time of the noise
    _omega_t: float, the frequency of the modulation in same units as _t
    _phase_0: float, the initial phase of the modulation
    _scale: float, the characteristic random switching time in sample units
    _sd: float, the standard deviation of the white noise added to the bi-levels"""
    rtn_samples = rtn(_key, _N, _scale, _sd)
    modulation = sinusoidal_coupling(_N, _t_max, _omega_t, _phase_0)

    for i, el in enumerate(rtn_samples):
        rtn_samples = rtn_samples.at[i].set(el * modulation[i])

    return rtn_samples


def generate_ou_noise_samples(
    rng,
    n_steps,
    noise_over_sampling_rate,
    ou_noise_params,
    t_total,
    _dtype,
    n_signals=2,
):
    # Time reference for interpolation
    t_ref = jnp.linspace(0,
                         t_total,
                         n_steps * noise_over_sampling_rate,
                         dtype=_dtype)

    # Split RNG
    rng, _key = jax.random.split(rng)
    keys = jax.random.split(_key, 4)

    # Generate OU process samples for chirp and env (pump)
    ou_samples_chirp_pump = ou_process(
        keys[0],
        n_steps * noise_over_sampling_rate,
        ou_noise_params[2],
        ou_noise_params[0] * 2 * jnp.pi,
        ou_noise_params[4] * 2 * jnp.pi,
    )
    ou_samples_chirp_pump = ou_samples_chirp_pump.astype(_dtype)
    ou_noise_chirp_pump = LinearInterpolation(ts=t_ref,
                                              ys=ou_samples_chirp_pump)

    ou_samples_env_pump = ou_process(
        keys[2],
        n_steps * noise_over_sampling_rate,
        ou_noise_params[3],
        ou_noise_params[1] * 2 * jnp.pi,
        ou_noise_params[5] * 2 * jnp.pi,
    )
    ou_samples_env_pump = ou_samples_env_pump.astype(_dtype)
    ou_noise_env_pump = LinearInterpolation(ts=t_ref, ys=ou_samples_env_pump)

    if n_signals == 2:
        # Generate OU process samples for chirp and env (stokes)
        ou_samples_chirp_stokes = ou_process(
            keys[1],
            n_steps * noise_over_sampling_rate,
            ou_noise_params[2],
            ou_noise_params[0] * 2 * jnp.pi,
            ou_noise_params[4] * 2 * jnp.pi,
        )
        ou_noise_chirp_stokes = LinearInterpolation(ts=t_ref,
                                                    ys=ou_samples_chirp_stokes)

        ou_samples_env_stokes = ou_process(
            keys[3],
            n_steps * noise_over_sampling_rate,
            ou_noise_params[3],
            ou_noise_params[1] * 2 * jnp.pi,
            ou_noise_params[5] * 2 * jnp.pi,
        )
        ou_noise_env_stokes = LinearInterpolation(ts=t_ref,
                                                  ys=ou_samples_env_stokes)

        return (ou_noise_chirp_pump, ou_noise_chirp_stokes, ou_noise_env_pump,
                ou_noise_env_stokes, ou_samples_env_pump,
                ou_samples_chirp_pump)
    else:
        # Return only pump signals
        return (ou_noise_chirp_pump, ou_noise_env_pump, ou_samples_env_pump,
                ou_samples_chirp_pump)


def generate_zero_noise_samples(
    n_steps,
    noise_over_sampling_rate,
    _dtype,
    n_signals=2,
):
    zeros_env_pump = jnp.zeros(n_steps * noise_over_sampling_rate,
                               dtype=_dtype)
    zeros_chirp_pump = jnp.zeros(n_steps * noise_over_sampling_rate,
                                 dtype=_dtype)

    if n_signals == 2:
        zeros_env_stokes = jnp.zeros(n_steps * noise_over_sampling_rate,
                                     dtype=_dtype)
        zeros_chirp_stokes = jnp.zeros(n_steps * noise_over_sampling_rate,
                                       dtype=_dtype)
        return zeros_chirp_pump, zeros_chirp_stokes, zeros_env_pump, zeros_env_stokes
    else:
        return zeros_chirp_pump, zeros_env_pump


def generate_gaussian_noise_samples(rng,
                                    n_steps,
                                    omega_p,
                                    omega_s,
                                    delta_p,
                                    delta_s,
                                    gaussian_noise_scaling_amp,
                                    gaussian_noise_scaling_phase,
                                    n_signals=2):
    # Compute scaling factors for Gaussian noise
    gaussian_noise_scaling_p = gaussian_noise_scaling_amp * omega_p
    gaussian_noise_scaling_s = gaussian_noise_scaling_amp * omega_s
    gaussian_noise_scaling_det_p = gaussian_noise_scaling_phase * delta_p
    gaussian_noise_scaling_det_s = gaussian_noise_scaling_phase * delta_s

    # Generate Gaussian noise
    rng, _key = jax.random.split(rng)
    keys_noise = jax.random.split(_key, 4)

    noise_env_p = gaussian_noise_scaling_p * jax.random.normal(keys_noise[0])
    noise_det_p = gaussian_noise_scaling_det_p * jax.random.normal(
        keys_noise[2])

    if n_signals == 2:
        noise_env_s = gaussian_noise_scaling_s * jax.random.normal(
            keys_noise[1])
        noise_det_s = gaussian_noise_scaling_det_s * jax.random.normal(
            keys_noise[3])

        # generate noise samples
        noise_samples_env_pump = jnp.array([noise_env_p] * n_steps)
        noise_samples_chirp_pump = jnp.array([noise_det_p] * n_steps)

        return noise_env_p, noise_det_p, noise_env_s, noise_det_s, noise_samples_env_pump, noise_samples_chirp_pump
    else:
        # generate noise samples
        noise_samples_env_pump = jnp.array([noise_env_p] * n_steps)
        noise_samples_chirp_pump = jnp.array([noise_det_p] * n_steps)

        return noise_env_p, noise_det_p, noise_samples_env_pump, noise_samples_chirp_pump
