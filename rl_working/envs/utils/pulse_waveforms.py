import jax
import jax.numpy as jnp


def gaussian(ts, amp, mean, sigma, t0, t1):
    gauss_waveform = amp * jnp.exp(-((ts - mean)**2) / (2.0 * sigma**2))
    gauss_waveform *= jnp.heaviside(t1 - ts, 1.0)
    gauss_waveform *= jnp.heaviside(ts - t0, 0.0)
    return gauss_waveform


def gauss_square(ts, amp, duration, width, sigma):
    mean_rise = 0.5 * (duration - width)
    gauss_rise = amp * jnp.exp(-((ts - mean_rise)**2) / (2.0 * sigma**2))
    gauss_rise *= jnp.heaviside(mean_rise - ts, 1.0)

    mean_fall = 0.5 * (duration + width)
    gauss_fall = amp * jnp.exp(-((ts - mean_fall)**2) / (2.0 * sigma**2))
    gauss_fall *= jnp.heaviside(ts - mean_fall, 0.0)

    gauss_square = (amp * jnp.heaviside(mean_fall - ts, 1.0) *
                    jnp.heaviside(ts - mean_rise, 0.0))
    return gauss_rise + gauss_square + gauss_fall
