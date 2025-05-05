import jax.numpy as jnp
from jax.scipy.integrate import trapezoid
import jax
import matplotlib.pyplot as plt
import chex

def default_kernel(window_length: int = 8) -> chex.Array:
    return jnp.ones(window_length) / window_length

def default_small_window(window_length: int = 8) -> chex.Array:
    return jnp.linspace(-0.5 * (window_length - 1), 0.5 * (window_length - 1),
                        window_length)

def gen_gauss_kernel(window_length: int, gauss_mean=0.0, gauss_std=1):
    """generate gaussian kernel
    input args:
    window_length: length of the kernel
    gauss_mean: mean of the gaussian
    gauss_std: standard deviation of the gaussian"""

    small_window = default_small_window(window_length)
    gauss_kernel = (1 / (jnp.sqrt(2 * jnp.pi) * gauss_std) *
                    jnp.exp(-((small_window - gauss_mean)**2) /
                            (2 * gauss_std**2)))
    gauss_kernel = gauss_kernel / jnp.sum(gauss_kernel)

    return gauss_kernel

class SignalToolClass:
    def __init__(self, gauss_kernel_amp, gauss_kernel_freq, t_total, waveform):
        """input args:
        gauss_kernel_amp: amplitude kernel
        gauss_kernel_freq: frequency kernel
        t_total: total time of the pulse
        waveform: waveform of the pulse"""
        self.gauss_kernel_amp = gauss_kernel_amp
        self.gauss_kernel_freq = gauss_kernel_freq
        self._t_total = t_total
        self.exp_waveform = waveform

    def drive_smoother(self, res_drive: jnp.ndarray, amp_kernel: bool = True):
        """input args:
        res_drive: signal as array
        amp_kernel: whether to use the amplitude kernel or the frequency kernel"""
        gauss_kernel = self.gauss_kernel_amp if amp_kernel else self.gauss_kernel_freq
        return jnp.convolve(res_drive, gauss_kernel, mode="same")

    def calculate_smoothness(self,
                             action,
                             method: str = "second_derivative",
                             _f_sampling: int = 50,
                             _f_cutoff: float = 3 * 2 * jnp.pi):
        """calculate smoothness of input signal
        input_args:
        action: full action description of signal"""
        ts = jnp.linspace(0, self._t_total, len(action))
        dx = 1.0
        if method == "second_derivative":
            first_deriv = jnp.diff(action, axis=-1) / dx
            second_deriv = jnp.diff(first_deriv, axis=-1) / dx
            smoothness = trapezoid(y=second_deriv**2, x=ts[2:], axis=-1)
        elif method == "lowpass_filter":
            signal_low_pass = self.second_order_butterworth(
                action,
                f_sampling=_f_sampling,
                f_cutoff=_f_cutoff,
                method="forward_backward")
            smoothness = jnp.sum((action - signal_low_pass)**2)
        else:
            raise NotImplementedError
        return smoothness

    def get_baseline_smoothness(self,
                                action,
                                _amp_kernel: bool = True,
                                method: str = "second_derivative",
                                _f_sampling: int = 50,
                                _f_cutoff: float = 3 * 2 * jnp.pi):
        """output the baseline smoothness of the pulse
        input_args:
        _amp_kernel: whether to use the amplitude kernel or the frequency kernel"""
        signal = action
        smoothed_signal = self.drive_smoother(signal, _amp_kernel)
        if method == "second_derivative":
            smoothness = self.calculate_smoothness(smoothed_signal,
                                                   method="second_derivative")
        elif method == "lowpass_filter":
            smoothness = self.calculate_smoothness(smoothed_signal,
                                                   method="lowpass_filter",
                                                   _f_sampling=_f_sampling,
                                                   _f_cutoff=_f_cutoff)
        else:
            raise NotImplementedError
        return smoothness

    def area_of_pulse(self, pulse_values: jnp.ndarray):
        """Calculates the area under the pulse with the trapezoidal rule
        input args:
        pulse_values: pulse values as an array"""
        t_values = jnp.linspace(0, self._t_total, len(pulse_values))
        # Compute the area using the trapezoidal rule
        area = trapezoid(pulse_values, t_values)
        return area

    def second_order_butterworth(
            self,
            signal: jax.Array,
            f_sampling: int = 50,
            f_cutoff: int = 10,
            method: str = "forward_backward") -> jax.Array:
        "https://stackoverflow.com/questions/20924868/calculate-coefficients-of-2nd-order-butterworth-low-pass-filter"
        if method == "forward_backward":
            signal = self.second_order_butterworth(signal, f_sampling,
                                                   f_cutoff, "forward")
            return self.second_order_butterworth(signal, f_sampling, f_cutoff,
                                                 "backward")
        elif method == "forward":
            pass
        elif method == "backward":
            signal = jnp.flip(signal, axis=0)
        else:
            raise NotImplementedError

        ff = f_cutoff / f_sampling
        ita = 1.0 / jnp.tan(jnp.pi * ff)
        q = jnp.sqrt(2.0)
        b0 = 1.0 / (1.0 + q * ita + ita**2)
        b1 = 2 * b0
        b2 = b0
        a1 = 2.0 * (ita**2 - 1.0) * b0
        a2 = -(1.0 - q * ita + ita**2) * b0

        def f(carry, x_i):
            x_im1, x_im2, y_im1, y_im2 = carry
            y_i = b0 * x_i + b1 * x_im1 + b2 * x_im2 + a1 * y_im1 + a2 * y_im2
            return (x_i, x_im1, y_i, y_im1), y_i

        init = (signal[1], signal[0]) * 2
        signal = jax.lax.scan(f, init, signal[2:])[1]
        signal = jnp.concatenate((signal[0:1], ) * 2 + (signal, ))

        if method == "backward":
            signal = jnp.flip(signal, axis=0)

        return signal


if __name__ == "__main__":
    # Create synthetic test data for the filter
    t_total = 1.0  # total time duration
    sampling_rate = 50  # samples per second
    f_sampling = sampling_rate
    f_cutoff = 8  # cutoff frequency for the Butterworth filter
    t = jnp.linspace(0, t_total, int(t_total * sampling_rate), endpoint=False)

    # Generate a noisy sinusoidal signal
    signal_freq = 5  # frequency of the signal in Hz
    noise_amplitude = 0.5  # noise amplitude
    clean_signal = jnp.sin(2 * jnp.pi * signal_freq * t)
    noise = noise_amplitude * jax.random.normal(jax.random.PRNGKey(0),
                                                shape=t.shape)
    noisy_signal = clean_signal + noise

    # Gaussian kernels for amplitude and frequency smoothing
    gauss_kernel_amp = jnp.exp(-0.5 * (jnp.linspace(-2, 2, 25)**2))
    gauss_kernel_freq = jnp.exp(-0.5 * (jnp.linspace(-2, 2, 25)**2))

    # Initialize SignalToolClass instance
    signal_tool = SignalToolClass(gauss_kernel_amp=gauss_kernel_amp,
                                  gauss_kernel_freq=gauss_kernel_freq,
                                  t_total=t_total,
                                  waveform=noisy_signal)

    # Apply the second-order Butterworth filter
    filtered_signal = signal_tool.second_order_butterworth(
        noisy_signal,
        f_sampling=f_sampling,
        f_cutoff=f_cutoff,
        method="forward_backward")

    # Print results for comparison
    print("Original Noisy Signal:", noisy_signal)
    print("Filtered Signal:", filtered_signal)

    plt.figure(figsize=(10, 6))
    plt.plot(t, noisy_signal, label="Noisy Signal", alpha=0.7)
    plt.plot(t, filtered_signal, label="Filtered Signal", linewidth=2)
    plt.plot(t, clean_signal, label="Original Clean Signal", linestyle='--')
    plt.xlabel("Time [s]")
    plt.ylabel("Amplitude")
    plt.title("Second-Order Butterworth Filter Test")
    plt.legend()
    plt.grid(True)
    plt.show()
