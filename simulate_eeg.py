# simulate_eeg.py

"""
EEG Simulation Defaults
=======================

This section defines the global default parameters used by the EEG simulation
utilities in this file. It specifies the sampling rate, epoch time window,
number of classes/trials, noise scaling, and a standard 32-channel EEG montage.

The montage is defined via CH_NAMES (10–20 style naming), and two functional
channel groups are automatically derived:

- posterior_idxs: parietal / occipital / centro-parietal channels
  (channels whose names start with 'P', 'O', or 'CP'), typically used as
  "signal-bearing" sensors in ERP/oscillation simulations.

- frontal_idxs: frontal channels including Fp* and F* (channels whose names
  start with 'F'), often used as a control region or as an alternative target
  region in connectivity/synchronization simulations.

These defaults are intended to provide a consistent, reproducible, and
physiologically plausible baseline configuration for generating synthetic
EEG datasets (X, y, times) for decoding and benchmarking experiments.

Author: Runhao Lu

"""

import numpy as np


# ===================== Defaults (edit here once) =====================
SFREQ = 100          
TMIN, TMAX = 0.0, 0.8

N_CLASSES = 2
N_TRIALS_PER_CLASS = 40
N_TOTAL_TRIALS = N_TRIALS_PER_CLASS * N_CLASSES
N_CHANNELS = 32
noise_scale = 1.0
# ========================= Standard 32-Channel System =========================
CH_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6',
    'T7', 'C3', 'Cz', 'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10',
    'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10'
]
N_CHANNELS = len(CH_NAMES)

# Pre-compute region indices
# 1. Posterior: Parietal (P*), Occipital (O*), Centro-parietal (CP*)
posterior_idxs = [
    i for i, name in enumerate(CH_NAMES)
    if name.startswith(('P', 'O', 'CP'))
]

# 2. Frontal: Frontal (F*), including frontal pole (Fp*)
frontal_idxs = [
    i for i, name in enumerate(CH_NAMES)
    if name.startswith('F')
]
print(f"Posterior Channels ({len(posterior_idxs)}): {[CH_NAMES[i] for i in posterior_idxs]}")


# ===================== Helpers =====================
def normalize_signal(x):
    """Normalize a 1D signal to mean=0, std=1 (with tiny epsilon guard)."""
    x = np.asarray(x)
    s = np.std(x)
    if s == 0:
        return x - np.mean(x)
    return (x - np.mean(x)) / s


def pink_noise(n_times, rng, sfreq=SFREQ):
    """
    Generate 1/f noise normalized to match white noise energy.

    Notes
    -----
    Uses frequency-domain shaping. DC is handled to avoid division issues.
    """
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    spectrum = rng.randn(len(freqs)) + 1j * rng.randn(len(freqs))

    scaling = np.sqrt(freqs)
    scaling[0] = 1.0
    spectrum /= scaling

    raw = np.fft.irfft(spectrum, n_times)
    return normalize_signal(raw)


def generate_1f_noise_physiological(n_times, slope, intercept, rng, pivot_freq=10.0, sfreq=SFREQ):
    """
    Physiological 1/f simulation with spectral rotation around a pivot frequency.

    slope: exponent (larger -> steeper 1/f)
    intercept: log10 gain offset applied after filtering
    """
    sig = rng.randn(n_times)
    freqs = np.fft.rfftfreq(n_times, d=1.0 / sfreq)
    fft_val = np.fft.rfft(sig)

    with np.errstate(divide="ignore", invalid="ignore"):
        f_norm = freqs / pivot_freq
        f_norm[0] = 1.0
        weights = 1.0 / (f_norm ** (slope / 2.0))

    weights[0] = 0.0  # remove DC
    sig_filtered = np.fft.irfft(fft_val * weights, n=n_times)

    gain = 10 ** (intercept)
    sig_filtered *= gain

    sig_filtered *= 2.0  # global scaling (EEG magnitude)
    return sig_filtered


def get_temporal_mask(times, t_start, t_end, transition=0.05, sfreq=SFREQ):
    """
    Create a smooth Tukey-like temporal mask between [t_start, t_end].

    transition: seconds used to smooth edges (approx)
    """
    times = np.asarray(times)
    mask = np.zeros_like(times, dtype=float)

    idx_start = np.searchsorted(times, t_start)
    idx_end = np.searchsorted(times, t_end)
    mask[idx_start:idx_end] = 1.0

    window_size = int(transition * sfreq)
    if window_size > 0:
        smooth_kernel = np.hamming(window_size * 2)
        smooth_kernel /= smooth_kernel.sum()
        mask = np.convolve(mask, smooth_kernel, mode="same")

    return np.clip(mask, 0.0, 1.0)


def generate_burst(times, center_time, duration, freq, phase):
    """Oscillatory burst with Gaussian envelope."""
    times = np.asarray(times)
    envelope = np.exp(-(times - center_time) ** 2 / (2 * (duration / 3) ** 2))
    oscillation = np.cos(2 * np.pi * freq * times + phase)
    return envelope * oscillation


# ===================== Main simulator =====================
def simulate_data(
    sub_id,
    mode="erp",
    target_freq=10.0,
    noise_scale_=noise_scale,
    sfreq=SFREQ,
    tmin=TMIN,
    tmax=TMAX,
    n_trials=N_TOTAL_TRIALS,
    n_channels=N_CHANNELS,
    n_classes=N_CLASSES,
    frontal=frontal_idxs,
    posterior=posterior_idxs,
):
    """
    Simulate EEG-like data: X, y, times.

    Parameters
    ----------
    sub_id : int
        Subject seed (controls RNG).
    mode : {"erp", "induced", "ispc", "slope", "intercept"}
    target_freq : float
        Carrier / pivot frequency (Hz).
    noise_scale_ : float
        Global noise scaling.
    sfreq : float
        Sampling frequency.
    tmin, tmax : float
        Epoch time range (seconds).
    n_trials : int
        Total trials/epochs.
    n_channels : int
        Number of channels.
    n_classes : int
        Number of classes.
    frontal, posterior : list[int]
        Channel indices for signal injection.

    Returns
    -------
    X : ndarray, shape (n_trials, n_channels, n_times)
    y : ndarray, shape (n_trials,)
    times : ndarray, shape (n_times,)
    """
    times = np.linspace(tmin, tmax, int((tmax - tmin) * sfreq) + 1)
    n_times = len(times)

    X = np.zeros((n_trials, n_channels, n_times), dtype=float)
    y = np.zeros(n_trials, dtype=int)

    rng = np.random.RandomState(sub_id+42)

    # Aperiodic modulation window
    MOD_START, MOD_END = 0.2, 0.6
    mod_mask = get_temporal_mask(times, MOD_START, MOD_END, transition=0.05, sfreq=sfreq)

    for i in range(n_trials):
        label = i % n_classes
        y[i] = label

        # ==================== A: Periodic / ERP Modes ====================
        if mode in ["erp", "induced", "ispc"]:
            class_amplitudes = np.linspace(1.0, 1.5, n_classes)

            white = rng.randn(n_channels, n_times)
            pink = np.array([pink_noise(n_times, rng, sfreq=sfreq) for _ in range(n_channels)])
            X[i] = noise_scale_ * (0.4 * white + 0.6 * pink)

            t_jitter = rng.uniform(-0.025, 0.025)
            f_jitter = rng.uniform(-0.5, 0.5)
            center_t = 0.4 + t_jitter
            curr_f = target_freq + f_jitter

            if mode == "ispc":
                target_kappas = [1.5, 0.1]  # class 0 high sync, class 1 low sync
                base_phase = rng.uniform(0, 2 * np.pi)
                phase_diff = rng.vonmises(0, target_kappas[label])

                SYNC_CENTER, SYNC_DUR = center_t, 0.4
                sig_F = generate_burst(times, SYNC_CENTER, SYNC_DUR, curr_f, base_phase)
                sig_P = generate_burst(times, SYNC_CENTER, SYNC_DUR, curr_f, base_phase + phase_diff)

                signal_gain = 1.2
                for ch in frontal:
                    X[i, ch, :] += sig_F * signal_gain * rng.uniform(0.8, 1.2)
                for ch in posterior:
                    X[i, ch, :] += sig_P * signal_gain * rng.uniform(0.8, 1.2)

            elif mode == "induced":
                rand_phase = rng.uniform(0, 2 * np.pi)
                sig = generate_burst(times, center_t, 0.4, curr_f, phase=rand_phase)
                sig *= class_amplitudes[label]
                for ch in posterior:
                    X[i, ch, :] += sig * rng.uniform(0.8, 1.2)

            elif mode == "erp":
                sigma1, amp1 = 0.08, 0.6
                bump1 = amp1 * np.exp(-(times - center_t) ** 2 / (2 * sigma1**2))
                sig = bump1 * class_amplitudes[label]
                for ch in posterior:
                    X[i, ch, :] += sig * rng.uniform(0.8, 1.2)

        # ==================== B: Aperiodic Modes ====================
        elif mode in ["slope", "intercept"]:
            if mode == "slope":
                class_params = [1.8, 1.2]  # exponent difference
                gain_params = [0.0, 0.0]
            else:  # intercept
                class_params = [1.2, 1.2]
                gain_params = [0.0, 0.2]  # gain difference (log10)

            target_exp = class_params[label]
            target_gain = gain_params[label]

            for ch in range(n_channels):
                BASE_EXP = 1.2
                BASE_GAIN = 0.0

                noise_base = generate_1f_noise_physiological(
                    n_times, BASE_EXP, BASE_GAIN, rng, pivot_freq=target_freq, sfreq=sfreq
                )
                noise_target = generate_1f_noise_physiological(
                    n_times, target_exp, target_gain, rng, pivot_freq=target_freq, sfreq=sfreq
                )

                if ch in posterior:
                    sig_combined = (1 - mod_mask) * noise_base + mod_mask * noise_target
                else:
                    sig_combined = noise_base

                sensor_noise = rng.randn(n_times) * 0.1
                X[i, ch, :] = sig_combined + sensor_noise
        else:
            raise ValueError(f"Unknown mode={mode}")

    return X, y, times
