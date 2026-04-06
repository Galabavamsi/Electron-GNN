import numpy as np
from scipy.signal import butter, filtfilt

def extrapolate_dipole_moment(frequencies, amplitudes, time_grid):
    """
    Extrapolates the time-dependent dipole moment infinitely out into the future
    based on the extracted Bohr frequencies and LASSO-fitted amplitudes.
    """
    dipole = np.zeros_like(time_grid)
    for freq, amp in zip(frequencies, amplitudes):
        dipole += amp * np.sin(freq * time_grid)
    return dipole

def apply_hauge_lowpass_filter(dipole_signal, dt, omega_max=4.0):
    """
    Applies the Butterworth low-pass filter specifically recommended by 
    Hauge et al. (2023) to focus on valence excitations under 4.0 a.u.
    """
    # Convert angular frequency ω_max to Hz equivalent for scipy filter
    f_max = omega_max / (2 * np.pi) 
    nyquist = 1 / (2 * dt)
    cutoff = f_max / nyquist
    b, a = butter(4, cutoff, btype='low')
    return filtfilt(b, a, dipole_signal)
