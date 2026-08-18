"""
All utils functions for audio_locate_isolate
"""
from copy import deepcopy

import numpy as np
from scipy.fft import irfft
from scipy.fft import get_workers
from scipy.fft import rfft
from scipy.fft import rfftfreq

from environement import Environment
from environement import Microphone
from environement import Source
from environement import Sound

float_array = np.ndarray # Type for float array


def normalize_array(arr: float_array, a: float, b: float):
    """
    Normalize an array : from [min(arr), max(arr)] to [a, b].
    :param arr: An array to normalize.
    :param a: Minimum value.
    :param b: Maximum value.
    :return: Normalized array.
    """
    try:
        arr_min = np.min(arr)
        arr_max = np.max(arr)
    except Exception as e:
        print(e)
        print(arr)
        quit()
    # Avoid dividing by 0
    if arr_min == arr_max:
        return np.full_like(arr, a)

    normalized_arr = (arr - arr_min) / (arr_max - arr_min)
    scaled_arr = a + (normalized_arr * (b - a))

    return scaled_arr


def spectrogram(x: np.array, fs: float, length: int, workers: int = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    This function makes the real spectrogram of a signal : it splits the signal into chunks of length 'length',
    and then realizes the real fft on each chunk and returns all real fft.
    Note : This function adds a window of Hamming to avoid border problems.
    :param x: The signal to transform into spectrogram.
    :param fs: The sample rate of the signal.
    :param length: The length of chunks in the spectrogram.
    :param workers: Number of CPU cores to use to calculate spectrogram.
    :return: The spectrogram of the signal.
    """
    # Add 0 at the end to fit the size
    x = deepcopy(x)
    if len(x) % length != 0:
        x = np.concatenate((x, np.zeros((length - len(x) % length,))))

    # hamming_window = np.hamming(length)

    parts = np.array(
        [x[i * length:(i + 1) * length]  for i in range(len(x) // length)]
    )

    y = rfft(parts, n=length, workers=get_workers() if workers is None else workers).T
    t = np.array([length * i / fs for i in range(len(x) // length)])
    f = rfftfreq(length, 1 / fs)

    return t, f, y


def inverse_spectrogram(y: np.ndarray, length: int, workers: int = None) -> np.ndarray:
    """
    This function if the reverse of the 'spectrogram' function: it transforms a spectrogram into a signal.
    :param y: The spectrogram.
    :param length: The length of chunks used.
    :param workers: Number of workers to use to calculate spectrogram.
    :return: The original signal.
    """
    # hamming_window = np.hamming(length)

    r: np.ndarray = irfft(y.T, n=length, workers=get_workers() if workers is None else workers)
    x = np.array([l for l in r]).flatten()

    return x


def absolute_error(signal: float_array, original: float_array, raw: bool = False) -> float_array | float:
    e = np.abs(normalize_array(signal, -1, 1) - normalize_array(original, -1, 1))
    if raw:
        return e

    else:
        return e / len(signal)


def relative_error(signal: float_array, original: float_array, raw: bool = False) -> float_array | float:
    e = np.abs((signal - original) / (np.abs(original) + 1))

    if raw:
        return e

    else:
        return e / len(signal)

def env_test_isolation(sound_list: list[Sound], length: float = 5.0, rate = 44100) :
    m1 = Microphone("m1", 0, 10, 0)
    m2 = Microphone("m2", 0, 0, 0)
    m3 = Microphone("m3", 5, 8.66, 0)

    s1 = Source(sound_list[0], "s1", 9, 5, 0)
    s2 = Source(sound_list[1], "s2", 3, 3, 0)
    s3 = Source(sound_list[2], "s3", 6, 2, 0)

    env = Environment([s1, s2, s3], [m1, m2, m3], rate=rate, microphone_tensors=[])

    start = 50.0
    end = start + length

    _, (r1, r2, r3) = env.listen_from_microphone(["m1", "m2", "m3"], start, end, start - 1,
                                                 attenuate=True,
                                                 compress=True,
                                                 noise=0.0025
                                                 )
    _, (real1,) = env.listen_from_microphone((9, 5, 0), start, end, start - 1,
                                             specific_sound=["s1"],
                                             attenuate=False)
    _, (real2,) = env.listen_from_microphone((3, 3, 0), start, end, start - 1,
                                             specific_sound=["s2"],
                                             attenuate=False)
    _, (real3,) = env.listen_from_microphone((6, 2, 0), start, end, start - 1,
                                             specific_sound=["s3"],
                                             attenuate=False)

    return (r1, r2, r3), (real1, real2, real3), [m1, m2, m3]
