"""
Functions to isolate a source at a specific coordinate
"""
import itertools
from math import floor
from typing import Callable

import numpy as np
from sklearn.decomposition import FactorAnalysis
from sklearn.decomposition import FastICA
from sklearn.decomposition import IncrementalPCA
from sklearn.decomposition import MiniBatchNMF
from sklearn.decomposition import NMF
from sklearn.decomposition import PCA
from sklearn.decomposition import SparsePCA
from sklearn.decomposition import TruncatedSVD

from environement import DEFAULT_TEMPERATURE
from environement import define_sound_speed
from environement import distance_gain
from environement import euclidian_distance
from environement import Microphone

from audio_locate_isolate.utils import inverse_spectrogram
from audio_locate_isolate.utils import normalize_array
from audio_locate_isolate.utils import spectrogram

SK_MODEL_NAME = ["FastICA", "NMF", "PCA", "TruncatedSVD"]

SK_MODEL = {
    "FactorAnalysis": FactorAnalysis,
    "FastICA": FastICA,
    "IncrementalPCA": IncrementalPCA,
    "MiniBatchNMF": MiniBatchNMF,
    "NMF": NMF,
    "PCA": PCA,
    "SparsePCA": SparsePCA,
    "TruncatedSVD": TruncatedSVD
}

SK_NORM = {
    "FactorAnalysis": 0,
    "FastICA": 0,
    "IncrementalPCA": 0,
    "MiniBatchNMF": 1,
    "NMF": 1,
    "PCA": 0,
    "SparsePCA": 0,
    "TruncatedSVD": 0
}


def align_records(x: float, y: float, z: float,
                  microphones: list[Microphone],
                  records: list[np.ndarray],
                  sample_rate: int,
                  temperature: float = DEFAULT_TEMPERATURE,
                  source_radius: float = 0.1
                  ) -> list[np.ndarray]:
    """
    This function aligns many records to be synchronized at a specific coordinate and also adjust gain.
    :param x: The x coordinate where sources are going to be aligned.
    :param y: The y coordinate where sources are going to be aligned.
    :param z: The z coordinate where sources are going to be aligned.
    :param microphones: List of microphones (corresponding to the record at the same index)
    :param records: List of records from microphones.
    :param sample_rate: The sample rate of records.
    :param temperature: The temperature of the room.
    :param source_radius: Radius of sources.
    :return: Records realigned and with the gain adjusted.
    """
    assert len(records) == len(microphones), "microphones and records must be the same length"

    # Create all variables needed
    dist: list[float] = [euclidian_distance(x, y, z, m.x, m.y, m.z) for m in microphones]
    reversed_gain = list(map(lambda d: 1 / distance_gain(d, source_radius), dist))
    time_shift = list(map(lambda d: d / define_sound_speed(temperature), dist))
    list_shift = list(map(lambda t: floor(t * sample_rate), time_shift))

    # add time shift
    shifted_records = [np.concatenate(
        (records[i][s:], np.zeros((s,)))
    ) for i, s in enumerate(list_shift)
    ]

    shifted_records: list = [s * g for s, g in zip(shifted_records, reversed_gain)]

    return shifted_records


def isolate_sources_spectrogram(x: float, y: float, z: float,
                                microphones: list[Microphone],
                                records: list[np.ndarray],
                                sample_rate: int,
                                error_function: Callable[[list[np.ndarray]], np.ndarray],
                                temperature: float = DEFAULT_TEMPERATURE,
                                spectrogram_length: int = 128,
                                workers: int = 1,
                                source_radius: float = 0.1
                                ) -> np.ndarray:
    """
    This function isolates sources using the spectrogram method: It takes the spectrogram of the n records and looks to
    find différencies, if differences are found, values of both n spectrograms are not kept.
    Then the function realizes an inverse spectrogram only with kept values.
    :param x: The x coordinate of a potential source.
    :param y: The y coordinate of a potential source.
    :param z: The z coordinate of a potential source.
    :param microphones: List of microphones.
    :param records: List of records from microphones.
    :param sample_rate: The sample rate of all records.
    :param temperature: The temperature of the room during the record.
    :param spectrogram_length: The length to use for spectrogram.
    :param error_function: Function to calculate error.
    :param workers: Number of CPU cores to use to calculate the spectrogram.
    :param source_radius: Radius of sources.
    :return: The potential signal from the source at the given coordinate.
    """
    assert len(records) >= 3, "records must contain at least 3 records"
    assert len(microphones) >= 3, "microphones must be at least 3 microphones"
    assert len({len(r) for r in records}) == 1, "All record must be of the same size"

    shifted_records = align_records(x, y, z, microphones, records, sample_rate,
                                    temperature=temperature,
                                    source_radius=source_radius)

    # Calculate spectrogram
    spectrogram_list: list[np.ndarray] = [
        spectrogram(r, fs=sample_rate, length=spectrogram_length, workers=workers)[2] for r in shifted_records
    ]

    result = error_function(spectrogram_list)
    sound = inverse_spectrogram(result, spectrogram_length, workers=workers)

    return sound[:len(records[0])]


def isolate_source_sklearn(x: float, y: float, z: float,
                           microphones: list[Microphone],
                           records: list[np.ndarray],
                           sample_rate: int,
                           methode_name: str,
                           temperature: float = DEFAULT_TEMPERATURE,
                           source_radius: float = 0.1
                           ) -> np.ndarray:
    """
    This function isolates sources using Independent Component Analysis (ICA) method:
    :param x: The x coordinate of a potential source.
    :param y: The y coordinate of a potential source.
    :param z: The z coordinate of a potential source.
    :param microphones: List of microphones.
    :param records: List of records from microphones.
    :param sample_rate: The sample rate of all records.
    :param temperature: The temperature of the room during the record.
    :param source_radius: Radius of sources.
    :param methode_name: Sklearn method name to decompose signals.
    :return: The potential signal from the source at the given coordinate.
    """
    assert len(records) >= 3, "records must contain at least 3 records"
    assert len(microphones) >= 3, "microphones must be at least 3 microphones"
    assert len({len(r) for r in records}) == 1, "All record must be of the same size"

    shifted_records = align_records(x, y, z, microphones, records, sample_rate,
                                    temperature=temperature,
                                    source_radius=source_radius)

    shifted_records = [r for r in shifted_records]
    mx = np.max(np.abs(np.array(shifted_records)))
    shifted_records = [r + mx * SK_NORM[methode_name] for r in shifted_records]

    methode = SK_MODEL[methode_name]

    decomp = methode(n_components=len(shifted_records))
    res = decomp.fit_transform(np.array(shifted_records).T)

    def choose_best(x1):
        s = np.corrcoef(x1, shifted_records[0])[0][1] + np.corrcoef(x1, shifted_records[1])[0][1] + \
            np.corrcoef(x1, shifted_records[2])[0][1]

        return float(s)

    res_final = max(
        [res[:, i] - mx * SK_NORM[methode_name] for i in range(len(shifted_records))],
        key=choose_best
    )

    return res_final


def isolate_source_rnn(x: float, y: float, z: float,
                       microphones: list[Microphone],
                       records: list[np.ndarray],
                       sample_rate: int,
                       model,
                       temperature: float = DEFAULT_TEMPERATURE,
                       chunk_size: int = 4410,
                       source_radius: float = 0.1):
    """
    This function isolates sources using a recurrent neural network method.
    This function is built for more dimension.
    :param x: The x coordinate of a potential source.
    :param y: The y coordinate of a potential source.
    :param z: The z coordinate of a potential source.
    :param microphones: List of microphones.
    :param records: List of records from microphones.
    :param sample_rate: The sample rate of all records.
    :param model: The LSTM model to use for isolation.
    :param temperature: The temperature of the room during the record.
    :param chunk_size: The input length of the LSTM.
    :param source_radius: Radius of sources.
    :return: The potential signal from the source at the given coordinate.
    """
    assert len(records) >= 3, "records must contain at least 3 records"
    assert len(microphones) >= 3, "microphones must be at least 3 microphones"
    assert len({len(r) for r in records}) == 1, "All record must be of the same size"

    normalize_a, normalize_b = model.get_normalization_param

    n = len(records)
    k = 0
    n_channel_input = 2

    # for _ in range(iter):
    sounds1 = []
    sounds2 = []

    for s in itertools.combinations(list(range(0, n)), n_channel_input):
        shifted_records = align_records(x, y, z,
                                        [microphones[i] for i in s],
                                        [records[i] for i in s],
                                        sample_rate,
                                        temperature=temperature,
                                        source_radius=source_radius)
        # shifted_records = [nr for nr in normalize_array(np.array(shifted_records), normalize_a, normalize_b)]
        shifted_records = [normalize_array(sr, normalize_a, normalize_b) for sr in shifted_records]
        sounds1.append(shifted_records[0])
        sounds2.append(shifted_records[1])
        k += 1

    out, _, _ = model(sounds1, sounds2, chunk_size=chunk_size, large=True)
    out = [normalize_array(o, -1, 1) for o in out]

    result = np.zeros(records[0].shape)

    for o in out:
        result += o

    result = result / k

    return result


def isolate_source_rnn2(x: float, y: float, z: float,
                        microphones: list[Microphone],
                        records: list[np.ndarray],
                        sample_rate: int,
                        model,
                        temperature: float = DEFAULT_TEMPERATURE,
                        chunk_size: int = 4410,
                        source_radius: float = 0.1):
    """
    Isolate sources using a recurrent neural network with two inputs
    :param x: The x coordinate of a potential source.
    :param y: The y coordinate of a potential source.
    :param z: The z coordinate of a potential source.
    :param microphones: List of microphones.
    :param records: List of records from microphones.
    :param sample_rate: The sample rate of all records.
    :param model: The LSTM model to use for isolation.
    :param temperature: The temperature of the room during the record.
    :param chunk_size: The input length of the LSTM.
    :param source_radius: Radius of sources.
    :return: The potential signal from the source at the given coordinate.
    """
    assert len(records) >= 3, "records must contain at least 3 records"
    assert len(microphones) >= 3, "microphones must be at least 3 microphones"
    assert len({len(r) for r in records}) == 1, "All record must be of the same size"

    normalize_a, normalize_b = model.get_normalization_param

    shifted_records = align_records(x, y, z,
                                    microphones,
                                    records,
                                    sample_rate,
                                    temperature=temperature,
                                    source_radius=source_radius)

    new_res = np.zeros(records[0].shape)

    for i in range(1):
        sounds1 = []
        sounds2 = []
        k = 0
        for s1, s2 in itertools.combinations(shifted_records, 2):
            sounds1.append(normalize_array(s1, normalize_a, normalize_b))
            sounds2.append(normalize_array(s2, normalize_a, normalize_b))
            k += 1

        out, _, _ = model(sounds1, sounds2, chunk_size=chunk_size, large=True)
        out = [normalize_array(o, -1, 1) for o in out]

        result = np.zeros(records[0].shape)

        for o in out:
            result += o

        result = result / k
        new_res = result

    return new_res


def isolate_source_rnn_triple(x: float, y: float, z: float,
                              microphones: list[Microphone],
                              records: list[np.ndarray],
                              sample_rate: int,
                              model,
                              temperature: float = DEFAULT_TEMPERATURE,
                              chunk_size: int = 4410,
                              source_radius: float = 0.1):
    """
    Isolate sources using a recurrent neural network with 3 inputs
    :param x: The x coordinate of a potential source.
    :param y: The y coordinate of a potential source.
    :param z: The z coordinate of a potential source.
    :param microphones: List of microphones.
    :param records: List of records from microphones.
    :param sample_rate: The sample rate of all records.
    :param model: The LSTM model to use for isolation.
    :param temperature: The temperature of the room during the record.
    :param chunk_size: The input length of the LSTM.
    :param source_radius: Radius of sources.
    :return: The potential signal from the source at the given coordinate.
    """
    normalize_a, normalize_b = model.get_normalization_param
    shifted_records = align_records(x, y, z,
                                    microphones,
                                    records,
                                    sample_rate,
                                    temperature=temperature,
                                    source_radius=source_radius)

    sounds = [normalize_array(s, normalize_a, normalize_b) for s in shifted_records]

    out = model([[sounds[0]], [sounds[1]], [sounds[2]]], chunk_size=chunk_size, large=True)

    return normalize_array(out[0], -1, 1)
