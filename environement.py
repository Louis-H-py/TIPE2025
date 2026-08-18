"""
Environment de simulation du son dans un espace.
"""
from copy import deepcopy
from math import floor
from typing import Callable

import numpy as np
from scipy.io import wavfile
from scipy.io.wavfile import write

DEFAULT_TEMPERATURE: float = 20.0
SOURCE_RADIUS: float = 0.1
COMPRESS_ALPHA: float = 0.9
NOISE_ALPHA: float = 0.00


def play_sound(sound: np.ndarray, rate: int = 44100) -> None:
    """
    This function plays sound in headphone.
    :param sound: Sound to play.
    :param rate: The rate of the music in Hz.
    :return: None
    """
    import pyaudio

    sound = np.clip(sound, -1, 1)
    p = pyaudio.PyAudio()

    stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=rate,
                    output=True)

    stream.write(sound.astype(np.float32).tobytes())
    stream.stop_stream()
    stream.close()
    p.terminate()


def save_sound(name: str, sound: np.ndarray, rate: int) -> None:
    """
    This function saves a sound as an .wav file.
    :param name: Name of the file without ".wav"
    :param sound: The sound as an array of float
    :param rate: The rate of the sound in Hz.
    :return: None
    """
    scaled = np.int16(sound / np.max(np.abs(sound)) * 32767)
    write(f'{name}.wav', rate, scaled)


def define_sound_speed(temperature: float = DEFAULT_TEMPERATURE) -> float:
    """
    This function calculates the speed of the sound in the air at a givent temperature.
    :param temperature: The temperature in °C.
    :return: The speed of sound in the air in m.s^-1
    """
    return 331.5 + 0.607 * temperature


def define_air_density(temperature: float = DEFAULT_TEMPERATURE) -> float:
    return 1.292 * (273 / (temperature + 273.15))


def distance_gain(distance: float | np.ndarray,
                  source_radius: float = SOURCE_RADIUS, array=False) -> float:
    """
    This function calculates the change of the audio gain at a specific distance of the source.

    P0 = N * p_max
    I0 = P0 ** 2 / (rho * c)
    W = I0 * 4 * pi * source_radius ** 2

    I1 = W / (4.0 * pi * distance ** 2)
    P = sqrt(I1 * (rho * c))
    N = P / p_max

    :param source_radius: Radius of the source.
    :param distance: Distance with the center of the source.
    :param array: Array mode
    :return: Gain of the new audio signal.
    """
    if array:
        res = source_radius / distance
        res[np.abs(distance) < source_radius] = 1
        return res

    else:
        if np.abs(distance) < source_radius:
            return 1
        return source_radius / distance


def euclidian_distance(x0, y0, z0, x1, y1, z1) -> float | np.ndarray:
    """
    This function calculates the euclidian distance between two points.
    """
    return ((x0 - x1) ** 2 + (y0 - y1) ** 2 + (z0 - z1) ** 2) ** 0.5


class Sound:
    """
    This class represents a sound with many properties.
    """

    def __init__(self,
                 data: np.ndarray | Callable[[float], float] | str,
                 rate: int,
                 start: float = 0.0,
                 end: float = None,
                 default_value: float = 0.0,
                 max_pressure: float = 20.0
                 ) -> None:
        """
        Init the sound class.
        :param data: Sound Data, many ways to create a sound:
                     - np.ndarray: Sound as a sequence of numerical values between -1 and 1.
                     - str: Path where a .wav file is stored
                     - Callable: Function for the sound time -> numerical values between -1 and 1.
        :param rate: The rate of the sound.
        :param start: Time in second when the sound starts.
        :param end:  Time in seconds when the sound stops.
        :param default_value: Default values where the sound is undefined (time outside [start; end]).
        """

        # Get the type of data given
        if isinstance(data, np.ndarray) or isinstance(data, list):
            self.rate: int = rate

            self.data = np.array(data)
            self.data_type = "array"
            self.end = len(data) + floor(rate * start)

        elif isinstance(data, Callable):
            self.rate: int = rate

            self.func_data = data
            self.data_type = "function"
            self.end: float = floor(end * rate) if end is not None else None

        elif isinstance(data, str):
            r, d = wavfile.read(data)
            self.rate = r
            if len(d.shape) > 1:
                self.data = d[0:, 1] / 32768.0
                self.end = len(self.data) + floor(rate * start)
            else:
                self.data = d / 32768.0
                self.end = len(self.data) + floor(rate * start)

            self.data_type = "array"

        self.start: int = floor(self.rate * start)
        self.default_value: float = default_value

        self.max_pressure: float = max_pressure

    def __getitem__(self, item: int | slice | np.ndarray) -> float | np.ndarray:
        """
        Get the numerical value of the sound at a given index (in 1/rate seconde).
        :param item: Time (in 1/rate seconde), many ways to get value at a specific time:
                     - int: Gives the value of sound at one specific index
                     - slice: Gives the value of sound at many indexes between start and end, and with step.
                     - np.ndarray: Give the value of sound at many specific indexes.
        :return: Numerical value(s) of sound at given index.
        """
        # Extract properties of the slice (if item is a slice)
        if isinstance(item, slice):
            start = item.start
            stop = item.stop
            step = item.step if not self.data_type == "function" else 1

        # Case of an array
        elif isinstance(item, np.ndarray):
            if self.data_type == "array":
                return self.data[item]
            else:
                return self.func_data(item / self.rate)

        # Default case
        else:
            stop = item
            start = 0
            step = 1

        # When item is an int or a slice.
        if self.data_type == "array":
            # Only one value
            if start is None:
                if self.start <= stop < self.end:
                    return self.data[stop]
                else:
                    return self.default_value

            # Multiples values
            else:
                return np.concatenate((
                    np.ones((min(start, self.start),)) * self.default_value,
                    self.data[start:stop:step],
                    np.ones((max(0, stop - self.end, ))) * self.default_value
                ))

        elif self.data_type == "function":
            # Only one value
            if start is None:
                if self.start <= stop < self.end:
                    return self.func_data(stop / self.rate)
                else:
                    return self.default_value
            # Multiples values
            else:
                x = np.arange(start, stop, step)
                x = x / self.rate
                x[x < self.start / self.rate] = np.nan
                x[x >= self.end / self.rate] = np.nan

                x[x != np.nan] = self.func_data(x[x != np.nan])
                x[np.isnan(x)] = self.default_value

                return x
        else:
            raise Exception("Unknown data_type '%s'" % self.data_type)

class Source:
    """
    This class represents a sound source with many properties.
    """

    def __init__(self, sound: Sound, name: str, x: float, y: float, z: float = 0, radius: float = 0.1) -> None:
        """
        Init the sound class.
        :param sound: The sound that the source has to play.
        :param name: The name of the source.
        :param x: The x coordinate of the source.
        :param y: The y coordinate of the source.
        :param z: The z coordinate of the source.
        :param radius: Radius of the source in m.
        """
        self.name: str = name

        # En mètres
        self.x: float = x
        self.y: float = y
        self.z: float = z

        self.sound: Sound = sound

        self.radius: float = radius

    @property
    def get_rate(self) -> float:
        return self.sound.rate

    @property
    def get_coord(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def get_radius(self) -> float:
        return self.radius

    @property
    def get_recording_pressure(self) -> float:
        return self.sound.max_pressure


class Microphone:
    """
    This class represents a microphone with many properties.
    """

    def __init__(self, name: str, x: float, y: float, z: float = 0, max_pressure: float = 20.0) -> None:
        """
        Init the microphone class.
        :param name: Name of the microphone.
        :param x: The x coordinate of the microphone.
        :param y: The y coordinate of the microphone.
        :param z: The z coordinate of the microphone.
        """
        self.name: str = name

        # En mètres
        self.x: float = x
        self.y: float = y
        self.z: float = z

        self.max_pressure: float = max_pressure

    @property
    def get_coord(self) -> tuple[float, float, float] | tuple[float, float]:
        return self.x, self.y, self.z

    @property
    def get_max_pressure(self):
        return self.max_pressure


class MicrophoneTensor:
    """
    This class represents a tensor/array of microphones.
    It is used to observe sound across a lots of microphones (for example, to see the sound propagation in a region)
    """

    def __init__(self, name: str, ax_x: np.ndarray, ax_y: np.ndarray, ax_z: np.ndarray,
                 max_pressure: float = 20.0) -> None:
        """
        Init the microphone tensor class.
        :param name: Name of the Tensor.
        :param ax_x: Many x coordinates for microphones on the tensor.
        :param ax_y: Many y coordinates for microphones on the tensor.
        :param ax_z: Many z coordinates for microphones on the tensor.
        """
        self.name: str = name

        self.ax_x: np.ndarray = ax_x
        self.ax_y: np.ndarray = ax_y
        self.ax_z: np.ndarray = ax_z

        self.mesh_x, self.mesh_y, self.mesh_z = np.meshgrid(self.ax_x, self.ax_y, self.ax_z)

        self.shape: tuple = (len(ax_x), len(ax_y), len(ax_z))

        self.max_pressure: float = max_pressure

    @property
    def get_max_pressure(self):
        return self.max_pressure


class Environment:
    """
    This class creates an audio environment to simulate sound propagation.
    It can handle many sources, microphones and microphones tensors to create a realistic simulation of sound.
    """

    def __init__(self,
                 sources: list[Source],
                 microphones: list[Microphone],
                 rate: int | None,
                 microphone_tensors: list[MicrophoneTensor] = None,
                 temperature: float = 20.0,
                 ) -> None:
        """
        Init the environment.
        :param sources: List of sources to add in the environment.
        :param microphones: List of microphones to add in the environment.
        :param rate: The rate of all sounds in the environment. (All sources must have the same rate)
        :param microphone_tensors: List of microphone tensors.
        :param temperature: The temperature of the iar in the environment.
        """
        assert  len({s.get_rate for s in sources}) == 1, "Sources must have the same rate"
        assert "all" not in [s.name for s in sources], "Source name can't be all"

        self.sources: dict[str, Source] = {s.name: s for s in sources}
        self.microphones: dict[str, Microphone] = {m.name: m for m in microphones}
        if microphone_tensors is not None:
            self.microphone_tensors: dict[str, MicrophoneTensor] = {m.name: m for m in microphone_tensors}
        else:
            self.microphone_tensors = dict()
        self.rate: int = sources[0].get_rate if rate is None else rate
        self.step: float = 1 / self.rate

        # In Celsius
        self.temperature: float = temperature
        self.sound_speed: float = define_sound_speed(temperature)

    @property
    def get_rate(self) -> int:
        return self.rate

    def applies_time_shift(self, sound: np.ndarray, time_shift: float) -> np.ndarray:
        """
        This functions applies a timeshift to a sound.
        Sound is completed with 0.
        :param sound: Sound as array
        :param time_shift: The timeshift to add to the sound in seconds.
        :return: The shifted sound.
        """
        shift = floor(time_shift * self.rate)

        new_sound = np.roll(sound, shift)
        new_sound[:shift] = 0

        return new_sound

    def listen_specific_sound(self, name: str, from_time: float, to_time: float) -> np.ndarray:
        """
        This function is used to listen to only one specific source.
        :param name: Name of the source to listen to.
        :param from_time: Time in seconde to start to listen.
        :param to_time: Time in seconde to stop to listen.
        :return: The sound during the [from_time; stop_time] period of the 'name' source.
        """
        indice_from: int = floor(self.rate * from_time)
        indice_to: int = floor(self.rate * to_time)
        return self.sources.get(name).sound[indice_from:indice_to]

    @staticmethod
    def audio_compressor(x: float | np.ndarray, alpha: float = COMPRESS_ALPHA) -> np.ndarray | float:
        """
        This function compresses a numerical audio signal to fit betweeen -1 and 1 with the fewer losses of information as
        possible.
        :param x: Numerical audio signal.
        :param alpha: Compression level of amplitude higher than 1 (1: Very compressed, 0: Not compressed).
        :return: Compressed signal.
        """
        y = deepcopy(x)
        y[x < -1] = (1 - alpha) * np.tanh(alpha / (1 - alpha) * x[x < -1] + alpha / (1 - alpha)) - alpha
        y[x > 1] = (1 - alpha) * np.tanh(alpha / (1 - alpha) * (x[x > 1] - 2) + alpha / (1 - alpha)) + alpha
        y[np.abs(x) <= 1] = alpha * x[np.abs(x) <= 1]

        return y

    @staticmethod
    def audio_noise(signal: np.ndarray, alpha: float = NOISE_ALPHA) -> np.ndarray:
        """
        Add noise to a given signal
        :param signal: A numpy array signal.
        :param alpha: Percentage of noise.
        :return: Noisy signal.
        """
        return signal + np.random.normal(scale=alpha, size=signal.shape)

    def listen_from_microphone(self,
                               mic_names: str | list[str] | tuple | list[tuple],
                               from_time: float,
                               to_time: float,
                               record_start: float,
                               attenuate: bool = True,
                               specific_sound: list[str] = None,
                               exclude_sound: list[str] = None,
                               compress: bool = True,
                               noise: float = NOISE_ALPHA,
                               ) -> tuple[np.ndarray, list[np.ndarray]]:
        """
        Simulate the environment for a period of time
        :param mic_names: microphone(s) or coordinate(s) where the environment is recorded.
        :param from_time: Start time of recordings.
        :param to_time: Stop time of recordings.
        :param record_start: Real time to start simulation (must be lower than start_time).
        :param attenuate: If the sound must be attenuated with distance.
        :param specific_sound: To allow only specific sources.
        :param exclude_sound: To remove specific sources.
        :param compress: If the signal has to be compressed.
        :param noise: The percentage of noise.
        :return: (x, [y1, ... yn])
        """
        # Manage different kind of 'microphones' types:
        # If 'microphone' is a string
        if isinstance(mic_names, str):
            # Record every microphone.
            if mic_names == "all":
                microphone: list[Microphone] = [s for s in self.microphones.values()]
            # Record to only one microphone.
            else:
                microphone: list[Microphone] = [self.microphones[mic_names]]
        # If 'microphone' is a list
        elif isinstance(mic_names, list):
            # If 'microphone' is a list of string.
            if isinstance(mic_names[0], str):
                microphone: list[Microphone] = [self.microphones[n] for n in mic_names]

            # If 'microphone' is a list of coordinate tuple
            elif isinstance(mic_names[0], tuple):
                microphone: list[Microphone] = [Microphone("temp", *c) for c in mic_names]
            else:
                raise Exception("Unsupported type for mic_names")

        # If 'microphone' is a coordinate tuple
        elif isinstance(mic_names, tuple):
            microphone: list[Microphone] = [Microphone("temp", *mic_names)]
        else:
            raise Exception("Unsupported type for mic_names")

        # Prepare time properties
        time_interval = np.arange(record_start, to_time, self.step)
        results = []

        indice_from: int = floor(self.rate * from_time)
        indice_to: int = floor(self.rate * to_time)
        indice_record_start: int = floor(self.rate * record_start)

        # Select specific sources needed.
        if specific_sound is None:
            dict_sound = {s.name: s.sound[indice_record_start:indice_to] for s in self.sources.values()}
        else:
            dict_sound = {s.name: s.sound[indice_record_start:indice_to] for s in self.sources.values() if
                          s.name in specific_sound}
        # Exclude the necessary sources
        if exclude_sound is not None:
            for name in exclude_sound:
                del dict_sound[name]

        # For each microphone, listen to the environment.
        for mic in microphone:
            mic_sound = np.zeros((indice_to - indice_record_start,))

            x, y, z = mic.x, mic.y, mic.z

            for name, sound in dict_sound.items():
                sx, sy, sz = self.sources[name].x, self.sources[name].y, self.sources[name].z
                d = euclidian_distance(x, y, z, sx, sy, sz)
                time_shift = d / self.sound_speed

                delayed_sound = self.applies_time_shift(sound, time_shift)
                delayed_sound = self.audio_noise(delayed_sound, noise)

                g = distance_gain(d, self.sources[name].get_radius) if attenuate else 1

                mic_sound = mic_sound + delayed_sound * g

            mic_sound = self.audio_compressor(mic_sound) if compress else mic_sound

            results.append(mic_sound)

        results = [r[(indice_from - indice_record_start):] for r in results]
        time_interval = time_interval[(indice_from - indice_record_start):]

        return time_interval, results

    def listen(self, mic_names: str, from_time: float, to_time: float, attenuate: bool = True) -> None:
        """
        This function is used to listen to the environment during a given period.
        :param mic_names: The name of the microphone to use to listen to the environment.
        :param from_time: Time in seconde to start to listen.
        :param to_time: Time in seconde to stop to listen.
        :param attenuate: If the sound has to be attenuated with the distance.
        :return: None
        """
        _, sound = self.listen_from_microphone(mic_names, from_time, to_time,
                                               record_start=max((from_time - 1.0), 0),
                                               attenuate=attenuate)
        sound = sound[0]

        play_sound(sound, self.rate)

    def save(self, name: str, mic_names: str, from_time: float, to_time: float, attenuate: bool = True):
        """
        Save the environment sound to a .wav file.
        :param name: Name of the file
        :param mic_names: Microphone to listen to
        :param from_time: Time in seconde to start to listen.
        :param to_time: Time in seconde to stop to listen.
        :param attenuate: If the sound has to be attenuated with the distance.
        :return: None
        """
        _, sound = self.listen_from_microphone(mic_names, from_time, to_time, max((from_time - 1.0), 0),
                                               attenuate=attenuate)
        sound = sound[0]

        scaled = np.int16(sound / np.max(np.abs(sound)) * 32767)
        write(f'{name}.wav', self.rate, scaled)

    def render_frame_tenor(self,
                           tensor_name: str,
                           render_time: float,
                           attenuate: bool = True,
                           compress: bool = True
                           ) -> np.ndarray:
        """
        This function renders the microphone tensor at a given time to see the environment.
        :param tensor_name: The name of the tensor to render.
        :param render_time: The time when the tensor is rendered (in second)
        :param attenuate: If the sound has to be attenuated with the distance.
        :param compress: If the signal has to be compressed.
        :return: The render of the microphone tensor.
        """
        mic_tensor = self.microphone_tensors[tensor_name]

        result = np.zeros(mic_tensor.shape)
        tx, ty, tz = mic_tensor.mesh_x, mic_tensor.mesh_y, mic_tensor.mesh_z

        for source in self.sources.values():
            sx, sy, sz = source.x, source.y, source.z

            d = euclidian_distance(tx, ty, tz, sx, sy, sz)
            time_shift = d / self.sound_speed  # s
            shift = time_shift * self.rate  # sans unité

            index_tensor = np.round(render_time * self.rate) - shift  # sans unité
            index_tensor = np.maximum(index_tensor, 0)  # sans unité
            sound_tensor = source.sound[np.array(index_tensor, dtype=np.int64)]
            sound_tensor[index_tensor < 0] = 0

            dist_gain = distance_gain(d, source.get_radius, array=True)

            result = result + sound_tensor * (dist_gain if attenuate else 1)

        result = self.audio_compressor(result) if compress else result

        return result
