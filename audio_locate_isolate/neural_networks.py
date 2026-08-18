"""
Neural networks used to isolate audio signal
"""
__import__("os").environ["KERAS_BACKEND"] = "jax"
__import__("os").environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

from copy import deepcopy
from datetime import datetime
from math import floor
import os
import random
from threading import main_thread
from threading import Thread
import tkinter as tk
import time
from typing import Any
from typing import Callable
import warnings

from keras.api.callbacks import LearningRateScheduler
from keras.api.layers import Bidirectional
from keras.api.layers import Dense
from keras.api.layers import Input
from keras.api.layers import GRU
from keras.api.layers import LSTM
from keras.api.models import Model
from keras.api.models import load_model
from keras.api.optimizers import Adam
import matplotlib
import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import numpy as np

from environement import DEFAULT_TEMPERATURE
from environement import define_sound_speed
from environement import distance_gain
from environement import euclidian_distance
from environement import Environment
from environement import Microphone
from environement import Source
from environement import Sound

from audio_locate_isolate.isolate import isolate_source_rnn_triple
from audio_locate_isolate.utils import absolute_error
from audio_locate_isolate.utils import env_test_isolation
from audio_locate_isolate.utils import float_array
from audio_locate_isolate.utils import inverse_spectrogram
from audio_locate_isolate.utils import normalize_array
from audio_locate_isolate.utils import relative_error
from audio_locate_isolate.utils import spectrogram

EVALUATE: bool = False
loaded_data: dict = {}
graph_results = []
saving_thread_queue: list[tuple[str, dict | float_array]] | list = []


class IsolateModel:
    def __init__(self,
                 neural_network: Model,
                 input_number: int = 2,
                 output_number: int = 3,
                 norm_a: float = -1,
                 norm_b: float = 1,
                 use_dropout: float | None = 0.1,
                 lr: float = 0.001):
        self.opt = Adam

        self.neural_network = neural_network

        self.input_number = input_number
        self.output_number = output_number

        self.norm_a = norm_a
        self.norm_b = norm_b

        self.use_dropout = use_dropout
        self.lr = lr

    def save_model(self, path: str) -> None:
        """
        Save the model in a file '.keras' file
        :param path: Path and filename (with .keras at the end)
        :return: None
        """
        self.neural_network.save(path)

    def load_model(self, path: str) -> None:
        """
        Load a model from a '.keras' file
        :param path: Path to a '.keras' file
        :return: None
        """
        new_network = load_model(path)
        self.neural_network.set_weights(new_network.get_weights())

    @property
    def get_normalization_param(self) -> tuple[float, float]:
        """
        Returns le lower and upper bond for normalization of inputs.
        :return: Normalization bounds.
        """
        return self.norm_a, self.norm_b

    def edit_dataset(self,
                     shifted_records,
                     result,
                     only_input: bool = False) -> tuple[list[float_array], list[float_array]]:
        """
        This function edits the dataset to fit with the model.
        :param shifted_records: Shifted records given as input for the model.
        :param result: The sound that the model has to isolate.
        :param only_input: To transforme only an input
        :return: X_train and Y_train
        """
        _ = self
        return [np.array([])], [np.array([])]

    def summary(self) -> None:
        self.neural_network.summary()

    def get_weights(self):
        return self.neural_network.get_weights()

    def set_weights(self, weights) -> None:
        self.neural_network.set_weights(weights)

    def train_model(self,
                    x_train: float_array,
                    y_train: float_array,
                    batch_size=8,
                    epochs=1,
                    *args, **kwargs) -> None:
        self.neural_network.fit(x_train, y_train, *args, batch_size=batch_size, epochs=epochs, **kwargs)


class RawIsolateModel(IsolateModel):
    def __init__(self, layers: list, use_dropout: float | None = 0.1, lr: float = 0.001):
        self.opt = Adam

        optimizer = self.opt(learning_rate=lr)
        inputs = Input(shape=(None, 3,))
        x = None

        for i, layer in enumerate(layers):
            if i == 0:
                x = layer(inputs)

            else:
                x = layer(x)

        y = Dense(3)(x)

        self.neural_network = Model(inputs=inputs, outputs=y)

        self.neural_network.compile(loss='mse',
                                    optimizer=optimizer,
                                    metrics=["accuracy", "mse"],
                                    run_eagerly=False,
                                    jit_compile=True,
                                    )

        super().__init__(neural_network=self.neural_network,
                         input_number=3,
                         output_number=1,
                         norm_a=-1,
                         norm_b=1,
                         use_dropout=use_dropout,
                         lr=lr)

    def __call__(self,
                 sounds: list[list[float_array]],
                 chunk_size: int = None,
                 large: bool = False,
                 *args, **kwargs) -> list[float_array]:
        """
        Call the neural network for audio isolation
        :param sounds: List of audio signals
        :param chunk_size: Output length of the model
        :param args: *
        :param kwargs: **
        :return: Return a signal (isolate)
        """
        n = len(sounds[0][0])
        chunk_size: int = n if chunk_size is None else chunk_size
        sound_amount = len(sounds[0])
        x = []
        x_index = []

        # create chunks
        for k, s_all in enumerate(zip(*sounds)):
            for i in range(0, n, chunk_size):
                d = np.array([normalize_array(s[i:min(i + chunk_size, n)], self.norm_a, self.norm_b) for s in s_all]).T
                x.append(d)
                x_index.append(k)

        # get the output of the neural network
        if not large:
            output = self.neural_network.call(np.array(x), *args, training=False, **kwargs)
        else:
            output = self.neural_network.predict(np.array(x), verbose=False, *args, **kwargs)

        # transform chunk data into one sigle signal
        y = [np.array([]) for _ in range(sound_amount)]

        for i, line in enumerate(output):
            ind = x_index[i]
            y[ind] = np.append(y[ind], np.reshape(line[:, 0], (chunk_size,)))

        return y

    def edit_dataset(self,
                     shifted_records: list[float_array],
                     result: list[float_array],
                     only_input: bool = False) -> tuple[list[float_array], list[float_array]]:

        data_x = np.array([normalize_array(r, self.norm_a, self.norm_b) for r in shifted_records]).T
        if not only_input:
            data_y = np.reshape(
                normalize_array(np.array(result), self.norm_a, self.norm_b),
                (len(result[0]), self.output_number)
            )
        else:
            return [data_x], []

        return [data_x], [data_y]


class SpectrogramIsolateModel(IsolateModel):
    def __init__(self,
                 layers: list,
                 rate: float = 44100,
                 length: int = 128,
                 use_dropout: float | None = 0.1,
                 lr: float = 0.001):
        self.rate = rate
        self.length = length

        self.opt = Adam

        optimizer = self.opt(learning_rate=lr)
        inputs = Input(shape=(None, 3 * 1 * (length // 2 + 1),))
        x = None

        for i, layer in enumerate(layers):
            if i == 0:
                x = layer(inputs)

            else:
                x = layer(x)

        y = Dense(1 * (length // 2 + 1))(x)

        self.neural_network = Model(inputs=inputs, outputs=y)

        self.neural_network.compile(loss='mse',
                                    optimizer=optimizer,
                                    metrics=["accuracy", "mse"],
                                    run_eagerly=False,
                                    jit_compile=True,
                                    )

        super().__init__(neural_network=self.neural_network,
                         input_number=3,
                         output_number=1,
                         norm_a=-1,
                         norm_b=1,
                         use_dropout=use_dropout,
                         lr=lr)

    def edit_dataset(self,
                     shifted_records: list[float_array],
                     result: list[float_array],
                     only_input: bool = False) -> tuple:
        fft_sr = [spectrogram(sr, self.rate, self.length)[2] for sr in shifted_records]
        res = [spectrogram(r, self.rate, self.length)[2] for r in result]

        x = []
        for lines in zip(*[s.T for s in fft_sr]):
            x.append(
                np.concatenate((*(
                    # [np.abs(l) / np.max(np.abs(l)) if np.max(np.abs(l)) > 0 else np.zeros((len(l),)) for l in lines] +
                    [np.angle(l) / np.pi for l in lines]),))
            )

        if not only_input:
            y = []
            for lines in zip(*[s.T for s in res]):
                y.append(
                    np.concatenate((*(
                        # [np.abs(l) / np.max(np.abs(l)) if np.max(np.abs(l)) > 0 else np.zeros((len(l),)) for l in lines] +
                        [np.angle(l) / np.pi for l in lines]),))
                )
            return [np.array(x)], [np.array(y)]

        else:
            return [np.array(x)], []

    def __call__(self,
                 sounds: list[list[float_array]],
                 chunk_size=None,
                 large: bool = False,
                 *args,
                 **kwargs) -> list[float_array]:

        x = []
        for ls in zip(*sounds):
            inp, _ = self.edit_dataset(list(ls), [], True)
            x.append(inp[0])

        if not large:
            output = self.neural_network.call(np.array(x), *args, training=False, **kwargs)
        else:
            output = self.neural_network.predict(np.array(x), *args, verbose=False, **kwargs)

        y = []

        for out, inp in zip(output, sounds[0]):
            y_fft = []

            spectro = spectrogram(inp, self.rate, self.length)[2].T

            for line, amp in zip(out, spectro):
                amp = np.abs(amp)
                y_fft.append(
                    amp * np.exp(1j * line * np.pi)
                )

            y_fft = np.array(y_fft).T

            y.append(inverse_spectrogram(np.array(y_fft), self.length)[:len(sounds[0][0])])
        return y


def align_records(x: float, y: float, z: float,
                  microphones: list[Microphone],
                  records: list[np.ndarray],
                  sample_rate: int,
                  temperature: float = DEFAULT_TEMPERATURE,
                  source_radius: float = 0.1
                  ) -> list[np.ndarray]:
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


def func_generate_train_env(sounds_list: list[Sound],

                            mics_amount: int = 10,
                            mic_coordinate_interval: tuple[float, float] = (-25, 25),
                            min_mics_distance: float = 2.0,

                            source_amount: tuple[int, int] = (1, 5),
                            source_coordinate_interval: tuple[float, float] = (-25, 25),
                            min_sources_distance: float = 1.0,

                            plan: bool = True,
                            temperature: float = DEFAULT_TEMPERATURE) -> Callable[[], Environment]:
    """
    Generate a function that will create a new environment with certains properties.
    :param sounds_list: List of sound to use in the environment.
    :param mics_amount: The number of microphones needed
    :param mic_coordinate_interval: An [a, b] coordinate interval for microphones.
    :param min_mics_distance: Minimum distance between two microphones
    :param source_amount: An [a, b] amount of sources.
    :param source_coordinate_interval: An [a, b] coordinate interval for sources
    :param min_sources_distance: The minimum distance between two sources.
    :param temperature: The temperature of the environment
    :param plan: If the environment is a 2D or 3D environment.
    :return: A function that generates environment.
    """
    min_mic_coordinate, max_mic_coordinate = mic_coordinate_interval
    min_source_coordinate, max_source_coordinate = source_coordinate_interval
    min_source_amount, max_source_amount = source_amount

    def generator():
        """
        This function generates a new environment.
        :return: A new environment.
        """
        # Create microphones
        mics = []

        for k in range(mics_amount):
            x1, y1, z1 = np.random.uniform(min_mic_coordinate, max_mic_coordinate, (3,))
            z1 = 0 if plan else z1

            # Check the minimum distance between microphones
            while any(map(lambda m: euclidian_distance(x1, y1, z1, *m.get_coord) < min_mics_distance, mics)):
                x1, y1, z1 = np.random.uniform(min_mic_coordinate, max_mic_coordinate, (3,))
                z1 = 0 if plan else z1

            mics.append(Microphone(f"m{k}", x1, y1, z1))

        # Create sources
        sources = []
        coordinates = []

        selected_source_amount = random.randint(min_source_amount, max_source_amount)

        for k in range(selected_source_amount):
            x, y, z = np.random.uniform(min_source_coordinate, max_source_coordinate, (3,))
            z = 0 if plan else z

            # Check the minimum distance between sources and microphones
            while (any(map(lambda c: euclidian_distance(x, y, z, *c) < min_sources_distance, coordinates)) and
                   any(map(lambda m: euclidian_distance(x, y, z, *m.get_coord) < min_mics_distance, mics))):
                x, y, z = np.random.uniform(min_source_coordinate, max_source_coordinate, (3,))
                z = 0 if plan else z

            sources.append(Source(
                sound=random.choice(sounds_list),
                name=f"s{k}",
                x=x, y=y, z=z
            ))
            coordinates.append((x, y, z))

        # Create the environment
        env = Environment(
            sources=sources,
            microphones=mics,
            rate=None,
            microphone_tensors=[],
            temperature=temperature
        )

        return env

    return generator


def func_dataset_generator(data_length: int,
                           sound_start_interval: tuple[float, float] = (1.0, 60.0),
                           compress: tuple[bool, bool] = (True, True),
                           noise: tuple[float, float] = (0.0025, 0.0)
                           ) -> Callable[
    [Environment, IsolateModel], tuple[list[float_array], list[float_array]]]:
    """
    This function returns a generatr that generate a piece of dataset
    :param data_length: Length of the dataset to generate
    :param sound_start_interval: Interval of time for the dataset
    :param compress: If the audio is compressed
    :param noise: True to add noise in the signal
    :return: A dataset generator.
    """
    min_sound_start, max_sound_start = sound_start_interval

    def generator(env: Environment, model: IsolateModel):
        input_number = model.input_number

        start = random.uniform(min_sound_start, max_sound_start)
        end = (start + data_length / env.get_rate * 2)  # * 2 because it takes the center of the data after

        selected_mics = [m for m in random.sample(list(env.microphones.values()), input_number)]

        _, records = env.listen_from_microphone([m.name for m in selected_mics], start, end, start - 1,
                                                attenuate=True,
                                                compress=compress[0],
                                                noise=noise[0]
                                                )

        # Select a source to isolate
        selected = random.choice(list(env.sources.values()))
        x, y, z = selected.get_coord

        shifted_records = align_records(x, y, z, selected_mics, records, env.get_rate, env.temperature)
        ltu = len(shifted_records[0])
        shifted_records = [sr[:ltu // 2] for sr in shifted_records]

        _, result = env.listen_from_microphone((x, y, z),
                                               start, end, start - 1,
                                               attenuate=False,
                                               specific_sound=[selected.name],
                                               compress=compress[1],
                                               noise=noise[1])
        result = result[0]
        result = result[:ltu // 2]

        return shifted_records, [result]

    return generator


def train_isolate_model(model,

                        generate_env_func: Callable[[], Environment],
                        dataset_generator_func: Callable[[Environment, IsolateModel], tuple[Any, Any, list[Any]]],

                        global_epoch: int,
                        data_amount: int,
                        data_diversity: int,

                        batch_size: int = 1024,
                        epochs: int = 1,

                        evaluation_function_test: Callable[[IsolateModel, int], None] = None,  # K

                        ge_shift: int = 0,
                        change_lr_step: tuple[tuple[int, int, float]] = ((-1, 100, 0.001), (99, 10000, 0.0001)),

                        *args,
                        **kwargs
                        ) -> None:
    """
    This function is the main training loop for all model.
    :param model: Model to train.
    :param generate_env_func: Function to generate an environment.
    :param dataset_generator_func: Function to generate a dataset from an environment.
    :param global_epoch: Total number of epoch to make
    :param data_amount: Amount of data for each epoch
    :param data_diversity: Amount of different environment for each epoch
    :param batch_size: Batch size for training
    :param epochs: Number of epaoch in each training loop
    :param evaluation_function_test: Function to evaluate the model at each global epoch
    :param ge_shift: Number of the first epoch.
    :param change_lr_step: Learning rate description
    :param args: *
    :param kwargs: **
    :return: None
    """
    for ge in range(ge_shift + 1, global_epoch):
        print(f"========================== Global epoch {ge:7d}/{global_epoch} ==========================")

        def change_lr(*a, **k) -> float | None:
            _ = a, k
            for a, b, lr in change_lr_step:
                if a <= ge < b:
                    return lr

            return None

        callback = LearningRateScheduler(change_lr)

        x_train = []
        y_train = []

        for _ in range(data_diversity):
            env = generate_env_func()

            for amount in range(data_amount // data_diversity):
                s, r = dataset_generator_func(env, model)

                x, y = model.edit_dataset(s, r)

                x_train += x
                y_train += y

        x_train = np.array(x_train)
        y_train = np.array(y_train)

        model.train_model(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=[callback], *args, **kwargs)

        if evaluation_function_test is not None:
            evaluation_function_test(model, ge)


def reload_train_and_evaluate(name: str,
                              path: str,
                              save_epoch: int,
                              sounds_list_eval: list[Sound],
                              rate: int = 44100,
                              ) -> tuple[Callable[[IsolateModel, int], None], int, list[float_array]]:
    """
    This function creates the evaluation environment for the model.
    And also load past information about the model (create it if the model is new).
    :param name: The name of the file where the data about the model and its training is stored.
    :param path: The paths of the file where the data about the model and its training is stored.
    :param save_epoch: How often model weights are saved.
    :param sounds_list_eval: Liste of sound to use to evaluate model.
    :param rate: Rate of records.
    :return: The evaluation function, the last epoch saved, the last model saved
    """
    global loaded_data

    (r1, r2, r3), (real1, real2, real3), mics = env_test_isolation(sounds_list_eval)

    # Create the result history file (if it already exists, load it)
    if os.path.exists(f'{path}{name}.npy'):
        print("Loading existing data.")
        train_results_dict = np.load(f"{path}{name}.npy", allow_pickle=True).item()
    else:
        train_results_dict = {"xm": [], "weights": []}

        np.save(f'{path}{name}.npy', deepcopy(train_results_dict))

    loaded_data = np.load(f"{path}{name}.npy", allow_pickle=True).item()

    # Model weights = None is no previous model saved
    w = train_results_dict["weights"][-1] if len(train_results_dict["weights"]) > 0 else None

    def evaluation(model: IsolateModel, n: int) -> None:
        """
        This function is used to save the performance of the model.
        :param model: Model to save.
        :param n: Epoch.
        :return: None
        """
        # Save the new model
        loaded_data["xm"].append(n)
        loaded_data["weights"].append(deepcopy(model.get_weights()))

        # Then add new and previous data to the queue to be saved
        if n % save_epoch == 0:
            saving_thread_queue.append((f'{path}{name}.npy', loaded_data))

        # Use neural network to predict the isolation of many sources
        if EVALUATE:
            o1 = isolate_source_rnn_triple(9, 5, 0, mics, [r1, r2, r3], rate, model)
            o2 = isolate_source_rnn_triple(3, 3, 0, mics, [r1, r2, r3], rate, model)
            o3 = isolate_source_rnn_triple(6, 2, 0, mics, [r1, r2, r3], rate, model)

            a1 = absolute_error(signal=o1, original=real1, raw=True) / 0.31
            a2 = absolute_error(signal=o2, original=real2, raw=True) / 0.31
            a3 = absolute_error(signal=o3, original=real3, raw=True) / 0.31
            q1 = relative_error(signal=o1, original=real1, raw=True)
            q2 = relative_error(signal=o2, original=real2, raw=True)
            q3 = relative_error(signal=o3, original=real3, raw=True)

            p = lambda sample: f"{np.average(sample):.4f} ± {(np.std(sample, ddof=1) / len(sample) ** 0.5):.4f}"

            graph_results.append(
                (
                    (np.average((a1 + a2 + a3) / 3),
                     min(map(lambda x: np.average(x), (a1, a2, a3))),
                     max(map(lambda x: np.average(x), (a1, a2, a3)))
                     ),
                    (np.average((q1 + q2 + q3) / 3),
                     min(map(lambda x: np.average(x), (q1, q2, q3))),
                     max(map(lambda x: np.average(x), (q1, q2, q3)))
                     )
                )
            )

            print(f"Average absolute error:          {p(a1)} | {p(a2)} | {p(a3)}")
            print(f"Average relative error:          {p(q1)} | {p(q2)} | {p(q3)}")

            # Get the previous data
            # data = np.load(f"{path}train_results {name}.npy", allow_pickle=True).item()

            # Del to avoir memory saturation (in case Python does not do it)
            del a1, a2, a3, o1, o2, o3, q1, q2, q3

    return evaluation, max(train_results_dict.get("xm", 0), default=0), w


def save_and_plot_thread(gui: bool = True) -> None:
    """
    This function, run as a thread function, saves data placed in queue
    :return: None
    """

    def save_files() -> None:
        """
        This function saves data placed in saving queue.
        :return: None
        """
        # If there is data to save in the queue
        if len(saving_thread_queue) > 0:
            # Save data
            print(f"Started saving training data ({len(saving_thread_queue) - 1} remaining).")
            t = time.time()
            path, data = saving_thread_queue.pop()
            np.save(path, data)
            print(f"Training saved at {datetime.now()} in {time.time() - t:.2f} secondes.")

    def update_func(i: int) -> tuple:
        # save new data
        save_files()

        if len(graph_results) < 2:
            return i,

        if not main_thread().is_alive():
            raise StopIteration

        x = list(range(len(graph_results)))

        y1 = list(map(lambda t: t[0][0], graph_results))
        y1_b = list(map(lambda t: t[0][1], graph_results))
        y1_t = list(map(lambda t: t[0][2], graph_results))

        y2 = list(map(lambda t: t[1][0], graph_results))
        y2_b = list(map(lambda t: t[1][1], graph_results))
        y2_t = list(map(lambda t: t[1][2], graph_results))

        try:
            # clear all charts
            axs[0].clear()
            axs[1].clear()

            axs[0].fill_between(x, y1_b, y1_t, alpha=0.25)
            axs[0].plot(x, y1)
            axs[0].set_title("Absolute error")

            axs[1].fill_between(x, y2_b, y2_t, alpha=0.25)
            axs[1].plot(x, y2)
            axs[1].set_title("Relative error")
        except Exception as error_:
            print(error_)

        return i,

    if gui:
        try:
            # Create a Tkinter windows because Matplotlib is not Thread safe.
            # The matplotlib chart acts as an element (Canvas) in the Tkinter window.
            matplotlib.use("TkAgg")

            fig, axs = plt.subplots(1, 2)

            # placer une image temporaire
            window = tk.Tk()
            window.config(bg="white")
            window.title("ALI - Training")

            canvas = FigureCanvasTkAgg(fig, window)
            canvas.get_tk_widget().pack(side="top", fill='both', expand=True)

            _ = animation.FuncAnimation(fig,
                                        update_func,
                                        interval=10,
                                        cache_frame_data=False)

            window.mainloop()
        except Exception as error:
            warnings.warn("-" * 76)
            warnings.warn(f"Error: {mod_name} - {error}")
            warnings.warn("-" * 76)

    while main_thread().is_alive():
        time.sleep(0.1)
        save_files()


def max_batch_size(n: int, input_length: int) -> int:
    """
    This function calculate the optimal batch size depending on the number of parameters in the model
    :param n: Number of parameter of the model
    :param input_length: Input length of the RNN
    :return: Max batch size.
    """
    f = lambda p: (1 * 10 ** 4) * p + (2.25 * 10 ** 9)

    mbs_44100 = (f(n) / f(100_000)) * (input_length / (44100 // 2))

    return floor(10 / mbs_44100)


MODELS = {
    # Raw models
    "R LSTM 64": RawIsolateModel([LSTM(64, return_sequences=True)]),
    "R LSTM 128": RawIsolateModel([LSTM(128, return_sequences=True)]),
    "R LSTM 256": RawIsolateModel([LSTM(256, return_sequences=True)]),
    "R LSTM 64-64": RawIsolateModel([LSTM(64, return_sequences=True), LSTM(64, return_sequences=True)]),
    "R LSTM 128-128": RawIsolateModel([LSTM(128, return_sequences=True), LSTM(128, return_sequences=True)]),

    "R GRU 64": RawIsolateModel([GRU(64, return_sequences=True)]),
    "R GRU 128": RawIsolateModel([GRU(128, return_sequences=True)]),
    "R GRU 256": RawIsolateModel([GRU(256, return_sequences=True)]),
    "R GRU 64-64": RawIsolateModel([GRU(64, return_sequences=True), GRU(64, return_sequences=True)]),
    "R GRU 128-128": RawIsolateModel([GRU(128, return_sequences=True), GRU(128, return_sequences=True)]),

    "R Bi-GRU 64": RawIsolateModel([Bidirectional(GRU(64, return_sequences=True))]),
    "R Bi-GRU 128": RawIsolateModel([Bidirectional(GRU(128, return_sequences=True))]),
    "R Bi-GRU 256": RawIsolateModel([Bidirectional(GRU(256, return_sequences=True))]),
    "R Bi-GRU 64-64": RawIsolateModel(
        [Bidirectional(GRU(64, return_sequences=True)), Bidirectional(GRU(64, return_sequences=True))]
    ),
    "R Bi-GRU 128-128": RawIsolateModel(
        [Bidirectional(GRU(128, return_sequences=True)), Bidirectional(GRU(128, return_sequences=True))]
    ),
    "R Bi-LSTM 64": RawIsolateModel([Bidirectional(LSTM(64, return_sequences=True))]),
    "R Bi-LSTM 128": RawIsolateModel([Bidirectional(LSTM(128, return_sequences=True))]),
    "R Bi-LSTM 256": RawIsolateModel([Bidirectional(LSTM(256, return_sequences=True))]),
    "R Bi-LSTM 64-64": RawIsolateModel(
        [Bidirectional(LSTM(64, return_sequences=True)), Bidirectional(LSTM(64, return_sequences=True))]
    ),
    "R Bi-LSTM 128-128": RawIsolateModel(
        [Bidirectional(LSTM(128, return_sequences=True)), Bidirectional(LSTM(128, return_sequences=True))]
    ),

    # Spectrogram models
    "S LSTM 64": SpectrogramIsolateModel([LSTM(64, return_sequences=True)]),
    "S LSTM 128": SpectrogramIsolateModel([LSTM(128, return_sequences=True)]),
    "S LSTM 256": SpectrogramIsolateModel([LSTM(256, return_sequences=True)]),
    "S LSTM 64-64": SpectrogramIsolateModel([LSTM(64, return_sequences=True), LSTM(64, return_sequences=True)]),
    "S LSTM 128-128": SpectrogramIsolateModel([LSTM(128, return_sequences=True), LSTM(128, return_sequences=True)]),

    "S GRU 64": SpectrogramIsolateModel([GRU(64, return_sequences=True)]),
    "S GRU 128": SpectrogramIsolateModel([GRU(128, return_sequences=True)]),
    "S GRU 256": SpectrogramIsolateModel([GRU(256, return_sequences=True)]),
    "S GRU 64-64": SpectrogramIsolateModel([GRU(64, return_sequences=True), GRU(64, return_sequences=True)]),
    "S GRU 128-128": SpectrogramIsolateModel([GRU(128, return_sequences=True), GRU(128, return_sequences=True)]),

    "S Bi-GRU 64": SpectrogramIsolateModel([Bidirectional(GRU(64, return_sequences=True))]),
    "S Bi-GRU 128": SpectrogramIsolateModel([Bidirectional(GRU(128, return_sequences=True))]),
    "S Bi-GRU 256": SpectrogramIsolateModel([Bidirectional(GRU(256, return_sequences=True))]),
    "S Bi-GRU 64-64": SpectrogramIsolateModel(
        [Bidirectional(GRU(64, return_sequences=True)), Bidirectional(GRU(64, return_sequences=True))]
    ),
    "S Bi-GRU 128-128": SpectrogramIsolateModel(
        [Bidirectional(GRU(128, return_sequences=True)), Bidirectional(GRU(128, return_sequences=True))]
    ),

    "S Bi-LSTM 64": SpectrogramIsolateModel([Bidirectional(LSTM(64, return_sequences=True))]),
    "S Bi-LSTM 128": SpectrogramIsolateModel([Bidirectional(LSTM(128, return_sequences=True))]),
    "S Bi-LSTM 256": SpectrogramIsolateModel([Bidirectional(LSTM(256, return_sequences=True))]),
    "S Bi-LSTM 64-64": SpectrogramIsolateModel(
        [Bidirectional(LSTM(64, return_sequences=True)), Bidirectional(LSTM(64, return_sequences=True))]
    ),
    "S Bi-LSTM 128-128": SpectrogramIsolateModel(
        [Bidirectional(LSTM(128, return_sequences=True)), Bidirectional(LSTM(128, return_sequences=True))]
    ),
}

if __name__ == '__main__':
    RATE = 44100
    D = 2  # 2
    LEN = RATE // 2

    PATH_TRAIN: str = __file__.replace("\\", "/").replace("audio_locate_isolate/neural_networks.py", "sound_data/")
    PATH_EVAL: str = __file__.replace("\\", "/").replace("audio_locate_isolate/neural_networks.py", "sound_data/test/")

    sounds_list_wav = [Sound(f"{PATH_TRAIN}{f}", rate=RATE) for f in os.listdir(PATH_TRAIN) if "test" not in f]
    sounds_list_eval_wav = [Sound(f"{PATH_EVAL}{f}", rate=RATE) for f in os.listdir(PATH_EVAL)]

    models_names = list(MODELS.keys())

    thread0 = Thread(target=save_and_plot_thread, daemon=True, kwargs={"gui": EVALUATE})
    thread0.start()

    for number, mod_name in enumerate(models_names):
        graph_results = []

        print("\n\n\n" + "-" * 76 * 2 + "\n\n\n" + "=" * 76)
        print(f"Training {number + 1}/{len(models_names)}: {mod_name}")
        print("=" * 76)
        mod: IsolateModel = MODELS[mod_name]
        mod.summary()

        max_bs = max_batch_size(mod.neural_network.count_params(), LEN) * 5
        print(f"Max batch_size: {max_bs}")

        env_func = func_generate_train_env(sounds_list_wav, source_amount=(3, 3))
        dataset_func = func_dataset_generator(LEN)

        eval_func, last_epoch, last_weights = reload_train_and_evaluate(name=mod_name,
                                                                        path="res/",
                                                                        save_epoch=5,
                                                                        sounds_list_eval=sounds_list_eval_wav, )
        if last_weights is not None:
            mod.set_weights(last_weights)

        # Start saving and charting thread
        print(f"Starting from: {last_epoch}")

        try:
            train_isolate_model(mod, env_func, dataset_func, 251, 10 * D, 5 * D,
                                batch_size=max_bs,
                                epochs=10,  # 5
                                ge_shift=last_epoch,
                                evaluation_function_test=eval_func)
        except Exception as err:
            warnings.warn("-" * 76)
            warnings.warn(f"Error: {mod_name} - {err}")
            warnings.warn("-" * 76)

    print("Train finished")
    thread0.join()
