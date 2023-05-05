import mne
import numpy as np
import pandas as pd
import abc
from typing import List
import scipy.stats
import scipy.signal

f_bands = np.arange(0.5, 24.5+3, 3)


class Features(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def to_vector(self) -> List[None]:
        pass

class PSDFeatures(Features):
    def __init__(self) -> None:
        # Dimension (bands, channels) - (M, N)
        self.energies = None
        # Dimension(channels)
        self.mean = None
        self.std = None
        self.var = None
        self.kurtosis = None
        self.skew = None

    def to_vector(self):
        return np.concatenate((
            self.energies,
            self.mean,
            self.std,
            self.var,
            self.kurtosis,
            self.skew
        ))

    @staticmethod
    def labels(channels: List[str], suffix: str):
        energies = [f"psd_energy_{f1}hz_{f2}hz_{ch}_w{suffix}" for ch in channels for f1, f2 in zip(
            f_bands[:-1], f_bands[1:])]
        other_labels = ["psd_mean", "psd_std",
                        "psd_var", "psd_kurtosis", "psd_skew"]

        return energies + [f"{label}_{ch}_w{suffix}" for label in other_labels for ch in channels]


class EpochFeatures(Features):
    def __init__(self) -> None:
        self.mean = None
        self.std = None
        self.var = None
        self.kurtosis = None
        self.skew = None
        self.local_min_count = None
        self.local_max_count = None
        self.mode = None
        self.q1 = None
        self.q2 = None
        self.q3 = None
        self.iqr = None

    def to_vector(self):
        return np.concatenate((
            self.mean,
            self.std,
            self.var,
            self.kurtosis,
            self.skew,
            self.local_min_count,
            self.local_max_count,
            self.mode,
            self.q1,
            self.q2,
            self.q3,
            self.iqr
        ))
    @staticmethod
    def labels(channels: List[str], suffix: str) -> List[str]:
        other_labels = [
            "epoch_mean",
            "epoch_std",
            "epoch_var",
            "epoch_kurtosis",
            "epoch_skew",
            "epoch_local_min_count",
            "epoch_local_max_count",
            "epoch_mode",
            "epoch_q1",
            "epoch_q2",
            "epoch_q3",
            "epoch_iqr"
        ]
        return [f"{label}_{ch}_w{suffix}" for label in other_labels for ch in channels]


def extract_psd_features(epoch_psd: mne.epochs.EpochsSpectrum) -> PSDFeatures:
    psd_features = PSDFeatures()
    psd, freqs = epoch_psd.get_data(return_freqs=True)

    psd_features.mean = psd.mean(axis=-1).flatten()
    psd_features.var = psd.var(axis=-1).flatten()
    psd_features.std = psd.std(axis=-1).flatten()
    psd_features.kurtosis = scipy.stats.kurtosis(psd, axis=-1).flatten()
    psd_features.skew = scipy.stats.skew(psd, axis=-1).flatten()

    psd_features.energies = []
    for fmin, fmax in zip(f_bands[:-1], f_bands[1:]):
        data = psd[:, :, ((freqs < fmax) & (freqs >= fmin))]
        psd_energy = data.sum(axis=-1).flatten()
        psd_features.energies.append(psd_energy)
    # place bands before channels
    psd_features.energies = np.array(
        psd_features.energies).transpose().flatten()
    return psd_features


def extract_epoch_features(epoch: mne.Epochs) -> EpochFeatures:
    epoch_features = EpochFeatures()
    data = epoch.get_data()[0, :, :]
    q = np.quantile(data, [0.25, 0.5, 0.75], axis=-1)
    epoch_features.mean = data.mean(axis=-1).flatten()
    epoch_features.std = data.std(axis=-1).flatten()
    epoch_features.var = data.var(axis=-1).flatten()
    epoch_features.kurtosis = scipy.stats.kurtosis(data, axis=-1).flatten()
    epoch_features.skew = scipy.stats.skew(data, axis=-1).flatten()
    epoch_features.local_min_count = np.array(
        [scipy.signal.argrelmin(channel, axis=-1)[0].shape[0]
         for channel in data]
    )
    epoch_features.local_max_count = np.array(
        [scipy.signal.argrelmax(channel, axis=-1)[0].shape[0] for channel in data])
    epoch_features.mode = scipy.stats.mode(
        data, axis=-1, keepdims=True)[0].flatten()
    epoch_features.q1 = q[0].flatten()
    epoch_features.q2 = q[1].flatten()
    epoch_features.q3 = q[2].flatten()
    epoch_features.iqr = epoch_features.q3 - epoch_features.q1

    assert (epoch_features.mean.shape == epoch_features.std.shape)
    assert (epoch_features.std.shape == epoch_features.var.shape)
    assert (epoch_features.var.shape == epoch_features.kurtosis.shape)
    assert (epoch_features.kurtosis.shape == epoch_features.skew.shape)
    assert (epoch_features.skew.shape == epoch_features.local_min_count.shape)
    assert (epoch_features.local_min_count.shape ==
            epoch_features.local_max_count.shape)
    assert (epoch_features.local_max_count.shape == epoch_features.mode.shape)
    assert (epoch_features.mode.shape == epoch_features.q1.shape)
    assert (epoch_features.q1.shape == epoch_features.q2.shape)
    assert (epoch_features.q2.shape == epoch_features.q3.shape)
    assert (epoch_features.q3.shape == epoch_features.iqr.shape)
    return epoch_features


def get_labels(channels: List[str], count: int) -> List[str]:
    labels = []
    for i in range(1, count + 1, 1):
        psd_labels = PSDFeatures.labels(channels, str(i))
        epoch_labels = EpochFeatures.labels(channels, str(i))
        labels += psd_labels + epoch_labels
    return labels


def get_vector(psd_features: PSDFeatures, epoch_features: EpochFeatures) -> np.ndarray:
    psd_vector = psd_features.to_vector()
    epoch_vector = epoch_features.to_vector()

    return np.concatenate((psd_vector, epoch_vector))


def get_window(epochs: mne.Epochs, psds: mne.time_frequency.EpochsSpectrum):
    assert (len(epochs) == 3)
    assert (len(psds) == 3)

    vectors = np.array([])
    for i in range(len(epochs)):
        epoch_features = extract_epoch_features(epochs[i])
        psd_features = extract_psd_features(psds[i])
        vectors = np.concatenate((vectors, get_vector(psd_features, epoch_features)))
    return vectors

def edf_to_pandas(epochs: mne.Epochs, spectrum: mne.time_frequency.EpochsSpectrum, window: int):
    for i in range(len(epochs) - window + 1):
        yield get_window(epochs[i:i+window], spectrum[i:i+window])
    return None

def save_edf_features(filename: str, epochs: mne.Epochs, spectrum: mne.time_frequency.EpochsSpectrum, window: int):
    labels = get_labels(epochs.ch_names, window)
    vectors = edf_to_pandas(epochs, spectrum, window)
    df = pd.DataFrame(data=vectors)
    df.to_csv(filename, header=False)

def load_features() -> pd.DataFrame:
    return pd.read_pickle("example.pck")
