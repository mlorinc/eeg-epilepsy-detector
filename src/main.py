import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.signal
from typing import List
from autoreject import AutoReject
import features.tf_features as features

def eeg_power_band(epochs, picks: str):
    # specific frequency bands
    FREQ_BANDS = {"delta": [0.5, 4.5],
                  "theta": [4.5, 8.5],
                  "alpha": [8.5, 11.5],
                  "sigma": [11.5, 15.5],
                  "beta": [15.5, 30]}

    spectrum = epochs.compute_psd(picks='eeg', fmin=0.5, fmax=25.)
    psds, freqs = spectrum.get_data(return_freqs=True)
    # Normalize the PSDs
    psds /= np.sum(psds, axis=-1, keepdims=True)

    X = []
    for fmin, fmax in FREQ_BANDS.values():
        data = psds[:, :, (freqs >= fmin) & (freqs < fmax)]
        psds_band_mean = data.mean(axis=-1)
        psds_band_var = data.var(axis=-1)
        psds_band_std = data.std(axis=-1)
        psds_band_average = data.average(axis=-1)

        X.append(psds_band_mean.reshape(len(psds), -1))

    return np.concatenate(X, axis=1)

def power_spectrum(epoch: mne.io.edf.edf.RawEDF, picks: str):
    f, pxx = scipy.signal.welch(epoch.get_data(picks=[picks]), fs=256, nfft=256, scaling="spectrum")
    print(pxx)
    print(f)
    # pxx = 10 * np.log10(pxx)
    return f.reshape(-1), pxx.reshape(-1)

def ica(raw, n_components):
    ica = mne.preprocessing.ICA(n_components=n_components, max_iter="auto", random_state=42)
    ica.fit(raw)
    ica.apply(raw)
    return raw

def main():
    raw = mne.io.read_raw_edf('physionet.org/files/chbmit/1.0.0/chb01/chb01_03.edf')

    raw.drop_channels(["P7-T7"])
    raw = raw.load_data().filter(l_freq=0.5, h_freq=25, n_jobs=-1, method="fir", fir_design="firwin")
    # raw.resample(128, npad="auto")
    # raw = ica(raw, len(raw.ch_names))

    events, event_id = mne.events_from_annotations(raw)
    print(event_id)
    print(events)

    # Get the channel names
    channel_names = raw.ch_names

    # Get the sampling frequency
    sampling_rate = int(raw.info['sfreq'])

    # Get the number of samples in the data
    num_samples = raw.n_times

    print(raw.info)
    print("names: ", channel_names)
    print("Count: ", num_samples)
    print("Rate: ", sampling_rate)

    # raw.plot_psd(ax=ax, fmin=0.5, fmax=25, tmin=2989, tmax=2992, picks=["FP2-F4"], n_fft=sampling_rate, n_overlap=1, estimate="welch")
    # raw.plot_psd(ax=ax, fmin=0.5, fmax=25, tmin=2994, tmax=2997, picks=["FP2-F4"], n_fft=sampling_rate, n_overlap=1, estimate="welch")

    epochs = mne.make_fixed_length_epochs(raw, duration=2, preload=True) 
    # print(len(epochs.events))

    # epochs.plot_image()

    # power = mne.time_frequency.tfr_morlet(epochs, n_cycles=1, return_itc=False,
    #                                     freqs=[1, 4.5, 8.5, 11.5, 15.5, 30], decim=3, use_fft=True)
    # power.plot(["FP2-F4"])

    sleep = epochs[1495].pick_channels(["FP2-F4"])
    seizure = epochs[1498].pick_channels(["FP2-F4"])

    psd = epochs.compute_psd(fmin=0.25, fmax=24.5, n_jobs=-1, verbose=True, bandwidth=8)
    features.save_edf_features("test.csv", epochs, psd, 3)

    # for i in range(2996, 3036+6, 6):
    #     seizure = psd[i//2:i//2+3]
    #     seizure.plot(color="b", axes=ax, picks=["FP2-F4"], spatial_colors=False)
    # for i in range(1024, 2000, 6):
    #     normal = psd[i//2:i//2+3]
    #     normal.plot(color="g", axes=ax, picks=["FP2-F4"], spatial_colors=False)

    # normal = psd[10//2:10//2+3]
    # sleep = psd[2990//2:2990//2+3]

    # sleep.plot(color="r", axes=ax, picks=["FP2-F4"], spatial_colors=False)
    # normal.plot(color="g", axes=ax, picks=["FP2-F4"], spatial_colors=False)

    # # psd.plot()
    # plt.show()
    # Close the EDF file
    raw.close()

main()