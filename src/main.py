import mne
import numpy as np
import pandas as pd
import scipy.signal
import features.tf_features as features
import features.summary as summary

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
    features.create_patients_feature_dataset(3, summary.get_seizure_summary(), (12, 13))

main()