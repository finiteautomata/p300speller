import mne
import numpy as np

def __normalize_subject(X):
    """
    Z-score normalizes per-column
    """
    mean = X.mean(axis=(0, 2)).reshape(-1, 1)
    std = X.std(axis=(0, 2)).reshape(-1, 1)
    return (X - mean) / std

def load_data(filename, normalize=True):
    """
    Loads data from .set file into a design matrix and target vector.

    Params
    ------

    filename: path to .set file
        Data from subject

    normalize: Boolean (default=True)
        Indicates whether data should be normalized with a z-score

        Normalization is performed channel-wise.
    """

    data_mne = mne.io.read_raw_eeglab(filename, preload=True, event_id={"0": 1, "1": 2})
    data_mne.filter(0, 20)
    events = mne.find_events(data_mne)
    epochs = mne.Epochs(
        data_mne, events,
        baseline=(None, 0), tmin=-0.1, tmax=0.7)

    epochs.load_data()

    ch_names = epochs.ch_names

    X = epochs.get_data()[:, :-1]
    y = (events[:, 2] == 2).astype('float')

    if normalize:
        X = __normalize_subject(X)

    return X, y

def load_multiple_data(paths, **kwargs):
    """
    Loads data from multiple filenames
    """
    X = None
    y = None
    for path in paths:
        try:
            X_subject, y_subject = load_data(path, **kwargs)

            if X is None:
                X, y = X_subject, y_subject
            else:
                X = np.vstack((X, X_subject))
                y = np.vstack((y.reshape(-1,1), y_subject.reshape(-1,1)))
        except ValueError as e:
            print(e)

    return X, y