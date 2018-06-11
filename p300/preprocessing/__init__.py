"""Processing module."""
import mne
import os
import re
import glob
import numpy as np
import pandas as pd


def normalize_subject(X):
    mean = X.mean(axis=(0, 2)).reshape(-1, 1)
    std = X.std(axis=(0, 2)).reshape(-1, 1)
    return (X - mean) / std    


def load_data_from_subject(filename, normalize=True):
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

    if len(events) != len(epochs):
        raise ValueError("Epochs events mismatch")
    if normalize: 
        X = normalize_subject(X)
    
    
    return X, y 

def load_data(filenames):
    X = None
    y = None
    for filename in filenames:
        try:
            X_subject, y_subject = load_data_from_subject(filename)

            if X is None:
                X, y = X_subject, y_subject
            else:
                print(X.shape, X_subject.shape)
                X = np.vstack((X, X_subject))
                print(y.shape, y_subject.shape)
                y = np.vstack((y.reshape(-1,1), y_subject.reshape(-1,1)))
        except ValueError as e:
            print(e)
    return X, y