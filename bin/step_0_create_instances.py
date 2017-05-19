"""Step 0: Create instances from MNE-Raw Files."""
import fire
import mne
import os
import glob

mne.set_log_level("WARNING")


def get_epochs_from(filename):
    """Extract epochs from EEG file.

    Parameters
    ----------

    filename: path
        Path to Raw .set file

    Returns
    -------

    epochs: mne.Epochs
        Epochs created from the EEG file
    events: array, shape
        Events created from file

        According to mne.find_events documentation

        All events that were found. The first column contains the event time
        in samples and the third column contains the event id. For output =
        'onset' or 'step', the second column contains the value of the stim
        channel immediately before the event/step. For output = 'offset',
        the second column contains the value of the stim channel after the
        event offset.


    """
    print("Extracting epochs from {}".format(filename))

    def event_id_func(x):
        return 1 if x == "0" else 2

    event_id = {"D": 1, "T": 2}

    data_mne = mne.io.read_raw_eeglab(filename, event_id_func=event_id_func, preload=True)
    data_mne.filter(1, 20)

    # Find events (stimuli)
    events = mne.find_events(data_mne)
    baseline = (None, 0)
    # Build epochs
    epochs = mne.Epochs(data_mne, events, event_id, baseline=baseline, tmin=-0.1, tmax=0.7)
    epochs.load_data()

    return epochs, events


def create_instances():
    """Create instances from Raw EEG files."""
    file_path = os.path.expanduser("~/projects/corpora/P3Speller/P3Speller-old-y-datos/sets/")
    files = glob.glob(os.path.join(file_path, "*.set"))

    for filename in files:
        epochs, events = get_epochs_from(filename)

if __name__ == '__main__':
    fire.Fire(create_instances)
