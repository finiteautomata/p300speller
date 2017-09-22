"""Step 0: Create instances from MNE-Raw Files."""
import fire
import mne
import os
import re
import glob
import numpy as np
import pandas as pd

mne.set_log_level("WARNING")
CORPORA_PATH = "~/projects/corpora/P3Speller/P3Speller-old-y-datos/sets/"


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
    def event_id_func(x):
        return 1 if x == "0" else 2

    print("Extracting epochs from {}".format(filename))

    data_mne = mne.io.read_raw_eeglab(
        filename,
        event_id_func=event_id_func,
        preload=True)
    data_mne.filter(1, 20)

    """
    Find events (stimuli)

    The first column contains the event time
    in samples and the third column contains the event id. For output =
    'onset' or 'step', the second column contains the value of the stim
    channel immediately before the event/step. For output = 'offset',
    the second column contains the value of the stim channel after the
    event offset.
    """

    events = mne.find_events(data_mne)
    baseline = (None, 0)
    """
    Build epochs

    D is distractor
    T is target

    Note: I guess this is of no use...
    """
    event_id = {"D": 1, "T": 2}
    epochs = mne.Epochs(
        data_mne, events, event_id,
        baseline=baseline, tmin=-0.1, tmax=0.7)
    epochs.load_data()

    if len(events) != len(epochs):
        # We've got a problem here
        msg = "Events and epochs do not match in {}".format(filename)
        raise ValueError(msg)

    return epochs, events


def get_subject_id(path):
    """
    Return subject id from path to .set file

    Parameters
    ----------

    path: string
        path to .set file
    """
    match = re.match(r".*_(\d*)\.set", path)

    if not match:
        raise ValueError("*** File path does not match 'name_id.set' pattern")

    return match.groups()[0]


def create_instances(path_to_sets=CORPORA_PATH,
                     output_path="output/preinstances.h5"):
    """Create instances from Raw EEG files.

    Parameters:
    -----------

    path_to_sets: string
        Path to directory containing .set files
    """
    file_path = os.path.expanduser(path_to_sets)
    files = glob.glob(os.path.join(file_path, "*.set"))

    instances = []

    hdf = pd.HDFStore(output_path)

    for filename in files:
        try:
            instances = []

            subject_id = get_subject_id(filename)
            epochs, events = get_epochs_from(filename)

            for i, (trial, event) in enumerate(zip(epochs, events)):
                instance_id = "{}_{}".format(subject_id, i)
                instance_filename = "output/npy/{}.npy".format(instance_id)
                instance_filename = os.path.abspath(instance_filename)

                np.save(instance_filename, trial)
                instances.append({
                    'id': instance_id,
                    'subject_id': subject_id,
                    'index': i,
                    'event_time': event[0],
                    'event_type': event[2],
                    'target': event[2] == 2,
                    'array_path': instance_filename,
                    'sfreq': epochs.info.get('sfreq'),
                    'ch_names': ",".join(epochs.ch_names),
                })

            df = pd.DataFrame(instances)
            df.set_index("id", inplace=True)

            hdf.put("subjects/s{}".format(subject_id), df)
        except ValueError as e:
            print("*** {}".format(e))
            continue

    hdf.close()
    print("Preinstances saved to {}".format(output_path))


if __name__ == '__main__':
    fire.Fire(create_instances)
