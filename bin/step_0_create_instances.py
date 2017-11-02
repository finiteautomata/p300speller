"""Step 0: Create instances from MNE-Raw Files."""
import fire
import mne
import os
import re
import glob
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, os.path.abspath("."))
from p300.preprocessing import get_epochs_from, get_subject_id

mne.set_log_level("WARNING")
CORPORA_PATH = "~/projects/corpora/P3Speller/P3Speller-old-y-datos/sets/"

def create_instances_from(filename):
    subject_id = get_subject_id(filename)
    epochs, events = get_epochs_from(filename)
    instances = []
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

    return pd.DataFrame(instances)

def create_instances(path_to_sets=CORPORA_PATH,
                     output_path="output/instances.h5"):
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
            df = create_instances_from(filename)
            subject_id = get_subject_id(filename)
            
            hdf.put("subjects/s{}".format(subject_id), df, format='t')
        except ValueError as e:
            print("*** {}".format(e))
            continue

    hdf.close()
    print("Preinstances saved to {}".format(output_path))


if __name__ == '__main__':
    fire.Fire(create_instances)
