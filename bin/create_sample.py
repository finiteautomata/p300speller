"""Step 0: Create instances from MNE-Raw Files."""
from p300.data import Store
import os
import fire
import mne
import numpy as np


def run(input_path="output/instances.h5", group="default", sample_size=10):
    """Feature extraction from  Raw EEG files.

    Parameters:
    -----------

    input_path: string
        Path to hdf file

    group: string
        group of HDF

    sample_size: integer (default=10)
        Sample size to save
    """
    print("Reading from {} at group {}".format(input_path, group))

    sample_group = "{}_sample".format(group)

    with Store(input_path, group) as store:
        sample_ids = np.random.choice(store.subject_ids, sample_size)

        for subject_id in sample_ids:
            group = "{}/{}".format(sample_group, subject_id)
            store.put(group, store.get_subject_data(subject_id))

    print("{} subjects saved to {}".format(sample_size, sample_group))

if __name__ == '__main__':
    fire.Fire({
        "run": run
    })
