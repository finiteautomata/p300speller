"""Step 0: Create instances from MNE-Raw Files."""
import fire
import mne
import os
import re
import glob
import numpy as np
import pandas as pd
import sys
from p300.preprocessing import (
    get_epochs_from,
    get_subject_id,
    create_instances_from,
)
from p300.data import Store

mne.set_log_level("WARNING")
CORPORA_PATH = "~/projects/corpora/P3Speller/P3Speller-old-y-datos/sets/"


def run(path_to_sets=CORPORA_PATH,
        output_path="output/instances.h5",
        group="default"):
    """Create instances from Raw EEG files.

    Parameters:
    -----------

    path_to_sets: string
        Path to directory containing .set files

    output_path: path
        Path to HDF file

    group: string
        group to store in the hdf
    """
    file_path = os.path.expanduser(path_to_sets)
    files = glob.glob(os.path.join(file_path, "*.set"))

    instances = []

    with Store(output_path, group) as store:
        store.clean()

        for filename in files:
            try:
                df = create_instances_from(filename)
                subject_id = get_subject_id(filename)

                store.put_subject_data(subject_id, df)
            except ValueError as e:
                print("*** {}".format(e))
                continue

        print("Preinstances saved to {} group {}".format(output_path, group))


if __name__ == '__main__':
    fire.Fire({"run": run})
