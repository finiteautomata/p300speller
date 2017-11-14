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

mne.set_log_level("WARNING")
CORPORA_PATH = "~/projects/corpora/P3Speller/P3Speller-old-y-datos/sets/"


def run(path_to_sets=CORPORA_PATH,
                     output_path="output/instances.h5",
                     key="subjects"):
    """Create instances from Raw EEG files.

    Parameters:
    -----------

    path_to_sets: string
        Path to directory containing .set files

    output_path: path
        Path to HDF file

    key: string
        Key to store in the hdf
    """
    file_path = os.path.expanduser(path_to_sets)
    files = glob.glob(os.path.join(file_path, "*.set"))

    instances = []

    hdf = pd.HDFStore(output_path)

    try:
        hdf.remove(key)
    except:
        pass

    for filename in files:
        try:
            df = create_instances_from(filename)
            subject_id = get_subject_id(filename)

            hdf.put(key, df, append=True, format='t')
        except ValueError as e:
            print("*** {}".format(e))
            continue

    hdf.close()
    print("Preinstances saved to {}".format(output_path))


if __name__ == '__main__':
    fire.Fire({"run": run})
