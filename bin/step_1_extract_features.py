"""Step 0: Create instances from MNE-Raw Files."""
import fire
import mne
import os
import re
import glob
import numpy as np
import pandas as pd

mne.set_log_level("WARNING")



def create_instances(input_csv="output/output.csv",
                     output_path="output/output.csv"):
    """Feature extraction from  Raw EEG files.

    Parameters:
    -----------

    input_csv: string
        Path to CSV of instances

    output_csv: string
        Where to output
    """
    df = pd.DataFrame(instances)
    df.set_index("id", inplace=True)
    df.to_csv(output_path)


if __name__ == '__main__':
    fire.Fire(create_instances)
