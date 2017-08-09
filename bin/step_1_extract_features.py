"""Step 0: Create instances from MNE-Raw Files."""
import sys
import os
# Change
sys.path.insert(0, os.path.abspath("."))

import fire
import mne
from sklearn_pandas import DataFrameMapper
from p300.feature_extraction import FrequencyExtractor, LoadArray
from sklearn.pipeline import make_pipeline
import pandas as pd

mne.set_log_level("WARNING")


def create_instances(input_csv="output/output.csv",
                     output_path="output/instances.csv"):
    """Feature extraction from  Raw EEG files.

    Parameters:
    -----------

    input_csv: string
        Path to CSV of instances

    output_csv: string
        Where to output
    """
    print("Reading csv from {}".format(input_csv))
    df = pd.read_csv(input_csv)

    load_array = DataFrameMapper([
        ('array_path', LoadArray()),
    ], input_df=True)

    pipe = make_pipeline(load_array, FrequencyExtractor())

    print("Extracting features")
    features = pipe.fit_transform(df)
    feature_names = pipe.named_steps['frequencyextractor'].get_feature_names()

    df_features = pd.DataFrame(features, columns=feature_names)

    output = pd.concat([
        df,
        df_features
    ], axis=1)
    output.to_csv(output_path)

    print("Features saved at {}".format(output_path))


if __name__ == '__main__':
    fire.Fire(create_instances)
