"""Step 0: Create instances from MNE-Raw Files."""
from p300.feature_extraction import (
    LoadArray,
    SubsamplingExtractor,
    WaveletExtractor
)
from p300.data import Store
from sklearn.pipeline import make_pipeline, FeatureUnion
from sklearn_pandas import DataFrameMapper
import os
# Change
import fire
import mne
import pandas as pd

mne.set_log_level("WARNING")


def create_frequency_extractors(window_sizes):
    """Create frequency extractor."""
    def make_tuple(window_size):
        name = 'freq_ws_{}'.format(window_size)
        return (name, FrequencyExtractor(window=window_size))

    return [make_tuple(window_size) for window_size in window_sizes]


def create_extractor():
    """Create feature extractor."""
    load_array = DataFrameMapper([
        ('array_path', LoadArray()),
    ], input_df=True)

    feature_union = FeatureUnion([
        ('subsample_', SubsamplingExtractor(13)),
        ('wavelets_', WaveletExtractor())
    ])

    pipe = make_pipeline(
        load_array,
        feature_union
    )

    return pipe


def extract_features_for(store, subject_id):
    print("Extracting features from {}".format(subject_id))

    df = store.get_subject_data(subject_id)
    pipe = create_extractor()
    features = pipe.fit_transform(df)
    feature_names = pipe.steps[-1][1].get_feature_names()

    df_features = pd.DataFrame(features, columns=feature_names)

    output = pd.concat([
        df,
        df_features
    ], axis=1)

    store.put_subject_data(subject_id, output)


def run(input_path="output/instances.h5", group="default"):
    """Feature extraction from  Raw EEG files.

    Parameters:
    -----------

    input_path: string
        Path to hdf file

    group: string
        group of HDF
    """
    print("Reading from {} at group {}".format(input_path, group))

    with Store(input_path, group) as store:
        for subject_id in store.subject_ids:
            extract_features_for(store, subject_id)


if __name__ == '__main__':
    fire.Fire({
        "run": run
    })
