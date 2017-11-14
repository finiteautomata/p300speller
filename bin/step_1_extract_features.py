"""Step 0: Create instances from MNE-Raw Files."""
from p300.feature_extraction import LoadArray, SubsamplingExtractor
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

    feature_union = FeatureUnion(
        [('subsample_', SubsamplingExtractor(13))]
    )

    pipe = make_pipeline(
        load_array,
        feature_union
    )

    return pipe


def run(input_path="output/instances.h5", key="plain"):
    """Feature extraction from  Raw EEG files.

    Parameters:
    -----------

    input_path: string
        Path to hdf file

    key: string
        Key of HDF
    """
    print("Reading from {}".format(input_path))
    hdf = pd.HDFStore(input_path)

    # TODO: enhance this
    new_key = key + "_new"

    subject_ids = hdf.select(key, columns=["subject_id"]).subject_id.unique()

    for subject_id in subject_ids:
        print("Extracting features from {}".format(subject_id))

        df = hdf.select(key, where='subject_id = "{}"'.format(subject_id))
        pipe = create_extractor()
        features = pipe.fit_transform(df)
        feature_names = pipe.steps[-1][1].get_feature_names()

        df_features = pd.DataFrame(features, columns=feature_names)

        output = pd.concat([
            df,
            df_features
        ], axis=1)
        hdf.append(new_key, output, format='t', data_columns=["subject_id"])

    hdf[key] = hdf[new_key]
    hdf.remove(new_key)
    hdf.close()

if __name__ == '__main__':
    fire.Fire({
        "run": run
    })
