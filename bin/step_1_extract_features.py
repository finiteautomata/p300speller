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


def create_instances(input_path="output/instances.h5"):
    """Feature extraction from  Raw EEG files.

    Parameters:
    -----------

    input_csv: string
        Path to CSV of instances

    output_csv: string
        Where to output
    """
    print("Reading from {}".format(input_path))
    hdf = pd.HDFStore(input_path)

    for key in hdf.keys():
        print("Extracting features from {}".format(key))

        df = hdf.get(key)
        pipe = create_extractor()
        features = pipe.fit_transform(df)
        feature_names = pipe.steps[-1][1].get_feature_names()

        df_features = pd.DataFrame(features, columns=feature_names)

        output = pd.concat([
            df,
            df_features
        ], axis=1)

        hdf.put(key, output, format='t')
    hdf.close()

if __name__ == '__main__':
    fire.Fire(create_instances)
