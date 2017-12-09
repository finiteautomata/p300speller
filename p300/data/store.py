"""Store class."""
import pandas as pd
import re
from .strategies import DifferentGroupStrategy

class Store:
    """Interface to storage."""
    def __init__(self, path, group, strategy_class=None):
        self.hdf = pd.HDFStore(path)
        strategy_class = strategy_class or DifferentGroupStrategy
        self.group = group
        self.strategy = strategy_class(self.hdf, group)

    @property
    def subject_ids(self):
        return strategy.subject_ids

    def clean(self):
        try:
            self.hdf.remove(self.group)
        except:
            pass

    def put_subject_data(self, subject_id, data, group=None):
        """Save subject data

        Parameters
        ----------

        subject_id: String
            Subject id

        data: pandas.DataFrame
            Data to be saved

        group: String
            Group to save the data in. If None, use default super group
        """
        return self.strategy.put_subject_data(subject_id, data, group=group)

    def get_subject_data(self, subject_id):
        """Get subject data

        Parameters
        ----------

        subject_id: String
            Subject id


        Returns
        -------

        data: pandas.DataFrame
            Data to be saved
        """

        return self.strategy.get_subject_data(subject_id)

    def get_subject_features(self, subject_id):
        """Get data from specific subject."""
        non_features = [
            'id', 'array_path', 'ch_names', 'event_time',
            'event_type', 'index', 'sfreq', 'subject_id', 'target'
        ]

        df = self.get_subject_data(subject_id)
        X = df[df.columns.difference(non_features)]
        y = df.target.as_matrix()

        return X.as_matrix(), y

    # TODO: Remove this stuff of asking subject id...
    def get_feature_names(self, subject_id):
        non_features = [
            'id', 'array_path', 'ch_names', 'event_time',
            'event_type', 'index', 'sfreq', 'subject_id', 'target'
        ]

        df = self.get_subject_data(subject_id)
        X = df[df.columns.difference(non_features)]
        return X.columns

    def put(self, *args, **kwargs):
        return self.hdf.put(*args, **kwargs)

    """ with's enter and exit function."""
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self.hdf.close()
