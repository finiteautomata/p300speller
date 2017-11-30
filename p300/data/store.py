"""Store class."""
import pandas as pd
import re

class Store:
    """Interface to storage."""
    def __init__(self, path, group):
        self.hdf = pd.HDFStore(path)
        self.group = group

    @property
    def subject_ids(self):
        regex = "^/{}/".format(self.group)

        def get_sid(key):
            return key.split("/")[-1].split("_")[1]

        return [
             get_sid(key) for key in self.hdf.keys()
             if re.match(regex, key)
        ]

    def clean(self):
        try:
            self.hdf.remove(self.group)
        except:
            pass

    def _group_for(self, subject_id, group=None):
        """Return group for given subject."""
        group = group or self.group
        return "{}/s_{}".format(group, subject_id)

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

        self.hdf.put(self._group_for(subject_id, group=group), data)

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

        return self.hdf.get(self._group_for(subject_id))

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
