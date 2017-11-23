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
            return key.split("_")[1]

        return [
             get_sid(key) for key in self.hdf.keys()
             if re.match(regex, key)
        ]

    def clean(self):
        try:
            self.hdf.remove(self.group)
        except:
            pass

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self.hdf.close()

    def put_subject_data(self, subject_id, data):
        """Save subject data

        Parameters
        ----------

        subject_id: String
            Subject id

        data: pandas.DataFrame
            Data to be saved
        """
        key = "{}/s_{}".format(self.group, subject_id)
        self.hdf.put(key, data)
