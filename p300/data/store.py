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

    def _group_for(self, subject_id):
        """Return group for given subject."""
        return "{}/s_{}".format(self.group, subject_id)

    def put_subject_data(self, subject_id, data):
        """Save subject data

        Parameters
        ----------

        subject_id: String
            Subject id

        data: pandas.DataFrame
            Data to be saved
        """

        self.hdf.put(self._group_for(subject_id), data)

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


    """ with's enter and exit function."""
    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return self.hdf.close()