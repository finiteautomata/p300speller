import re

class DifferentGroupStrategy:
    """Different Group for each subject."""

    def __init__(self, hdf, group):
        self.hdf = hdf
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
