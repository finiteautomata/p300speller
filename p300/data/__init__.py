"""Data handling package."""


non_features = [
    'id', 'array_path', 'ch_names', 'event_time', 'event_type', 'index',
    'sfreq', 'subject_id', 'target']


def get_subject_ids(hdf, data_key):
    return hdf.select(data_key, columns=["subject_id"]).subject_id.unique()


def get_data_for(hdf, data_key, subject_id):
    return hdf.select(data_key, where='subject_id = "{}"'.format(subject_id))


def get_features_for(hdf, data_key, subject_id):
    """Get data from specific subject."""
    df = get_data_from(hdf, data_key, subject_id)
    X = df[df.columns.difference(non_features)]
    y = df.target.as_matrix()

    return X.as_matrix(), y, X.columns
