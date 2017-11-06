from p300.feature_extraction import SubsamplingExtractor
import numpy as np
import unittest


class SubsamplingExtractorTest(unittest.TestCase):
    def test_feature_names(self):
        extractor = SubsamplingExtractor(4)

        # Dimensions: n x m x k where
        # n = number of trials
        # m = number of channels
        # k = samples
        X = np.ones([10, 20, 12])

        X_subsampled = extractor.transform(X)

        np.testing.assert_array_equal(X_subsampled, np.ones([10, 60]))

if __name__ == '__main__':
    unittest.main()
