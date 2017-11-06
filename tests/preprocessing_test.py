from p300.feature_extraction import SubsamplingExtractor
import numpy as np
import unittest


class SubsamplingExtractorTest(unittest.TestCase):
    def test_transform_ones(self):
        extractor = SubsamplingExtractor(4)

        # Dimensions: n x m x k where
        # n = number of trials
        # m = number of channels
        # k = samples
        X = np.ones([10, 20])

        X_subsampled = extractor.transform(X)

        np.testing.assert_array_equal(X_subsampled, np.ones([10, 5]))

    def test_feature_names(self):
        extractor = SubsamplingExtractor(4)

        # Dimensions: n x m x k where
        # n = number of trials
        # m = number of channels
        # k = samples
        X = np.ones([10, 20])

        X_subsampled = extractor.transform(X)

        # 14 is for the fixed number of channels
        self.assertEqual(14 * 5, len(extractor.get_feature_names()))

    def test_one_dimensional_array(self):
        extractor = SubsamplingExtractor(2)

        # Dimensions: n x m x k where
        # n = number of trials
        # m = number of channels
        # k = samples
        X = np.ones([5, 6]).cumsum(axis=1)

        X_subsampled = extractor.transform(X)

        # 14 is for the fixed number of channels
        np.testing.assert_array_equal(
            X_subsampled,
            np.tile([1.5, 3.5, 5.5], (5, 1))
        )

if __name__ == '__main__':
    unittest.main()
