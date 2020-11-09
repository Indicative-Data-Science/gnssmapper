"""Unittests for the Four Parameter Logistic Regression Model."""

import unittest
from source.classifier.FourParamLR import *


class TestFourParamLR(unittest.TestCase):

    def setUp(self) -> None:
        self.model = FourParamLogisticRegression()

        # Generating well separated data for classification
        good_x = [[val] for val in np.random.randint(100, size=100)]
        good_y = [0 for _ in range(100)]
        bad_y = [1 for _ in range(100)]
        bad_x = [[val1] for val1 in np.random.randint(100, 200, size=100)]
        self.train_x = good_x + bad_x
        self.train_y = good_y + bad_y

    def test_four_param_sigmoid(self):
        self.assertTrue(1 > self.model.four_param_sigmoid([10]) > 0)

    def test_fit(self):
        X_train, X_test, y_train, y_test = train_test_split(self.train_x, self.train_y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)

        # Checking that a and d have plausible values
        self.assertTrue(1 > self.model.a > 0)
        self.assertTrue(1 > self.model.d > 0)

        # Test perfect roc auc score for the test data
        self.assertEqual(roc_auc_score(y_test, self.model.predict(X_test)), 1.0)

    def test_mini_batch(self):
        # testing for the mini-batch case, NOTE: reduce batch-size for a faster test
        self.model = FourParamLogisticRegression(solver='mini-batch')
        X_train, X_test, y_train, y_test = train_test_split(self.train_x, self.train_y, test_size=0.2, random_state=42)
        self.model.fit(X_train, y_train)
        # Test perfect roc auc score for the test data
        self.assertEqual(roc_auc_score(y_test, self.model.predict(X_test)), 1.0)


if __name__ == '__main__':
    unittest.main()
