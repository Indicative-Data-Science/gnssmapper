"""Unittests for the functions in svid_location, using the true data from 2020-02-11."""

import unittest
from source.simulation.svid_location import *


class TestSVIDLocation(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.sp3 = getSP3File('2020-02-11')
        cls.sp3_file = getOrbits(cls.sp3)
        cls.pre_ = pd.read_csv('test_measurements')
        cls.post_ = pd.read_csv('test_processed_measurements')

    def test_dateToGPS(self) -> None:
        self.assertIsInstance(dateToGPS('2020-02-11'), dict)
        self.assertEquals(dateToGPS('2020-02-11'), {'week': 2092, 'day': 2})

    def test_getSP3File(self) -> None:
        self.esm_path = 'esm' + str(dateToGPS('2020-02-11')['week']) + str(dateToGPS('2020-02-11')['day']) + '.sp3.gz'
        self.assertTrue(os.path.exists(self.esm_path))

    def test_getOrbits(self) -> None:
        self.assertIsInstance(self.sp3_file, pd.DataFrame)
        self.assertEqual(sorted(self.sp3_file.columns),
                         sorted(['Epoch', 'UTC Time', 'svid', 'x', 'y', 'z', 'clockError']))
        self.assertEqual(self.sp3_file['UTC Time'].dtype, np.dtype('datetime64[ns]'))

    def test_getSVIDLocation(self) -> None:
        self.loc_func = getSVIDLocation(self.sp3_file)
        for row in range(len(self.pre_)):
            print(self.pre_.iloc[row, -1], type(self.pre_.iloc[row, -1]))
            if np.isnan(self.pre_.iloc[row, -1]):
                estimate = True
            else:
                estimate = False

            response = self.loc_func(estimate, self.pre_.iloc[row, -1], self.pre_.iloc[row, -2], self.pre_.iloc[row, 1])
            for col in self.post_:
                if col in response:
                    if np.isnan(self.post_[col][row]):
                        self.assertEqual(None if np.isnan(self.post_[col][row]) else False, response[col])
                    else:
                        self.assertEqual(round(self.post_[col][row], -4), round(response[col], -4))

    @classmethod
    def tearDownClass(cls):
        os.remove('esm20922.sp3.gz')


if __name__ == '__main__':
    unittest.main()





