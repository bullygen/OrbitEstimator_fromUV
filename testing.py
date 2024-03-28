from timeUtilities import *
from pysofa_ctypes import *

from satelliteClass import Satellite
from groundTelescopeClass import GroundTelescope
from sourceClass import Source

from UVplaneMaker import *
from UVplaneGuessor import *

import unittest


class TimeUtilitiesMethods(unittest.TestCase):

    def test_openBulletin(self):
        TAI_UTC, dUT_array, MJD0 = openBulletin()
        self.assertEqual(TAI_UTC, 37.0)

    def test_TT_from_UTC(self):
        self.assertEqual(TT_from_UTC(0, 0), 32.184)

    def test_UT1_from_UTC(self):
        self.assertEqual(UT1_from_UTC([0], [0], 0), [0])


class SatelliteClassMethods(unittest.TestCase):

    def test_get_period(self):
        GM_EARTH = 3.986004418E5
        a_ = 1e4
        sat = Satellite(a_, 0, 0, 0, 0, 0, 0)
        T = 2 * np.pi / np.sqrt(GM_EARTH / a_**3)
        self.assertEqual(sat.get_period(), T)

    def test_get_XYZ(self):
        a_ = 1e4
        e_ = 0.1
        sat = Satellite(a_, e_, 0, 0, 0, 0, 0)
        time = 0
        acc = 1e-6
        X, Y, Z = sat.get_XYZ(time, acc)
        self.assertEqual(int(X), int(a_ * (1 - e_)))


class SourceClassMethods(unittest.TestCase):

    def test_get_XYZ_1(self):
        src = Source(1, 0, 0, 'ITRF')
        X, Y, Z = src.get_XYZ()
        self.assertEqual(X, 1)

    def test_get_XYZ_2(self):
        src = Source(0, 0, 'ITRF')
        X, Y, Z = src.get_XYZ()
        self.assertEqual(Y, 1)


# test of UV-plane generation
sat_array, tel_array, src, time_array, JD0, separation, velocity = generateObjects()
sat_array, tel_array, src, reconstruction, recostruction_scale = testRayCalculations(
    sat_array, tel_array, src, time_array, JD0, separation, velocity, plotting=1, setscale=0)


# test of satellite picture
sat = sat_array[0]
sat.graphingOrbit(time_array, isPerturbed=0, isEarth=1)

# perform unit tests
if __name__ == '__main__':
    unittest.main()
