import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from pysofa_ctypes import *

# EARTH

R_EARTH_EQUATORIAL = 6378.1366	  # - (km)
GM_EARTH = 3.986004418E5		  # - (km 3 s -2)
J2_EARTH = 1.0826359E-3			  # - const for precession
W_EARTH = 7.292115E-5			  # - (rad 1 s -1)

# SUN

GM_SUN = 1.32712442099E11		  # - (km 3 s -2)
EPSILON = 84381.406 * np.pi / (180. * 3600)  # - (rad)
AU = 1.49597870700E8			  # - (km)
E_SUN = 0.0167086

# from au/d to v/c
L = 5.787037037E-03


def findTransorm(RF_new, RF_old, JD_TT, JD_UT1):
    if ((RF_new == 'J2000') and (RF_old == 'ITRF')):
        result = findTransorm(RF_new='ICRF', RF_old='ITRF')

    elif ((RF_new == 'J2000') and (RF_old == 'ICRF')):
        result = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]])

    elif ((RF_new == 'ITRF') and (RF_old == 'ICRF')):
        result = np.array(c2t00b(JD_TT, 0, JD_UT1, 0, 0, 0))

    elif ((RF_old == 'J2000') and (RF_new == 'ITRF')):
        result = findTransorm(RF_new='ITRF', RF_old='ICRF')

    elif ((RF_old == 'J2000') and (RF_new == 'ICRF')):
        result = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]])

    elif ((RF_old == 'ITRF') and (RF_new == 'ICRF')):
        result = np.linalg.inv(np.array(c2t00b(JD_TT, 0, JD_UT1, 0, 0, 0)))

    else:
        print('Incorrect reference frames. Available options are: J2000, ITRF, ICRF. Transformation is set to unitary matrix')
        result = np.array([[1., 0., 0.],
                           [0., 1., 0.],
                           [0., 0., 1.]])

    return result


class Source():

    # RA - (rad)
    # DEC - (rad)
    # RefFrame - string (e.g. ITRF, ICRF, J2000)
    #
    #
    def __init__(self, *args):
        if (len(args) == 4):
            X = args[0]
            Y = args[1]
            Z = args[2]
            self.RefFrame = args[3]

            R = np.sqrt(X**2 + Y**2 + Z**2)
            self.DEC = np.arcsin(Z / R)
            self.RA = np.arctan2(X, Y)

        elif (len(args) == 3):
            self.RA = args[0]
            self.DEC = args[1]
            self.RefFrame = args[2]

        else:
            self.RA = 0.
            self.DEC = 0.
            self.RefFrame = 'J2000'

    # def __del__(self):
    #    print('clearing Source')

    def get_RA(self):
        return self.RA

    def get_DEC(self):
        return self.DEC

    def get_RefFrame(self):
        return self.RefFrame

    def set_RA(self, RA_):
        self.RA = RA_

    def set_DEC(self, DEC_):
        self.DEC = DEC_

    def set_RefFrame(self, RF_):
        self.RefFrame = RF_

    def get_XYZ(self):
        DEC_ = self.get_DEC()
        RA_ = self.get_RA()

        X = np.cos(DEC_) * np.sin(RA_)
        Y = np.cos(DEC_) * np.cos(RA_)
        Z = np.sin(DEC_)

        return X, Y, Z

    def set_FromXYZ(self, X_, Y_, Z_):
        R_ = np.sqrt(X_**2 + Y_**2 + Z_**2)
        DEC_ = np.arcsin(Z_ / R_)
        RA_ = np.arctan2(Y_, X_)

        self.set_RA(RA_)
        self.set_DEC(DEC_)

    # UT1, TT are optional.
    def transform_to(self, RF_new, JD_TT, JD_UT1):
        RF_old = self.get_RefFrame()
        X, Y, Z = self.get_XYZ()

        if (RF_new == RF_old):
            print('Reference Frame is the same, no transformation performed')
        else:
            matrix = findTransorm(RF_new, RF_old, JD_TT, JD_UT1)
            r_old = np.array([X, Y, Z])
            r_new = np.matmul(matrix, r_old)

            X_ = r_new[0]
            Y_ = r_new[1]
            Z_ = r_new[2]

            self.set_FromXYZ(X_, Y_, Z_)
            self.set_RefFrame(RF_new)

    def corrAberration(self, JD_UTC):
        X_, Y_, Z_ = self.get_XYZ()
        r_initial = np.array([X_, Y_, Z_])
        r_initial = np.reshape(r_initial, (1, 3))

        PosVel_Heliocentric, PosVel_Baricentric = epv00(JD_UTC, 0)

        r_Sun = np.sqrt(np.sum(np.array(PosVel_Heliocentric[0, :])**2))
        lorentz_factor = np.sqrt(
            1. - L**2 * np.sum(np.array(PosVel_Baricentric[1, :])**2))

        r_corrected = Ab(
            r_initial, PosVel_Baricentric[1, :] * L, r_Sun, lorentz_factor)

        self.set_FromXYZ(r_corrected[0, 0],
                         r_corrected[0, 1], r_corrected[0, 2])
