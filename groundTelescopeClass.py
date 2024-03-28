import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from pysofa_ctypes import *

# EARTH

R_EARTH_EQUATORIAL = 6378.1366	  # - (km)
GM_EARTH = 3.986004418E5		  	  # - (km 3 s -2)
J2_EARTH = 1.0826359E-3			  # - const for precession
W_EARTH = 7.292115E-5			  # - (rad 1 s -1)

# SUN

GM_SUN = 1.32712442099E11		  # - (km 3 s -2)
EPSILON = 84381.406 * np.pi / (180. * 3600)  # - (rad)
AU = 1.49597870700E8			  # - (km)
E_SUN = 0.0167086


class GroundTelescope():

    # coords should be in ITRF
    #
    # 1) dummy way to initialize - (phi, lambda)
    # 2) XYZ are the best way to transform
    #
    #
    def __init__(self, *args):
        if (args[3] == 'geography'):
            varphi_ = args[0]
            lambda_ = args[1]
            h = args[2]  # in km !!!

            self.x = (R_EARTH_EQUATORIAL + h) * \
                np.cos(varphi_) * np.cos(lambda_)
            self.y = (R_EARTH_EQUATORIAL + h) * \
                np.cos(varphi_) * np.sin(-lambda_)
            self.z = (R_EARTH_EQUATORIAL + h) * np.sin(varphi_)

        elif (args[3] == 'xyz'):
            x_ = args[0]
            y_ = args[1]
            z_ = args[2]

            self.x = x_
            self.y = y_
            self.z = z_

        else:
            print(
                'Give exact location, this is not a proper way to do it. Set the vector to zero')
            self.x = 0.
            self.y = 0.
            self.z = 0.

    def __del__(self):
        print('clearing GroundTelescope')

    def get_x(self):
        return self.x

    def get_y(self):
        return self.y

    def get_z(self):
        return self.z

    def set_x(self, x_):
        self.x = x_

    def set_y(self, y_):
        self.y = y_

    def set_z(self, z_):
        self.z = z_

    def updatePosition(self, velocity, JD_UTC):
        x_ = self.get_x()
        y_ = self.get_y()
        z_ = self.get_z()

        x_ = x_ + velocity[0] * (JD_UTC - 2451545.0) / 365.25
        y_ = y_ + velocity[1] * (JD_UTC - 2451545.0) / 365.25
        z_ = z_ + velocity[2] * (JD_UTC - 2451545.0) / 365.25

        self.set_x(x_)
        self.set_y(y_)
        self.set_z(z_)

    def get_xyz_ICRF(self, JD_UTC, JD_TT, JD_UT1, velocity):
        self.updatePosition(velocity, JD_UTC)
        r_ICRF = np.zeros(3)

        x_ = self.get_x()
        y_ = self.get_y()
        z_ = self.get_z()
        r_ITRF = np.array([x_, y_, z_])

        matrix = c2t00b(JD_TT, 0, JD_UT1, 0, 0, 0)
        matrix = np.array(matrix)

        r_ICRF = np.matmul(matrix, r_ITRF)

        return r_ICRF[0], r_ICRF[1], r_ICRF[2]
