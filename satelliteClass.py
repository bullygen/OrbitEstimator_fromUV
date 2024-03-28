import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

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

# MOON POSITIONS FROM EPHEUMERIDS

# gets 		M (rad), e [0, 1), acc (float >0)
# returns 	E (rad)


def solve_cubic(a, c, d):
    p = c / a
    q = d / a
    k = np.sqrt(q**2 / 4 + p**3 / 27)
    return np.real(np.cbrt(-q / 2 - k) + np.cbrt(-q / 2 + k))


def machin(e, M):
    n = np.sqrt(5 + np.sqrt(16 + 9 / e))
    a = n * (e * (n**2 - 1) + 1) / 6.
    c = n * (1 - e)
    d = -M
    s = solve_cubic(a, c, d)
    return n * np.arcsin(s)


def keplerSolver(M, e, acc):
    cnt = 0
    if (e != 0):
        acc_current = 1.
        sign_ = 1
        if (M > np.pi):
            M = -(M - 2 * np.pi)
            sign_ = -1

        if (e > 0.60 and np.abs(M) < 0.42):
            E_old = machin(e, M)
        else:
            E_old = M + 0.85 * e

        E_new = E_old
        while (acc_current > acc):
            cnt += 1
            E_old = E_new
            denom = (E_old - 2 * (M + e * np.sin(E_old)) +
                     M + e * np.sin(M + e * np.sin(E_old)))
            # catching denom zero - using Newton instead
            if (denom != 0):
                E_new = E_old - (M + e * np.sin(E_old) - E_old)**2 / denom
            else:
                E_new = E_old - (E_old - e * np.sin(E_old) -
                                 M) / (1 - e * np.cos(E_old))
            acc_current = np.abs(E_old - E_new)
    else:
        E_new = M
    # print(cnt)
    return E_new * sign_

# lunar perturbance (ephem?)


def extractLunarCoordinates():
    X = 0
    Y = 0
    Z = 0

    return X, Y, Z

# solar perturbance (e = 0)
# date0 - since equinox in sec
# time - in sec


def extractSolarCoordinates(date0, time):
    lambda_ = 2 * np.pi * (date0 + time)
    EOT = 60. * (7.53 * np.cos(lambda_) + 1.5 *
                 np.sin(lambda_) - 9.87 * np.sin(2 * lambda_))

    alpha = lambda_ + EOT
    delta = EPSILON * np.sin(lambda_)

    X = AU * np.cos(delta) * np.cos(alpha)
    Y = AU * np.cos(delta) * np.sin(alpha)
    Z = AU * np.sin(delta)

    return X, Y, Z

# some thing to do gravity right


def extractEarthParameters():
    return 0

# everything here in ICRF, no coordinate transformations.


class Satellite():

    # initiation of orbital parameters
    # a 								- (km)
    # e 								- [0, 1)
    # inclination 						- [0, pi] (rad)
    # node 								- [0, 2pi) (rad)
    # pericenter 						- [0, 2pi) (rad)
    # epoch - mean anomaly at t=0 		- [0, 2pi) (rad)
    # lambda0 - longtitude under (t=0)  - [0, 2pi) (rad)

    def __init__(
            self,
            a_,
            e_,
            inclination_,
            node_,
            pericenter_,
            epoch_,
            lambda0_):
        self.a = a_
        self.e = e_
        self.inclination = inclination_
        self.node = node_
        self.pericenter = pericenter_
        self.epoch = epoch_
        self.lambda0 = lambda0_

    def __del__(self):
        print('clearing Satellite')
    # getters

    def get_a(self):
        return self.a

    def get_e(self):
        return self.e

    def get_inclination(self):
        return self.inclination

    def get_node(self):
        return self.node

    def get_pericenter(self):
        return self.pericenter

    def get_epoch(self):
        return self.epoch

    def get_lambda0(self):
        return self.lambda0

    # setters

    def set_a(self, a_):
        self.a = a_

    def set_e(self, e_):
        self.e = e_

    def set_inclination(self, inclination_):
        self.inclination = inclination_

    def set_node(self, node_):
        self.node = node_

    def set_pericenter(self, pericenter_):
        self.pericenter = pericenter_

    def set_epoch(self, epoch_):
        self.epoch = epoch_

    def set_epoch(self, lambda0_):
        self.lambda0 = lambda0_

    # calculations

    def get_period(self):
        a_ = self.get_a()
        T = 2 * np.pi / np.sqrt(GM_EARTH / a_**3)
        return T

    # acc - for angles (in rad)
    def get_XYZ(self, time, acc):
        a_ = self.get_a()
        e_ = self.get_e()
        M0 = self.get_epoch()
        T = self.get_period()

        M = (M0 + 2 * np.pi * (time / T)) % (2 * np.pi)
        E = keplerSolver(M, e_, acc)

        eta = a_ * np.sqrt(1. - e_**2) * np.sin(E)
        xi = a_ * (np.cos(E) - e_)

        pericenter_ = self.get_pericenter()
        node_ = self.get_node()
        inclination_ = self.get_inclination()

        Px = np.cos(pericenter_) * np.cos(node_) - \
            np.sin(pericenter_) * np.sin(node_) * np.cos(inclination_)
        Py = np.cos(pericenter_) * np.sin(node_) + \
            np.sin(pericenter_) * np.cos(node_) * np.cos(inclination_)
        Pz = np.sin(pericenter_) * np.sin(inclination_)
        Qx = -np.sin(pericenter_) * np.cos(node_) - \
            np.cos(pericenter_) * np.sin(node_) * np.cos(inclination_)
        Qy = -np.sin(pericenter_) * np.sin(node_) + \
            np.cos(pericenter_) * np.cos(node_) * np.cos(inclination_)
        Qz = np.cos(pericenter_) * np.sin(inclination_)

        X = Px * xi + Qx * eta
        Y = Py * xi + Qy * eta
        Z = Pz * xi + Qz * eta
        return X, Y, Z

    # only J2 disturbance
    # add Moon and Sun!!!

    def include_J2linearDisturbance(self, time):
        T = self.get_period()
        a_ = self.get_a()
        e_ = self.get_e()
        inclination_ = self.get_inclination()

        n = 2 * np.pi / T

        nu1 = 0.75 * J2_EARTH * (R_EARTH_EQUATORIAL / a_)**2 * \
            (2 - 3 * np.sin(inclination_)**2) / (1 - e_**2)**(1.5)
        nu2 = 0.75 * J2_EARTH * (R_EARTH_EQUATORIAL / a_)**2 * \
            (4 - 5 * np.sin(inclination_)**2) / (1 - e_**2)**(2)
        nu3 = -1.5 * J2_EARTH * (R_EARTH_EQUATORIAL / a_)**2 * \
            np.cos(inclination_) / (1 - e_**2)**(2)

        delta_epoch = time * n * nu1
        delta_pericenter = time * n * nu2
        delta_node = time * n * nu3

        delta_a = a_ * 0.75 * J2_EARTH * \
            (R_EARTH_EQUATORIAL / a_)**2 * (2 - 3 * np.sin(inclination_)**2)

        a_ = a_ + delta_a
        self.set_a(a_)

        epoch_ = self.get_epoch()
        epoch_ = epoch_ + delta_epoch
        self.set_epoch(epoch_ + delta_epoch)

        pericenter_ = self.get_pericenter()
        pericenter_ = pericenter_ + delta_pericenter
        self.set_pericenter(pericenter_ + delta_pericenter)

        node_ = self.get_node()
        node_ = node_ + delta_node
        self.set_node(node_ + delta_node)

    def include_SUNlinearDisturbance(self, time):
        T = self.get_period()
        a_ = self.get_a()
        e_ = self.get_e()
        inclination_ = self.get_inclination()

        n = 2 * np.pi / T

        nu2 = 0.375 * (GM_SUN / GM_EARTH) * (a_ / AU)**3 * (5 * np.sin(inclination_)**2 - e_ **
                                                            2 - 4) * (1.5 * np.sin(EPSILON)**2 - 1) / (1 - e_**2)**(0.5) / (1 - E_SUN**2)**(1.5)
        nu3 = 0.75 * (GM_SUN / GM_EARTH) * (a_ / AU)**3 * (1 + 1.5 * e_**2) * np.cos(inclination_) * \
            (1.5 * np.sin(EPSILON)**2 - 1) / (1 - e_**2)**(0.5) / (1 - E_SUN**2)**(1.5)

        delta_pericenter = time * n * nu2
        delta_node = time * n * nu3

        pericenter_ = self.get_pericenter()
        pericenter_ = pericenter_ + delta_pericenter
        self.set_pericenter(pericenter_ + delta_pericenter)

        node_ = self.get_node()
        node_ = node_ + delta_node
        self.set_node(node_ + delta_node)

    def include_MOONlinearDisturbance(self, time):
        return 0

    def calculateUnderPoint(self, time):
        acc = 1E-4
        lambda0_ = self.get_lambda0()
        X, Y, Z = self.get_XYZ(time, acc)

        lambda_ = (lambda0_ - np.arctan2(X, Y) + W_EARTH * time) % (2 * np.pi)
        if (lambda_ > np.pi):
            lambda_ -= 2 * np.pi
        varphi_ = np.arctan(Z / np.sqrt(X**2 + Y**2))
        return lambda_, varphi_

    def includePeriodicCorrections():
        return 0

    # neural network analysis later

    def includeAtmosphericDrag():
        return 0

    def includeRadiationPressure():
        return 0

    # including drag from atmosphere! Orbitron-like system

    def composeFromTelemetry(file_):
        return 0

    # graphics (using matplotlib)

    def graphingOrbit(self, time_array, isPerturbed, isEarth):
        N = len(time_array)
        X = np.zeros((N),)
        Y = np.zeros((N),)
        Z = np.zeros((N),)
        acc = 1E-4

        for i in range(N):
            if ((isPerturbed == 1) and (i > 0)):
                self.include_J2linearDisturbance(
                    time_array[i] - time_array[i - 1])
                self.include_SUNlinearDisturbance(
                    time_array[i] - time_array[i - 1])
                # self.include_MOONlinearDisturbance(self, time)
            X[i], Y[i], Z[i] = self.get_XYZ(time_array[i], acc)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # EARTH
        if (isEarth == 1):
            u = np.linspace(0, 2 * np.pi, 100)
            v = np.linspace(0, np.pi, 100)
            x_E = R_EARTH_EQUATORIAL * np.outer(np.cos(u), np.sin(v))
            y_E = R_EARTH_EQUATORIAL * np.outer(np.sin(u), np.sin(v))
            z_E = R_EARTH_EQUATORIAL * np.outer(np.ones(np.size(u)), np.cos(v))

            ax.plot_surface(x_E, y_E, z_E)

        # SATELLITE

        ax.plot(X, Y, Z, 'r')

        # BOX for equality
        a_ = self.get_a()
        e_ = self.get_e()
        a_ = a_ * (1. + e_)
        ax.plot([a_, a_, a_, a_, -a_, -a_, -a_, -a_], [a_, a_, -a_, -a_,
                a_, a_, -a_, -a_], [a_, -a_, a_, -a_, a_, -a_, a_, -a_], 'wo')

        ax.set_box_aspect([1, 1, 1])
        plt.show()

    def graphingUnderpoint(self, time_array):
        N = len(time_array)
        varphi_ = np.zeros((N),)
        lambda_ = np.zeros((N),)

        for i in range(N):
            lambda_[i], varphi_[i] = self.calculateUnderPoint(time_array[i])
            lambda_[i] = lambda_[i] * 180. / np.pi
            varphi_[i] = varphi_[i] * 180. / np.pi

        plt.plot([180, 180, -180, 180], [90, -90, 90, -90], 'wo', markersize=1)
        plt.plot(lambda_, varphi_, 'ro', markersize=1)

        plt.xlabel(r'$\lambda,^\circ$')
        plt.ylabel(r'$\varphi,^\circ$')
        plt.grid()
        plt.show()
