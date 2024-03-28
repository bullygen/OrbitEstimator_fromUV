import timeit
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

from pysofa_ctypes import *
from timeUtilities import *
from satelliteClass import Satellite
from groundTelescopeClass import GroundTelescope
from sourceClass import Source

R_EARTH_EQUATORIAL = 6378.1366
# ----------------------------------------------------------------------------


def isVisibleTel(tel, src, JD_UTC, JD_TT, JD_UT1, velocity):
    X_tel, Y_tel, Z_tel = tel.get_xyz_ICRF(JD_UTC, JD_TT, JD_UT1, velocity)
    X_src, Y_src, Z_src = src.get_XYZ()

    if (X_src * X_tel + Y_src * Y_tel + Z_src * Z_tel >= 0):
        return True
    else:
        return False


def isVisibleSat(sat, src, JD0, JD_UTC):
    time = (JD_UTC - JD0) * 86400
    acc = 1E-4
    X_sat, Y_sat, Z_sat = sat.get_XYZ(time, acc)
    X_src, Y_src, Z_src = src.get_XYZ()

    R_sat = np.sqrt(X_sat**2 + Y_sat**2 + Z_sat**2)

    angle_Earth_Source = np.arccos(
        (X_src * X_sat + Y_src * Y_sat + Z_src * Z_sat) / R_sat)
    if (R_EARTH_EQUATORIAL < R_sat):
        angle_Earth_Horizon = np.arcsin(R_EARTH_EQUATORIAL / R_sat)
    else:
        angle_Earth_Horizon = 90.

    if (angle_Earth_Horizon <= angle_Earth_Source):
        return True
    else:
        return False


def angularDistance(src, JD_UTC):
    PosVel_Heliocentric, PosVel_Baricentric = epv00(JD_UTC, 0)
    r_Sun = np.sqrt(np.sum(np.array(PosVel_Heliocentric[0, :])**2))
    X_sun = PosVel_Heliocentric[0, 0]
    Y_sun = PosVel_Heliocentric[0, 1]
    Z_sun = PosVel_Heliocentric[0, 2]
    X_src, Y_src, Z_src = src.get_XYZ()

    separation = np.arccos(
        (X_src * X_sun + Y_src * Y_sun + Z_src * Z_sun) / r_Sun)
    return separation


def UVWmaker(src, r_ICRF, JD):
    X = r_ICRF[0]
    Y = r_ICRF[1]
    Z = r_ICRF[2]

    RA = src.get_RA()
    DEC = src.get_DEC()

    U = -np.sin(RA) * X + np.cos(RA) * Y
    V = -np.cos(RA) * np.sin(DEC) * X - np.sin(RA) * \
        np.sin(DEC) * Y + np.cos(DEC) * Z
    W = np.cos(RA) * np.cos(DEC) * X + np.sin(RA) * \
        np.cos(DEC) * Y + np.sin(DEC) * Z

    return U, V, W


def getUVWarray(
        sat_array,
        tel_array,
        src,
        time_array,
        JD0,
        separation,
        velocity_array):
    N = len(time_array)
    time_array_JD_UTC = time_array / 86400. + JD0

    TAI_UTC, dUT_array, MJD0 = openBulletin()
    time_array_JD_TT = TT_from_UTC(time_array_JD_UTC, TAI_UTC)
    time_array_JD_UT1 = UT1_from_UTC(time_array_JD_UTC, dUT_array, MJD0)

    n1 = len(sat_array)
    n2 = len(tel_array)

    m = n1 + n2

    Nvec = int(m * (m - 1) / 2)

    U = np.zeros((N, Nvec))
    V = np.zeros((N, Nvec))
    W = np.zeros((N, Nvec))
    marker = np.zeros((N, Nvec))

    acc = 1E-5
    velocity = np.zeros(3)

    for i in range(N):
        # time utilities
        t = time_array[i]
        JD_UTC = time_array_JD_UTC[i]
        JD_TT = time_array_JD_TT[i]
        JD_UT1 = time_array_JD_UT1[i]

        RA_ = src.get_RA()
        DEC_ = src.get_DEC()
        RefFrame_ = src.get_RefFrame()
        src_corrected = Source(RA_, DEC_, RefFrame_)
        src_corrected.corrAberration(JD_UTC)

        # visibility check
        separation_ = angularDistance(src, JD_UTC)
        # print(separation_ * 180. / np.pi)

        # calculate XYZ for all objects
        X_tel = np.zeros(n2)
        Y_tel = np.zeros(n2)
        Z_tel = np.zeros(n2)
        cond_tel = np.zeros(n2, dtype=np.bool)

        X_sat = np.zeros(n1)
        Y_sat = np.zeros(n1)
        Z_sat = np.zeros(n1)
        cond_sat = np.zeros(n1, dtype=np.bool)

        for index_obj in range(n2):
            X_tel[index_obj], Y_tel[index_obj], Z_tel[index_obj] = tel_array[index_obj].get_xyz_ICRF(
                JD_UTC, JD_TT, JD_UT1, velocity)
            cond_tel[index_obj] = isVisibleTel(
                tel_array[index_obj], src, JD_UTC, JD_TT, JD_UT1, velocity)
            X_tel[index_obj], Y_tel[index_obj], Z_tel[index_obj] = UVWmaker(
                src_corrected, np.array([X_tel[index_obj], Y_tel[index_obj], Z_tel[index_obj]]), JD_UTC)

        for index_obj in range(n1):
            X_sat[index_obj], Y_sat[index_obj], Z_sat[index_obj] = sat_array[index_obj].get_XYZ(
                time_array[i], acc)
            cond_sat[index_obj] = isVisibleSat(
                sat_array[index_obj], src, JD0, JD_UTC)
            X_sat[index_obj], Y_sat[index_obj], Z_sat[index_obj] = UVWmaker(
                src_corrected, np.array([X_sat[index_obj], Y_sat[index_obj], Z_sat[index_obj]]), JD_UTC)

        # calculate UV for all objects
        for index_obj1 in range(1, m):
            for index_obj2 in range(index_obj1):

                index_vec = int(
                    index_obj1 * (index_obj1 - 1) / 2 + index_obj2 - 1)
                # print(index_obj1, index_obj2, index_vec)

                # define objects (sat-sat, tel-sat , tel-tel)
                if ((index_obj1 < n1) and (index_obj2 < n1)):
                    # sat-sat configuration
                    if (cond_sat[index_obj1] and (separation_ >
                                                  separation) and cond_sat[index_obj2]):
                        X_1, Y_1, Z_1 = X_sat[index_obj1], Y_sat[index_obj1], Z_sat[index_obj1]
                        X_2, Y_2, Z_2 = X_sat[index_obj2], Y_sat[index_obj2], Z_sat[index_obj2]
                        U[i, index_vec] = X_1 - X_2
                        V[i, index_vec] = Y_1 - Y_2
                        W[i, index_vec] = Z_1 - Z_2
                    else:
                        marker[i, index_vec] = 1

                elif ((index_obj1 >= n1) and (index_obj2 >= n1)):
                    # tel-tel configuration
                    if (cond_tel[index_obj1 - n1] and (separation_ >
                                                       separation) and cond_tel[index_obj2 - n1]):
                        X_1, Y_1, Z_1 = X_tel[index_obj1 -
                                              n1], Y_tel[index_obj1 -
                                                         n1], Z_tel[index_obj1 -
                                                                    n1]
                        X_2, Y_2, Z_2 = X_tel[index_obj2 -
                                              n1], Y_tel[index_obj2 -
                                                         n1], Z_tel[index_obj2 -
                                                                    n1]
                        U[i, index_vec] = X_1 - X_2
                        V[i, index_vec] = Y_1 - Y_2
                        W[i, index_vec] = Z_1 - Z_2
                    else:
                        marker[i, index_vec] = 1

                else:
                    # tel-sat configuration
                    index_tel = max(index_obj1, index_obj2) - n1
                    index_sat = min(index_obj1, index_obj2)
                    # print(i, N)
                    if (cond_sat[index_sat] and (separation_ >
                                                 separation) and cond_tel[index_tel]):
                        X_s, Y_s, Z_s = X_sat[index_sat], Y_sat[index_sat], Z_sat[index_sat]
                        X_t, Y_t, Z_t = X_tel[index_tel], Y_tel[index_tel], Z_tel[index_tel]
                        U[i, index_vec] = X_s - X_t
                        V[i, index_vec] = Y_s - Y_t
                        W[i, index_vec] = Z_s - Z_t
                    else:
                        marker[i, index_vec] = 1

    # delete impossible to observe time moments
    count1 = 0
    for i in range(N):
        for j in range(Nvec):
            if (marker[i, j] == 1):
                count1 += 1

    U_ = np.zeros(N * Nvec - count1)
    V_ = np.zeros(N * Nvec - count1)
    W_ = np.zeros(N * Nvec - count1)

    count2 = 0
    for i in range(N):
        for j in range(Nvec):
            if (marker[i, j] == 1):
                count2 += 1
            else:
                U_[i * Nvec + j - count2] = U[i, j]
                V_[i * Nvec + j - count2] = V[i, j]
                W_[i * Nvec + j - count2] = W[i, j]

    return U_, V_, W_


def getUVWarray2(sat, tel, src, time_array, JD0, separation, velocity):
    N = len(time_array)
    time_array_JD_UTC = time_array / 86400. + JD0

    TAI_UTC, dUT_array, MJD0 = openBulletin()
    time_array_JD_TT = TT_from_UTC(time_array_JD_UTC, TAI_UTC)
    time_array_JD_UT1 = UT1_from_UTC(time_array_JD_UTC, dUT_array, MJD0)

    U = np.zeros(N)
    V = np.zeros(N)
    W = np.zeros(N)
    marker = np.zeros(N)

    acc = 1E-5
    velocity = np.zeros(3)

    for i in range(N):
        # time utilities (later)

        t = time_array[i]
        JD_UTC = time_array_JD_UTC[i]
        JD_TT = time_array_JD_TT[i]
        JD_UT1 = time_array_JD_UT1[i]

        # visibility check
        separation_ = angularDistance(src, JD_UTC)

        # main
        if (isVisibleTel(tel, src, JD_UTC, JD_TT, JD_UT1, velocity) and (
                separation_ > separation) and isVisibleSat(sat, src, JD0, JD_UTC)):
            X_s, Y_s, Z_s = sat.get_XYZ(time_array[i], acc)
            X_t, Y_t, Z_t = tel.get_xyz_ICRF(JD_UTC, JD_TT, JD_UT1, velocity)
            r_ICRF = np.array([X_s - X_t, Y_s - Y_t, Z_s - Z_t])
            U[i], V[i], W[i] = UVWmaker(src, r_ICRF, time_array_JD_UTC[i])
        else:
            marker[i] = 1

    # delete impossible to observe time moments
    count1 = 0
    for i in range(N):
        if (marker[i] == 1):
            count1 += 1

    U_ = np.zeros(N - count1)
    V_ = np.zeros(N - count1)
    W_ = np.zeros(N - count1)

    count2 = 0
    for i in range(N):
        if (marker[i] == 1):
            count2 += 1
        else:
            U_[i - count2] = U[i]
            V_[i - count2] = V[i]
            W_[i - count2] = W[i]

    return U_, V_, W_


def makeRay(U, V, Npix, wavelength, maxSpatialValue):
    weigthFunction = np.zeros((Npix, Npix))
    N = len(U)
    k = 1.

    # wavelength in meters
    # angular freq in arcsec
    maxAngularValue = Npix / (2 * maxSpatialValue) * wavelength

    U = U / (2 * maxSpatialValue) * Npix * k + Npix / 2
    V = V / (2 * maxSpatialValue) * Npix * k + Npix / 2

    for i in range(N):
        if ((U[i] >= 0) and (U[i] < Npix) and (V[i] >= 0) and (V[i] < Npix)):
            indexU = int(U[i])
            indexV = int(V[i])
            weigthFunction[indexU, indexV] = 1
            weigthFunction[-indexU, -indexV] = 1

    weigthFunction_shifted = np.fft.fftshift(weigthFunction)
    ray_shifted = np.fft.fft2(weigthFunction_shifted).real
    ray = np.fft.fftshift(ray_shifted)

    return weigthFunction, ray, maxSpatialValue, maxAngularValue

# ----------------------------------------------------------------------------


def generateObjects():
    # general params
    JD0 = 2400000.5 + 60473  # test date = 12.06.2024
    time_array = np.linspace(0, 3 * 86400, 3000)  # in seconds
    separation = 45. * np.pi / 180.

    # satellite params
    a_ = 2e4
    e_ = 0.2
    inclination_ = 50. * np.pi / 180.
    node_ = 77. * np.pi / 180.
    pericenter_ = 30. * np.pi / 180.
    epoch_ = 0.
    lambda0_ = 0.

    # ground telescope params
    varphi_ = 60. * np.pi / 180.
    lambda_ = 0. * np.pi / 180.
    h_ = 0.
    velocity = np.zeros(3)

    # source params
    RA_ = 0. * np.pi / 180.
    DEC_ = 90. * np.pi / 180.
    RefFrame_ = 'J2000'

    sat = Satellite(a_, e_, inclination_, node_, pericenter_, epoch_, lambda0_)
    # tel = GroundTelescope(0, 0, 0, 'xyz')
    tel = GroundTelescope(varphi_, lambda_, h_, 'geography')
    src = Source(RA_, DEC_, RefFrame_)

    sat_array = [sat]
    tel_array = [tel]

    return sat_array, tel_array, src, time_array, JD0, separation, velocity


def generateObjectsEHI():
    # general params
    JD0 = 2400000.5 + 60473  # test date = 12.06.2024
    time_array = np.linspace(0, 30 * 86400, 3000)  # in seconds
    separation = 45. * np.pi / 180.
    velocity = np.zeros(3)

    # satellite 1 params
    a_ = 6400 + 7500
    e_ = 0.
    inclination_ = 90. * np.pi / 180.
    node_ = 137.1 * np.pi / 180.
    pericenter_ = 0. * np.pi / 180.
    epoch_ = 0.
    lambda0_ = 0.

    da = 30  # altitude difference [km]
    dM = 0.  # mean anomaly difference [rad]

    # source params (Sgr A*)
    RA_ = (17 + 45. / 60) * 15 * np.pi / 180.
    DEC_ = -29. * np.pi / 180.
    RefFrame_ = 'J2000'

    sat1 = Satellite(
        a_,
        e_,
        inclination_,
        node_,
        pericenter_,
        epoch_,
        lambda0_)
    sat2 = Satellite(
        a_ - da,
        e_,
        inclination_,
        node_,
        pericenter_,
        epoch_ - dM,
        lambda0_)
    src = Source(RA_, DEC_, RefFrame_)

    sat_array = [sat1, sat2]
    tel_array = []

    return sat_array, tel_array, src, time_array, JD0, separation, velocity

# ----------------------------------------------------------------------------


def testRayCalculations(
        sat_array,
        tel_array,
        src,
        time_array,
        JD0,
        separation,
        velocity,
        plotting,
        setscale):
    U, V, W = getUVWarray(sat_array, tel_array, src,
                          time_array, JD0, separation, velocity)

    # sat.graphingOrbit(time_array, isPerturbed = 0, isEarth = 1)

    Npix = 1024
    wavelength = 1.
    if (setscale == 0):
        if (len(U) != 0):
            maxValueU, maxValueV = np.max(np.abs(U)), np.max(np.abs(V))
            maxSpatialValue = maxValueU
            if (maxValueU < maxValueV):
                maxSpatialValue = maxValueV
        else:
            maxSpatialValue = R_EARTH_EQUATORIAL
    else:
        maxSpatialValue = setscale

    weigthFunction, ray, scaleSpatial, scaleAngular = makeRay(
        U, V, Npix, wavelength, maxSpatialValue)
    scaleSpatial = scaleSpatial / (2 * R_EARTH_EQUATORIAL)
    scaleAngular = scaleAngular * 206265

    if (np.max(ray) > 0):
        ray = ray / np.max(ray)

    if (np.max(ray) != 0):
        centerU, centerV = tuple(
            np.unravel_index(
                np.argmax(
                    ray, axis=None), ray.shape))
    else:
        centerU, centerV = int(Npix / 2), int(Npix / 2)
    frameSize = 32

    scaleAngular = scaleAngular / Npix * (2 * frameSize)

    ray_shifted = np.fft.fftshift(ray[centerU -
                                      frameSize: centerU +
                                      frameSize, centerV -
                                      frameSize: centerV +
                                      frameSize])
    reconstruction_shifted = np.fft.fft2(ray_shifted).real
    reconstruction = np.fft.fftshift(reconstruction_shifted)
    recostruction_scale = 206265 * frameSize * \
        wavelength / (scaleAngular * 2 * R_EARTH_EQUATORIAL)

    if (plotting == 1):
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        plt.figure

        axes[0, 0].imshow(weigthFunction, extent=[-scaleSpatial,
                          scaleSpatial, -scaleSpatial, scaleSpatial])
        axes[0, 0].set_xlabel(r'$u / D_\oplus$')
        axes[0, 0].set_ylabel(r'$v / D_\oplus$')

        axes[0,
             1].imshow(ray[centerU - frameSize: centerU + frameSize,
                           centerV - frameSize: centerV + frameSize],
                       extent=[-scaleAngular,
                               scaleAngular,
                               -scaleAngular,
                               scaleAngular])
        axes[0, 1].set_xlabel(r'$\theta_u,$' + ' mas')
        axes[0, 1].set_ylabel(r'$\theta_v,$' + ' mas')

        axes[1, 0].imshow(reconstruction, extent=[-recostruction_scale,
                          recostruction_scale, -recostruction_scale, recostruction_scale])
        axes[1, 0].set_xlabel(r'$u / D_\oplus$')
        axes[1, 0].set_ylabel(r'$v / D_\oplus$')

        plt.show()

    return sat_array, tel_array, src, reconstruction, recostruction_scale

# ----------------------------------------------------------------------------
# uv plane generator for test of fitting


def generateBlurredUV(filename):
    sat_array, tel_array, src, time_array, JD0, separation, velocity = generateObjects()
    sat_array, tel_array, src, reconstruction, recostruction_scale = testRayCalculations(
        sat_array, tel_array, src, time_array, JD0, separation, velocity, plotting=1, setscale=0)

    # timeframe:
    # observation time = 3sd
    # test date = 12.06.2024
    # to set time calc to 1.7s N = 3000 is needed

    n1 = len(sat_array)
    n2 = len(tel_array)

    # source
    RA_ = src.get_RA()
    DEC_ = src.get_DEC()

    # writing into file
    file = open('testBlurredUV/' + filename + '.dat', 'w')

    # satellite array
    for i in range(n1):
        sat = sat_array[i]
        a_ = sat.get_a()
        e_ = sat.get_e()
        inclination_ = sat.get_inclination()
        node_ = sat.get_node()
        pericenter_ = sat.get_pericenter()
        epoch_ = sat.get_epoch()
        file.write('Satellite params (to fit): \n')
        file.write('a = ' + str(a_) + ' km\n')
        file.write('e = ' + str(e_) + '\n')
        file.write('inclination = ' +
                   str(int(inclination_ * 180 / np.pi)) + ' deg\n')
        file.write('node = ' + str(int(node_ * 180 / np.pi)) + ' deg\n')
        file.write('pericenter = ' +
                   str(int(pericenter_ * 180 / np.pi)) + ' deg\n')
        file.write('epoch = ' + str(int(epoch_ * 180 / np.pi)) + ' deg\n\n')

    # ground telescope
    for i in range(n2):
        tel = tel_array[i]
        X_tel = tel.get_x()
        Y_tel = tel.get_y()
        Z_tel = tel.get_z()
        file.write('Telescope ITRF coordinates:\n')
        file.write('X = ' + str(X_tel) + ' km\n')
        file.write('Y = ' + str(Y_tel) + ' km\n')
        file.write('Z = ' + str(Z_tel) + ' km\n\n')

    file.write('Source J2000.0 coordinates:\n')
    file.write('RA = ' + str(int(RA_ * 180 / np.pi)) + ' rad\n')
    file.write('DEC = ' + str(int(DEC_ * 180 / np.pi)) + ' rad\n\n')

    file.write('Spatial scale (from center to edge) of the image below:\n')
    file.write(
        'scale = ' +
        str(recostruction_scale) +
        ' in Earth diameters\n\n')

    file.write('Number of dots calculated: ')
    file.write('N = ' + '3000' + ' for 3 days of observation\n')

    file.write('Blurred image:\n')
    N = len(reconstruction[0, :])
    for i in range(N):
        string = ''
        for j in range(N):
            string = string + ' ' + '{:.2f}'.format(reconstruction[i, j])
        file.write(string + '\n')

    file.close()


'''
pt1 = timeit.default_timer()
generateBlurredUV('test_initial')
pt2 = timeit.default_timer()
print(pt2 - pt1)
'''
