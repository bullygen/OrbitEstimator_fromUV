import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import timeit
from mpl_toolkits.mplot3d import axes3d
from scipy.interpolate import CubicSpline
from scipy.optimize import Bounds

from pysofa_ctypes import *
from timeUtilities import *
from UVplaneMaker import *
from satelliteClass import Satellite
from groundTelescopeClass import GroundTelescope
from sourceClass import Source
from radioastron_utils import AMS_to_rad, get_RA_test_params

from scipy.optimize import minimize

N = 64


def readPicture(filename):
    file = open(filename, 'r')
    count = 0
    picture = np.zeros((N, N))

    for line in file:
        words = line.strip().split()
        count += 1
        if (count > 25):
            for i in range(N):
                picture[count - 26, i] = float(words[i])
    file.close()
    return picture


def readConstantParams(filename):
    file = open(filename, 'r')
    count = 0
    RA_ = 0
    DEC_ = 0
    X_tel = 0
    Y_tel = 0
    Z_tel = 0
    for line in file:
        words = line.strip().split()
        count += 1
        if (count == 10):
            X_tel = float(words[2])
        if (count == 11):
            Y_tel = float(words[2])
        if (count == 12):
            Z_tel = float(words[2])
        if (count == 15):
            RA_ = float(words[2]) * np.pi / 180.
        if (count == 16):
            DEC_ = float(words[2]) * np.pi / 180.
        if (count == 19):
            scale = float(words[2])

    file.close()
    constants = RA_, DEC_, X_tel, Y_tel, Z_tel, scale
    return constants


def readRA_file(filename):
    file = open(filename, 'r')
    count = 0
    scale = 0.
    picture = np.zeros((N, N))
    for line in file:
        words = line.strip().split()
        if (count == 1):
            scale = float(words[2])
        if (count > 4):
            for i in range(N):
                picture[count - 5, i] = float(words[i])
        count += 1
    RA, DEC, phi, lmbda = get_RA_test_params()
    file.close()
    constants = RA, DEC, phi, lmbda, scale
    return picture, constants


def readTrueSatellite(filename):
    file = open(filename, 'r')
    count = 0
    a_ = 0
    e_ = 0
    inclination_ = 0
    node_ = 0
    pericenter_ = 0
    epoch_ = 0
    for line in file:
        words = line.strip().split()
        count += 1
        if (count == 2):
            a_ = float(words[2])
        if (count == 3):
            e_ = float(words[2])
        if (count == 4):
            inclination_ = float(words[2]) * np.pi / 180.
        if (count == 5):
            node_ = float(words[2]) * np.pi / 180.
        if (count == 6):
            pericenter_ = float(words[2]) * np.pi / 180.
        if (count == 7):
            epoch_ = float(words[2])

    file.close()
    constants = a_, e_, inclination_, node_, pericenter_, epoch_
    return constants


ifRA = 1
filename = 'testBlurredUV/RA.dat'
if (ifRA == 0):
    picture = readPicture(filename)
    constants = readConstantParams(filename)
else:
    picture, constants = readRA_file(filename)


def kepler_to_khol(params):
    a_, e_, inclination_, node_, pericenter_, epoch_ = params

    sqrtP = np.sqrt(a_ * (1 - e_**2) / R_EARTH_EQUATORIAL)

    U_x = sqrtP * np.sin(inclination_) * np.sin(node_)
    U_y = - sqrtP * np.sin(inclination_) * np.cos(node_)
    U_z = sqrtP * np.cos(inclination_)

    V_x = e_ * sqrtP * (np.cos(pericenter_) * np.cos(node_) -
                        np.cos(inclination_) * np.sin(pericenter_) * np.sin(node_))
    V_y = e_ * sqrtP * (np.cos(pericenter_) * np.sin(node_) +
                        np.cos(inclination_) * np.sin(pericenter_) * np.cos(node_))
    V_z = e_ * sqrtP * np.sin(inclination_) * np.sin(pericenter_)

    V_1 = V_x * np.sin(node_) * np.cos(inclination_) - V_y * \
        np.cos(node_) * np.cos(inclination_) - V_z * np.sin(inclination_)
    V_2 = V_x * np.cos(node_) + V_y * np.sin(node_)
    # V_3 = V_x * np.sin(inclination_) * np.sin(node_) - V_y * np.sin(inclination_) * np.cos(node_) + V_z * np.cos(inclination_)

    new_params = U_x, U_y, U_z, V_1, V_2, epoch_
    return new_params


def khol_to_kepler(params):
    U_x, U_y, U_z, V_1, V_2, epoch_ = params

    U = np.sqrt(U_x**2 + U_y**2 + U_z**2)

    inclination_ = np.arccos(U_z / U)
    node_ = np.arctan2(U_x, -U_y)

    V = np.sqrt(V_1**2 + V_2**2)
    V_x = V_1 * np.cos(inclination_) * np.sin(node_) - V_2 * np.cos(node_)
    V_y = - V_1 * np.cos(inclination_) * np.cos(node_) + V_2 * np.sin(node_)
    V_z = - V_1 * np.sin(inclination_)

    sinPeri = V_z / (V * np.sin(inclination_))
    cosPeri = (V_x * np.cos(node_) + V_y * np.sin(node_)) / V
    pericenter_ = np.arctan2(sinPeri, cosPeri)

    e_ = V / U
    a_ = U**2 / (1 - e_**2) * R_EARTH_EQUATORIAL

    new_params = a_, e_, inclination_, node_, pericenter_, epoch_
    return new_params


def measureOfSimilarity(params):
    if (ifRA == 0):
        RA_, DEC_, X_tel, Y_tel, Z_tel, scale = constants
        lambda0_ = 0.
        a_, e_, inclination_, node_, pericenter_, epoch_ = params[
            0], params[1], params[2], params[3], params[4], params[5]

        sat = Satellite(
            a_,
            e_,
            inclination_,
            node_,
            pericenter_,
            epoch_,
            lambda0_)
        tel = GroundTelescope(X_tel, Y_tel, Z_tel, 'xyz')
        sat_array = [sat]
        tel_array = [tel]
        src = Source(RA_, DEC_, 'J2000')

        velocity = np.zeros(3)
        JD0 = 2400000.5 + 60473
        time_array = np.linspace(0, 3 * 86400, 3000)
        separation = 45. * np.pi / 180.
        Npix = 1024
        wavelength = 1.

    else:
        RA, DEC, phi, lmbda, scale = constants
        lambda0_ = 0.
        a_, h_min, inclination_, node_, pericenter_, epoch_ = params[
            0], params[1], params[2], params[3], params[4], params[5]
        e_ = 1 - (h_min + R_EARTH_EQUATORIAL) / a_

        sat = Satellite(
            a_,
            e_,
            inclination_,
            node_,
            pericenter_,
            epoch_,
            lambda0_)
        src = Source(RA, DEC, 'J2000')
        sat_array = [sat]
        tel_array = [
            GroundTelescope(
                phi[i],
                lmbda[i],
                0.,
                'geography') for i in range(
                len(phi))]

        velocity = np.zeros(3)
        JD0 = 2400000.5 + 60473
        time_array = np.linspace(0, 64801, 3000)
        separation = 0. * np.pi / 180.
        Npix = 1024
        wavelength = 1.36e-2

    start2 = timeit.default_timer()

    U, V, W = getUVWarray(sat_array, tel_array, src,
                          time_array, JD0, separation, velocity)

    stop2 = timeit.default_timer()
    print('Time to generate UV: ', stop2 - start2)

    # generate pixelized picture here
    weigthFunction, ray, scaleSpatial, scaleAngular = makeRay(
        U, V, Npix, wavelength, scale * 2 * R_EARTH_EQUATORIAL)
    ray = ray / np.max(ray)
    centerU, centerV = tuple(
        np.unravel_index(
            np.argmax(
                ray, axis=None), ray.shape))
    frameSize = 32

    ray_shifted = np.fft.fftshift(ray[centerU -
                                      frameSize: centerU +
                                      frameSize, centerV -
                                      frameSize: centerV +
                                      frameSize])
    reconstruction_shifted = np.fft.fft2(ray_shifted).real
    reconstruction = np.fft.fftshift(reconstruction_shifted)

    kernel = np.outer(
        signal.windows.gaussian(
            N, 5), signal.windows.gaussian(
            N, 5))
    blurred = signal.fftconvolve(reconstruction - picture, kernel, mode='same')
    # result = np.sum((reconstruction - picture)**2) / N**2
    result = np.sum(blurred**2 / N**2)

    print('iteration performed, MoS = ' + str(result))

    return result


def initialGuess(picture, constants):
    if (ifRA == 0):
        RA_, DEC_, X_tel, Y_tel, Z_tel, scale = constants
    else:
        RA_, DEC_, phi, lmbda, scale = constants
    picture_flatten = picture.flatten()
    picture_95perc = np.percentile(picture_flatten, 0.95)
    picture_dist = np.zeros(len(picture_flatten))

    for i in range(len(picture_flatten)):
        if (picture_flatten[i] >= picture_95perc):
            U, V = tuple(np.unravel_index(i, picture.shape))
            picture_dist[i] = (U - N / 2)**2 + (V - N / 2)**2

    argflatten = np.argmax(picture_dist)
    min_dist_argU, min_dist_argV = tuple(
        np.unravel_index(argflatten, picture.shape))
    max_dist = picture_dist[argflatten]

    min_dist = max_dist
    for i in range(len(picture_flatten)):
        if ((picture_dist[i] < min_dist) and (picture_dist[i] != 0)):
            min_dist = picture_dist[i]

    R = N / (4 * scale)
    a = scale / N * 2 * R_EARTH_EQUATORIAL * \
        (np.sqrt(max_dist) + np.sqrt(min_dist) + R)
    e = np.abs((np.sqrt(max_dist) - np.sqrt(min_dist) - 2 * R) /
               (np.sqrt(max_dist) + np.sqrt(min_dist)))
    inclination = np.pi / 2 - DEC_
    node = RA_ + np.pi / 2
    pericenter = np.arctan2(min_dist_argU - N / 2, min_dist_argV - N / 2)
    epoch = 0
    # print(a, e, inclination, node, pericenter, epoch)
    return a, e, inclination, node, pericenter, epoch

# --------------------- minimization ---------------------


def minimization(picture, constants, filename, plotting, method):
    params_kepler0 = initialGuess(picture, constants)

    if (ifRA == 0):
        a, e, inclination, node, pericenter, epoch = params_kepler0 = params_kepler0
        params0 = np.array([a, e, inclination, node, pericenter, epoch])
    else:
        a, e, inclination, node, pericenter, epoch = params_kepler0
        h_min = a * (1 - e) - R_EARTH_EQUATORIAL
        params0 = np.array([a, h_min, inclination, node, pericenter, epoch])

    params_khol0 = kepler_to_khol(params_kepler0)

    if (method == 'NM'):
        if (ifRA == 0):
            result = minimize(
                measureOfSimilarity,
                params0,
                method='Nelder-Mead',
                options={
                    'disp': True,
                    'maxiter': 50,
                    'fatol': 1.,
                    'adaptive': True})
            num_results = result.x
        else:
            bnds = Bounds([R_EARTH_EQUATORIAL, 600, 0, -np.pi, -np.pi, -
                          np.pi], [np.inf, np.inf, np.pi, np.pi, np.pi, np.pi])
            result = minimize(
                measureOfSimilarity,
                params0,
                method='Nelder-Mead',
                bounds=bnds,
                options={
                    'disp': True,
                    'maxiter': 50,
                    'fatol': 1.,
                    'adaptive': True})
            num_results = result.x
    # not finished yet, to be upgraded
    else:
        ineq_cons = {'type': 'ineq', 'fun': lambda x: x[0]**2 + x[1]**2 + x[2]**2 - x[3]**2 - x[4]
                     ** 2, 'jac': lambda x: np.array([2 * x[0], 2 * x[1], 2 * x[2], -2 * x[3], -2 * x[4], 0])}
        result = minimize(
            measureOfSimilarity,
            params0,
            method='SLSQP',
            jac="2-point",
            constraints=ineq_cons,
            options={
                'ftol': 1,
                'disp': True})
        num_results = result.x

    # khol_params_final = num_results[0], num_results[1], num_results[2], num_results[3], num_results[4], num_results[5]
    # kepler_params_final = khol_to_kepler(khol_params_final)
    if (ifRA == 0):
        kepler_params_final = num_results[0], num_results[1], num_results[
            2], num_results[3], num_results[4], num_results[5]
    else:
        e_final = 1 - (num_results[1] + R_EARTH_EQUATORIAL) / num_results[0]
        kepler_params_final = num_results[0], e_final, num_results[2], num_results[3], num_results[4], num_results[5]

    khol_params_final = kepler_to_khol(kepler_params_final)

    a_final, e_final, inclination_final, node_final, pericenter_final, epoch_final = kepler_params_final
    if (ifRA == 0):
        a_true, e_true, inclination_true, node_true, pericenter_true, epoch_true = readTrueSatellite(
            filename)
    else:
        a_true, e_true, inclination_true, node_true, pericenter_true, epoch_true = 0, 0, 0, 0, 0, 0
    a_, e_, inclination_, node_, pericenter_, epoch_ = params_kepler0
    params_kepler_true = a_true, e_true, inclination_true, node_true, pericenter_true, epoch_true

    Ux, Uy, Uz, V1, V2, epoch = params_khol0
    params_init_vec = np.array([Ux, Uy, Uz, V1, V2])
    Ux, Uy, Uz, V1, V2, epoch = khol_params_final
    params_true = kepler_to_khol(params_kepler_true)
    Ux, Uy, Uz, V1, V2, epoch = params_true
    params_true_vec = np.array([Ux, Uy, Uz, V1, V2])
    Ux, Uy, Uz, V1, V2, epoch = khol_params_final
    params_fin_vec = np.array([Ux, Uy, Uz, V1, V2])

    d_IT = np.sqrt(np.sum((params_true_vec - params_init_vec)**2))
    d_FT = np.sqrt(np.sum((params_true_vec - params_fin_vec)**2))
    d_0 = np.sqrt(a_true / R_EARTH_EQUATORIAL)

    L_init = measureOfSimilarity(params0)
    L_fin = measureOfSimilarity(kepler_params_final)

    print('METRIC DIF TRUE-INITIAL:' + str(d_IT))
    print('METRIC DIF TRUE-FINAL:' + str(d_FT))
    print('Q1 INITIAL = ' + str(d_IT / d_0))
    print('Q1 FINAL = ' + str(d_FT / d_0))
    print('Q2 = ' + str(L_fin / L_init))

    print('RESULTS ' + filename)
    print(
        'INITIAL:',
        a_,
        e_,
        inclination_ *
        180. /
        np.pi,
        node_ *
        180. /
        np.pi,
        pericenter_ *
        180. /
        np.pi,
        epoch_)
    print(
        'FINAL:  ',
        a_final,
        e_final,
        inclination_final *
        180. /
        np.pi,
        node_final *
        180. /
        np.pi,
        pericenter_final *
        180. /
        np.pi,
        epoch_final *
        180. /
        np.pi)
    print(
        'TRUE:   ',
        a_true,
        e_true,
        inclination_true *
        180. /
        np.pi,
        node_true *
        180. /
        np.pi,
        pericenter_true *
        180. /
        np.pi,
        epoch_true)

    # technical, residuals (2x2 graphs)
    if (plotting == 1):
        RA_, DEC_, X_tel, Y_tel, Z_tel, scale = constants
        velocity = np.zeros(3)
        JD0 = 2400000.5 + 60473  # test date = 12.06.2024
        time_array = np.linspace(0, 3 * 86400, 3000)  # in seconds
        separation = 45. * np.pi / 180.
        sat_final = Satellite(
            a_final,
            e_final,
            inclination_final,
            node_final,
            pericenter_final,
            epoch_final,
            0.)
        sat_initial = Satellite(
            a_,
            e_,
            inclination_,
            node_,
            pericenter_,
            epoch_,
            0.)

        tel = GroundTelescope(X_tel, Y_tel, Z_tel, 'xyz')
        src = Source(RA_, DEC_, 'J2000')

        sat_final_array, tel_array, src, reconstruction_final, reconstruction_scale = testRayCalculations(
            [sat_final], [tel], src, time_array, JD0, separation, velocity, plotting=0, setscale=scale * 2 * R_EARTH_EQUATORIAL)
        sat_initial_array, tel_array, src, reconstruction_initial, reconstruction_scale = testRayCalculations(
            [sat_initial], [tel], src, time_array, JD0, separation, velocity, plotting=0, setscale=scale * 2 * R_EARTH_EQUATORIAL)

        fig, ax = plt.subplots(2, 2, figsize=(8, 8))

        ax[0, 0].imshow(reconstruction_final)
        ax[0, 1].imshow(picture)
        ax[1, 0].imshow(reconstruction_final - picture)
        ax[1, 1].imshow(reconstruction_initial)

        ax[0, 0].set_title('final result')
        ax[0, 1].set_title('real result')
        ax[1, 0].set_title('residuals')
        ax[1, 1].set_title('initial guess')

        plt.show()

    # 3D model (EARTH + ORBIT + SOURCE)
    if (plotting == 2):
        RA_, DEC_, X_tel, Y_tel, Z_tel, scale = constants
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # SOURCE DIRECTION

        src = Source(RA_, DEC_, 'J2000')
        X_s, Y_s, Z_s = src.get_XYZ()
        ax.plot([0, a_final * X_s], [0, a_final * Y_s],
                [0, a_final * Z_s], 'g')

        # EARTH

        u = np.linspace(0, 2 * np.pi, 100)
        v = np.linspace(0, np.pi, 100)
        x_E = R_EARTH_EQUATORIAL * np.outer(np.cos(u), np.sin(v))
        y_E = R_EARTH_EQUATORIAL * np.outer(np.sin(u), np.sin(v))
        z_E = R_EARTH_EQUATORIAL * np.outer(np.ones(np.size(u)), np.cos(v))

        ax.plot_surface(x_E, y_E, z_E)

        # SATELLITE

        acc = 1E-4
        sat_final = Satellite(
            a_final,
            e_final,
            inclination_final,
            node_final,
            pericenter_final,
            epoch_final,
            0.)
        sat_true = Satellite(
            a_true,
            e_true,
            inclination_true,
            node_true,
            pericenter_true,
            epoch_true,
            0.)
        JD0 = 2400000.5 + 60473  # test date = 12.06.2024
        time_array = np.linspace(0, 3 * 86400, 3000)  # in seconds
        N = len(time_array)
        X_final = np.zeros((N),)
        Y_final = np.zeros((N),)
        Z_final = np.zeros((N),)
        X_true = np.zeros((N),)
        Y_true = np.zeros((N),)
        Z_true = np.zeros((N),)
        for i in range(N):
            X_final[i], Y_final[i], Z_final[i] = sat_final.get_XYZ(
                time_array[i], acc)
            X_true[i], Y_true[i], Z_true[i] = sat_true.get_XYZ(
                time_array[i], acc)

        ax.plot(X_final, Y_final, Z_final, 'r')
        ax.plot(X_true, Y_true, Z_true, 'b')

        # BOX

        a_ = a_final * (1. + e_final)
        ax.plot([a_, a_, a_, a_, -a_, -a_, -a_, -a_], [a_, a_, -a_, -a_,
                a_, a_, -a_, -a_], [a_, -a_, a_, -a_, a_, -a_, a_, -a_], 'wo')

        ax.set_box_aspect([1, 1, 1])
        plt.show()

    # for article (2 UV + 2 RAY)
    if (plotting == 3):
        if (ifRA == 0):
            RA_, DEC_, X_tel, Y_tel, Z_tel, scale = constants
            velocity = np.zeros(3)
            JD0 = 2400000.5 + 60473  # test date = 12.06.2024
            time_array = np.linspace(0, 3 * 86400, 3000)  # in seconds
            separation = 45. * np.pi / 180.

            sat_final = Satellite(
                a_final,
                e_final,
                inclination_final,
                node_final,
                pericenter_final,
                epoch_final,
                0.)
            tel = GroundTelescope(X_tel, Y_tel, Z_tel, 'xyz')
            src = Source(RA_, DEC_, 'J2000')
            sat_array = [sat_final]
            tel_array = [tel]

            Npix = 1024
            wavelength = 1.

        else:
            RA, DEC, phi, lmbda, scale = constants
            velocity = np.zeros(3)
            JD0 = 2400000.5 + 60473
            time_array = np.linspace(0, 64801, 3000)
            separation = 0. * np.pi / 180.

            sat_final = Satellite(
                a_final,
                e_final,
                inclination_final,
                node_final,
                pericenter_final,
                epoch_final,
                0.)
            src = Source(RA, DEC, 'J2000')
            sat_array = [sat_final]
            tel_array = [
                GroundTelescope(
                    phi[i],
                    lmbda[i],
                    0.,
                    'geography') for i in range(
                    len(phi))]

            Npix = 1024
            wavelength = 1.36e-2
        '''
        velocity = np.zeros(3)
        JD0 = 2400000.5 + 60473  # test date = 12.06.2024
        time_array = np.linspace(0, 3 * 86400, 3000)  # in seconds
        separation = 45. * np.pi / 180.
        sat_final = Satellite(
            a_final,
            e_final,
            inclination_final,
            node_final,
            pericenter_final,
            epoch_final,
            0.)

        tel = GroundTelescope(X_tel, Y_tel, Z_tel, 'xyz')
        src = Source(RA_, DEC_, 'J2000')
        '''
        sat_final_array, tel_array, src, reconstruction_final, scaleSpatial = testRayCalculations(
            sat_array, tel_array, src, time_array, JD0, separation, velocity, plotting=0, setscale=scale * 2 * R_EARTH_EQUATORIAL)

        U, V, W = getUVWarray(sat_array, tel_array, src,
                              time_array, JD0, separation, velocity)
        if (len(U) != 0):
            maxValueU, maxValueV = np.max(np.abs(U)), np.max(np.abs(V))
            maxSpatialValue = maxValueU
            if (maxValueU < maxValueV):
                maxSpatialValue = maxValueV
        else:
            maxSpatialValue = R_EARTH_EQUATORIAL

        weigthFunction, ray, scaleSpatial, scaleAngular = makeRay(
            U, V, Npix, wavelength, scale * 2 * R_EARTH_EQUATORIAL)
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
        sigma = 10

        scaleAngular = scaleAngular / Npix * (2 * frameSize)

        ray = ray[centerU - frameSize: centerU + frameSize,
                  centerV - frameSize: centerV + frameSize]
        ray_shifted = np.fft.fftshift(ray)
        reconstruction_shifted = np.fft.fft2(ray_shifted).real
        reconstruction = np.fft.fftshift(reconstruction_shifted)
        scaleSpatial = 206265 * frameSize * \
            wavelength / (scaleAngular * 2 * R_EARTH_EQUATORIAL)

        xnew = np.linspace(0, scaleAngular, num=1001)
        # optimized ray, dots
        frameSize = len(ray[0, :])
        opt_ray_cutU = ray[int(frameSize / 2), int(frameSize / 2):]
        opt_ray_cutV = ray[int(frameSize / 2):, int(frameSize / 2)]
        angles_opt = np.linspace(0, scaleAngular, len(opt_ray_cutU))
        # real ray, dots
        picture_shifted = np.fft.fftshift(picture)
        real_ray_raw = np.fft.fft2(picture_shifted)
        real_ray = np.fft.fftshift(real_ray_raw)
        if (np.max(real_ray) > 0):
            real_ray = real_ray / np.max(real_ray)
        frameSize = len(real_ray[0, :])
        real_ray_cutU = real_ray[int(frameSize / 2), int(frameSize / 2):]
        real_ray_cutV = real_ray[int(frameSize / 2):, int(frameSize / 2)]
        angles_real = np.linspace(0, scaleAngular, len(opt_ray_cutU))
        # optimized ray, interp1d
        opt_ray_splU = CubicSpline(angles_opt, opt_ray_cutU)
        opt_ray_splV = CubicSpline(angles_opt, opt_ray_cutV)
        # real ray, interp1d
        real_ray_splU = CubicSpline(angles_real, real_ray_cutU)
        real_ray_splV = CubicSpline(angles_real, real_ray_cutV)

        fig, ax = plt.subplots(
            2, 2, figsize=(
                10, 8), gridspec_kw={
                'width_ratios': [
                    2, 3]})

        ax[0, 0].imshow(picture, extent=[-scale, scale, -
                        scale, scale], cmap='gray_r', vmin=0)
        ax[0, 0].set_xlabel(r'$u / D_\oplus$')
        ax[0, 0].set_ylabel(r'$v / D_\oplus$')
        ax[0, 0].set_title('Test synthetic UV', loc='left')

        ax[1, 0].imshow(reconstruction, extent=[-scaleSpatial,
                        scaleSpatial, -scaleSpatial, scaleSpatial], cmap='gray_r', vmin=0)
        ax[1, 0].set_xlabel(r'$u / D_\oplus$')
        ax[1, 0].set_ylabel(r'$v / D_\oplus$')
        ax[1, 0].set_title('Optimized UV', loc='left')

        U1 = np.abs(opt_ray_cutU)
        U2 = np.abs(opt_ray_splU(xnew))

        U1_ = np.abs(real_ray_cutU)
        U2_ = np.abs(real_ray_splU(xnew))

        ax[0, 1].plot(angles_opt, U1, 'ro', markersize=2, label='optimized')
        ax[0, 1].plot(xnew, U2, 'r--')
        ax[0, 1].plot(angles_real, U1_, 'bo', markersize=2, label='real')
        ax[0, 1].plot(xnew, U2_, 'b--')
        ax[0, 1].set_xlabel(r'$\theta_u,$' + ' mas')
        ax[0, 1].set_ylabel('Ray along U-axis')
        ax[0, 1].legend(loc='best')
        ax[0, 1].set_yscale('log')
        ax[0, 1].grid()
        ax[0, 1].set_xlim(xmin=0, xmax=scaleAngular / frameSize * 3 * sigma)
        ax[0, 1].set_ylim(ymin=1e-3)

        V1 = np.abs(opt_ray_cutV)
        V2 = np.abs(opt_ray_splV(xnew))

        V1_ = np.abs(real_ray_cutV)
        V2_ = np.abs(real_ray_splV(xnew))

        ax[1, 1].plot(angles_opt, V1, 'ro', markersize=2, label='optimized')
        ax[1, 1].plot(xnew, V2, 'r--')
        ax[1, 1].plot(angles_real, V1_, 'bo', markersize=2, label='real')
        ax[1, 1].plot(xnew, V2_, 'b--')
        ax[1, 1].set_xlabel(r'$\theta_v,$' + ' mas')
        ax[1, 1].set_ylabel('Ray along V-axis')
        ax[1, 1].legend(loc='best')
        ax[1, 1].set_yscale('log')
        ax[1, 1].grid()
        ax[1, 1].set_xlim(xmin=0, xmax=scaleAngular / frameSize * 3 * sigma)
        ax[1, 1].set_ylim(ymin=1e-3)

        plt.show()


minimization(picture, constants, filename, plotting=3, method='NM')
