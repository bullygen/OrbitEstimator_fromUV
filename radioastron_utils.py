import numpy as np
import matplotlib.pyplot as plt

from pysofa_ctypes import *
from timeUtilities import *
from satelliteClass import *
from UVplaneMaker import *


def AMS_to_rad(A, M, S):
    return (A + M / 60 + S / 60**2) * np.pi / 180.


def get_RA_test_params():
    RA = (8 + 54. / 60 + 48.875 / 60**2) * np.pi / 12.
    DEC = (20 + 6. / 60 + 30.641 / 60**2) * np.pi / 180.

    # Effelsberg     +50 31 29    -06 52 58
    # Jodrell Bank   +53 14 10    +02 18 26
    # Onsala         +57 23 35    -11 55 04
    # Noto           +36 52 34    -14 59 21
    # Torun          +53 05 43    -18 33 46
    # Yebes          +40 31 31    +03 05 19
    # Seshan         +31 05 47   -121 11 19
    # Hartrao        -25 53 25    -27 41 08
    # Kalyazin       +57 13 23    -37 54 01
    # Green Bank     +38 25 58    +79 50 23
    # Korean VLBI 1  +37 33 55   -126 56 28
    # Korean VLBI 2  +35 32 44   -129 14 59
    # Korean VLBI 3  +33 17 20   -126 27 35

    phi = [AMS_to_rad(50, 31, 29),
           AMS_to_rad(53, 14, 10),
           AMS_to_rad(57, 23, 35),
           AMS_to_rad(36, 52, 34),
           AMS_to_rad(53, 5, 43),
           AMS_to_rad(40, 31, 31),
           AMS_to_rad(31, 5, 47),
           AMS_to_rad(-25, -53, -25),
           AMS_to_rad(57, 13, 23),
           AMS_to_rad(38, 25, 58),
           AMS_to_rad(37, 33, 55),
           AMS_to_rad(35, 32, 44),
           AMS_to_rad(33, 17, 20)]
    lmbda = [AMS_to_rad(-6, -52, -58),
             AMS_to_rad(2, 18, 26),
             AMS_to_rad(-11, -55, -4),
             AMS_to_rad(-14, -59, -21),
             AMS_to_rad(-18, -33, -46),
             AMS_to_rad(3, 5, 19),
             AMS_to_rad(-121, -11, -19),
             AMS_to_rad(-27, -41, -8),
             AMS_to_rad(-37, -54, -1),
             AMS_to_rad(79, 50, 23),
             AMS_to_rad(-126, -56, -28),
             AMS_to_rad(-129, -14, -59),
             AMS_to_rad(-126, -27, -35)]
    return RA, DEC, phi, lmbda


def get_RA_XYZ(filename):
    strings = open(filename)
    N = 0
    for line in strings:
        N += 1
    strings.close()

    R = np.zeros((N, 3))
    V = np.zeros((N, 3))
    strings = open(filename)
    cnt = 0
    for line in strings:
        words = line.strip().split(' ')
        words[:] = [x for x in words if x]
        R[cnt, 0] = float(words[1])
        R[cnt, 1] = float(words[2])
        R[cnt, 2] = float(words[3])
        V[cnt, 0] = float(words[4])
        V[cnt, 1] = float(words[5])
        V[cnt, 2] = float(words[6])
        cnt += 1
    strings.close()
    return R, V


def get_RA_UV():
    strings = open('RA_raw_data/RAKS04E_UV.txt')
    N = 0
    for line in strings:
        N += 1
    strings.close()

    UV = np.zeros((N, 2))
    strings = open('RA_raw_data/RAKS04E_UV.txt')
    cnt = 0
    for line in strings:
        words = line.strip().split('\t')
        UV[cnt, 0] = float(words[3])
        UV[cnt, 1] = float(words[4])
        cnt += 1
    return UV


def find_true_elements(R, V, plotting):
    N = len(R[:, 0])
    a = np.zeros(N)
    e = np.zeros(N)
    inclination = np.zeros(N)
    pericenter = np.zeros(N)
    node = np.zeros(N)
    epoch = np.zeros(N)
    for i in range(N):

        # c (Omega, i, p)

        R_ = R[i, :]
        V_ = V[i, :]
        c = np.cross(R_, V_)
        p = np.sum(c**2) / GM_EARTH
        inclination[i] = np.arctan(np.sqrt(c[0]**2 + c[1]**2) / c[2])
        node[i] = np.arctan2(c[0], -c[1])

        # h (a, e)

        h = np.sum(V[i, :]**2) - 2 * GM_EARTH / np.sqrt(np.sum(R[i, :]**2))
        a[i] = - GM_EARTH / h
        # e[i] = np.sqrt(1 - p / a[i])

        # lambda (omega)

        L = np.cross(V_, c) - GM_EARTH * R[i, :] / np.sqrt(np.sum(R[i, :]**2))
        sinPeri = L[2] / (np.sqrt(np.sum(L**2)) * np.sin(inclination[i]))
        cosPeri = (L[0] * np.cos(node[i]) + L[1] *
                   np.sin(node[i])) / np.sqrt(np.sum(L**2))
        pericenter[i] = np.arctan2(sinPeri, cosPeri)
        e[i] = np.sqrt(np.sum(L**2)) / GM_EARTH

        # epoch
        E0 = np.arccos(1. / e[i] * (1 - np.sqrt(np.sum(R[i, :]**2)) / a[i]))
        if (i == 0):
            v0 = 2 * \
                np.arctan(np.sqrt((1 + e[i]) / (1 - e[i])) * np.tan(E0 / 2))
            print(v0, 'anomaly at t = 0')
        if (np.dot(V[i, :], L) > 0):
            E0 = - E0
        epoch[i] = E0 - e[i] * np.sin(E0) - i * np.sqrt(GM_EARTH / a[i]**3)

    if (plotting == 1):
        fig, ax = plt.subplots(3, 2)
        ax[0, 0].plot(a, label='a')
        ax[1, 0].plot(e, label='e')
        ax[2, 0].plot(inclination * 180 / np.pi, label='inclination')
        ax[0, 1].plot(node * 180 / np.pi, label='node')
        ax[1, 1].plot(pericenter * 180 / np.pi, label='pericenter')
        ax[2, 1].plot(epoch, label='epoch')

        ax[0, 0].legend()
        ax[1, 0].legend()
        ax[2, 0].legend()
        ax[0, 1].legend()
        ax[1, 1].legend()
        ax[2, 1].legend()
        plt.show()

    a_ = np.mean(a)
    e_ = np.mean(e)
    inclination_ = np.mean(inclination)
    node_ = np.mean(node)
    pericenter_ = np.mean(pericenter)
    epoch_ = np.mean(epoch)

    return a_, e_, inclination_, node_, pericenter_, epoch_


def write_RA_UV(UV):
    U = UV[:, 0] * (2 * R_EARTH_EQUATORIAL)
    V = UV[:, 1] * (2 * R_EARTH_EQUATORIAL)
    Npix = 1024
    wavelength = 1.
    maxValueU, maxValueV = np.max(np.abs(U)), np.max(np.abs(V))
    maxSpatialValue = maxValueU
    if (maxValueU < maxValueV):
        maxSpatialValue = maxValueV

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
    reconstruction_scale = 206265 * frameSize * \
        wavelength / (scaleAngular * 2 * R_EARTH_EQUATORIAL)
    return reconstruction, reconstruction_scale, maxSpatialValue


def write_UV_to_file(realData, realData_scale, maxSpatialValue):
    filename = 'RA'
    file = open('testBlurredUV/' + filename + '.dat', 'w')

    file.write('Spatial scale (from center to edge) of the image below:\n')
    file.write(
        'scale = ' +
        str(realData_scale) +
        ' in Earth diameters\n\n')

    file.write('Number of dots calculated: ')
    file.write('N = ' + '3000' + ' for 3 days of observation\n')

    file.write('Blurred image:\n')
    N = len(realData[0, :])
    for i in range(N):
        string = ''
        for j in range(N):
            string = string + ' ' + '{:.2f}'.format(realData[i, j])
        file.write(string + '\n')

    file.close()


def RA_direct_problem()


# filename = 'RA_raw_data/RA120413_2000_v02.scf'
filename = 'RA_raw_data/RA140404_1100_v03_all.scf'
R, V = get_RA_XYZ(filename)
a_, e_, inclination_, node_, pericenter_, epoch_ = find_true_elements(
    R, V, plotting=0)
print(a_, e_, inclination_, node_, pericenter_, epoch_)
N = len(R[:, 0])
print(N)
N_points = 3000

# real UV coverage
UV = get_RA_UV()
realData, realData_scale, maxSpatialValue = write_RA_UV(UV)
# writing into a file UV to fit
# write_UV_to_file(realData, realData_scale, maxSpatialValue)
plt.imshow(realData)
plt.show()

# reconstructed UV coverage
RA, DEC, phi, lmbda = get_RA_test_params()

sat = Satellite(a_, e_, inclination_, node_, pericenter_, epoch_, 0.)
src = Source(RA, DEC, 'J2000')

sat_array = [sat]
tel_array = [
    GroundTelescope(
        phi[i],
        lmbda[i],
        0.,
        'geography') for i in range(
        len(phi))]

time_array = np.linspace(0, N - 1, N_points)
JD0 = 2400000.5 + 60473
separation = 0.
velocity = np.zeros(3)

sat_array, tel_array, src, reconstruction, reconstruction_scale = testRayCalculations(
    sat_array,
    tel_array,
    src,
    time_array,
    JD0,
    separation,
    velocity,
    plotting=1,
    setscale=0)

# plotting
fig, ax = plt.subplots(1, 2)

ax[0].imshow(realData, extent=[-reconstruction_scale, reconstruction_scale, -
             reconstruction_scale, reconstruction_scale], cmap='gray_r', vmin=0)
ax[0].set_xlabel(r'$u / D_\oplus$')
ax[0].set_ylabel(r'$v / D_\oplus$')
ax[0].set_title('Actual RadioAstron + ground telescopes UV')

ax[1].imshow(reconstruction, extent=[-reconstruction_scale, reconstruction_scale, -
             reconstruction_scale, reconstruction_scale], cmap='gray_r', vmin=0)
ax[1].set_xlabel(r'$u / D_\oplus$')
ax[1].set_ylabel(r'$v / D_\oplus$')
ax[1].set_title('Reconstructed UV (DP)')

plt.show()

# done 2nd time, no optimization so far ;(
Npix = 1024
wavelength = 1.

U, V, W = getUVWarray(sat_array, tel_array, src,
                      time_array, JD0, separation, velocity)
weigthFunction_simulated, ray1, scaleSpatial1, scaleAngular1 = makeRay(
    U, V, Npix, wavelength, maxSpatialValue)
weigthFunction_real, ray0, scaleSpatial0, scaleAngular0 = makeRay(
    UV[:, 0] * (2 * R_EARTH_EQUATORIAL), UV[:, 1] * (2 * R_EARTH_EQUATORIAL), Npix, wavelength, maxSpatialValue)

scaleSpatial0 = scaleSpatial0 / (2 * R_EARTH_EQUATORIAL)
scaleSpatial1 = scaleSpatial1 / (2 * R_EARTH_EQUATORIAL)

fig, ax = plt.subplots(1, 2)

ax[0].imshow(weigthFunction_real, extent=[-scaleSpatial0,
                                          scaleSpatial0, -scaleSpatial0, scaleSpatial0], cmap='gray_r', vmin=0)
ax[0].set_xlabel(r'$u / D_\oplus$')
ax[0].set_ylabel(r'$v / D_\oplus$')
ax[0].set_title('Actual RadioAstron + ground telescopes UV')

ax[1].imshow(weigthFunction_simulated, extent=[-scaleSpatial1,
                                               scaleSpatial1, -scaleSpatial1, scaleSpatial1], cmap='gray_r', vmin=0)
ax[1].set_xlabel(r'$u / D_\oplus$')
ax[1].set_ylabel(r'$v / D_\oplus$')
ax[1].set_title('Reconstructed UV (DP)')

plt.show()
