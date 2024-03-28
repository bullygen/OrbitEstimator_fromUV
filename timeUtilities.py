import numpy as np
import requests
from scipy import interpolate

from pysofa_ctypes import *

TT_TAI = 32.184
JD_to_MJD = 2400000.5


def findLatestBulletin():
    response = requests.get(
        'https://datacenter.iers.org/data/latestVersion/bulletinA.txt')
    with open("downloads/IERS_bulletinA.txt", 'wb') as f:
        f.write(response.content)
        print('Latest IERS bulletin A is saved to downloads')


def openBulletin():
    TAI_UTC = 0.
    dUT_array = np.zeros(365)
    MJD0 = 0.
    with open("downloads/IERS_bulletinA.txt", 'r') as f:
        count = 0
        for line in f:
            count += 1
            if (count == 76):
                TAI_UTC = float(line.strip().split()[3])
            elif ((count >= 84) and (count < 449)):
                dUT_array[count - 84] = float(line.strip().split()[6])
                if (count == 84):
                    MJD0 = float(line.strip().split()[3])
    return TAI_UTC, dUT_array, MJD0


def calculateLibrations(JD_UTC):
    return 0


def TT_from_UTC(JD_UTC, TAI_UTC):
    JD_TT = JD_UTC + TAI_UTC + TT_TAI
    return JD_TT


def UT1_from_UTC(JD_UTC, dUT_array, MJD0):
    JD_UT1 = JD_UTC
    N = len(JD_UTC)
    dUT_real = np.zeros(N)

    JD_begin = MJD0 + JD_to_MJD
    JD_end = JD_begin + 365.

    if ((JD_UTC[0] < JD_begin) or (JD_UTC[-1] > JD_end)):
        print('not enough data, download another bulletin')
        print(JD_UTC[0], JD_begin)
        print(JD_UTC[-1], JD_end)
    else:
        for i in range(N):
            MJD_nearest = int(JD_UTC[i] - JD_to_MJD)
            index = int(MJD_nearest - MJD0)
            offset = 3

            MJD_cut = np.linspace(-offset, offset, 2 * offset + 1)
            dUT_cut = dUT_array[index - offset: index + offset + 1]
            polynomial = interpolate.interp1d(MJD_cut, dUT_cut)

            dUT_real[i] = polynomial(JD_UTC[i] - JD_to_MJD - MJD_nearest)

    JD_UT1 = JD_UT1 + dUT_real / 86400.

    return JD_UT1
