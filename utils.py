import numpy as np
from uncertainties import ufloat
from math import *
import scipy.constants as const


def ghz_to_mum(frequency_GHz):
    return (const.c/frequency_GHz/1e9) * 1e6

def mum_to_ghz(wavelength_um):
    return (const.c/(wavelength_um / 1e6))/1e9

def luminosity_distance(x,H_0=70,W_M=0.3):
    z=x
    W_R = 0.                #Omega Radiation
    W_K = 0.                #Omega curvature
    c = 299792.458          #speed of light in km/s
    H_0 = H_0 #69.6             #Hubbles constant
    W_M = W_M #0.286               #Omega matter
    W_V = 1 - W_M           #Omega vacuum
    Tyr = 977.8             # coefficent for converting 1/H into Gyr

    h = H_0/100
    W_R = 4.165e-5/(h*h)
    W_K = 1-W_M-W_R-W_V
    a_z = 1.0/(1+1.0*z)
    age = 0
    n=1000

    for i in range (n):
        a = a_z * (i + 0.5) / n
        a_dot = np.sqrt(W_K+(W_M/a)+(W_R/(a*a))+(W_V*a*a))
        age =age + 1./a_dot
    z_age = a_z*age/n
    z_age_Gyr=(Tyr/H_0)*z_age

    DTT = 0.0
    DCMR = 0.0

    # do integral over a=1/(1+z) from az to 1 in n steps, midpoint rule
    for i in range(n):
        a = a_z + (1 - a_z) * (i + 0.5) / n
        adot = np.sqrt(W_K + (W_M / a) + (W_R / (a * a)) + (W_V * a * a))
        DTT = DTT + 1. / adot
        DCMR = DCMR + 1. / (a * adot)

    DTT = (1. - a_z) * DTT / n
    DCMR = (1. - a_z) * DCMR / n
    age = DTT + z_age
    age_Gyr = age * (Tyr / H_0)
    DTT_Gyr = (Tyr / H_0) * DTT
    DCMR_Gyr = (Tyr / H_0) * DCMR
    DCMR_Mpc = (c / H_0) * DCMR

    # tangential comoving distance

    ratio = 1.00
    x = sqrt(abs(W_K)) * DCMR
    if x > 0.1:
        if W_K > 0:
            ratio = 0.5 * (exp(x) - exp(-x)) / x
        else:
            ratio = sin(x) / x
    else:
        y = x * x
        if W_K < 0: y = -y
        ratio = 1. + y / 6. + y * y / 120.
    DCMT = ratio * DCMR
    DA = a_z * DCMT
    DA_Mpc = (c / H_0) * DA
    kpc_DA = DA_Mpc / 206.264806
    DA_Gyr = (Tyr / H_0) * DA
    DL = DA / (a_z * a_z)
    DL_Mpc = (c / H_0) * DL
    return DL_Mpc




def line_luminosity_solar(I, obs_freq, err_I=0, z=0, D_Mpc=0, err_D_Mpc=0, mu=1, err_mu=0, H_0=70, W_M=0.3):

    """

    :param I: Integrated Line Flux (in Jy kms-1)
    :param obs_freq: Observed Frequency of the Line (in GHz)
    :param err_I: error on the Integrated Line Flux (in Jy kms-1)
    :param z: Redshift
    :param D_Mpc: Luminosity Distance in Mpc
    :param err_D_Mpc: error on the Luminosity Distance in Mpc
    :param mu: Gravitational Magnification Factor
    :param err_mu: error on Gravitational Magnification Factor
    :param H_0: Hubbles constant
    :param W_M: Omega matter
    :return:
    """

    print("Line Luminosity (in terms for solar L_(.)")

    integrated_line_flux = ufloat(I, err_I)
    mu = ufloat(mu, err_mu)

    if z != 0 and D_Mpc == 0:
        Luminosity_Distance_Mpc = luminosity_distance(z, H_0, W_M)

    if D_Mpc != 0:
        if err_D_Mpc == 0:
            Luminosity_Distance_Mpc = D_Mpc
        else:
            Luminosity_Distance_Mpc = ufloat(D_Mpc, err_D_Mpc)

    print(f"Integrated Line Flux = {integrated_line_flux} JyKm/s")
    print(f"Luminosity Distance = {Luminosity_Distance_Mpc} Mpc")
    print(f"Observed Frequency = {obs_freq} GHz")
    if mu != 1:
        print(f"Gravitational Magnification Factor = {mu}")
    print("")

    constant = 0.00104

    line_luminsoity = (constant * integrated_line_flux * (Luminosity_Distance_Mpc ** 2) * obs_freq) / 1e8

    print(f"Line Luminosity Before magnification correction = {line_luminsoity} x10^8 L_(.)")
    print(f"Line Luminosity After magnification correction = {line_luminsoity / mu} x10^8 L_(.)")




