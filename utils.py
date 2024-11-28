import matplotlib.pyplot as plt
import numpy as np
from uncertainties import ufloat
from math import *
import scipy.constants as const
from scipy.interpolate import CubicSpline, Akima1DInterpolator
import astropy.units as u
from astropy.cosmology import FlatLambdaCDM

def ghz_to_mum(frequency_GHz):
    return (const.c/frequency_GHz/1e9) * 1e6

def mum_to_ghz(wavelength_um):
    return (const.c/(wavelength_um / 1e6))/1e9


def mass_kgs_to_solar(mass,unit_of_input_mass):
    # 1 solar mass in kilograms
    solar_mass_kg = 1.989e+30 #Solar mass in kg

    if unit_of_input_mass == 'kg' or unit_of_input_mass == 'kgs':
        return mass / solar_mass_kg # Convert kg to solar masses

    elif unit_of_input_mass == 'solar':
        return mass * solar_mass_kg

    else:
        raise ValueError("Invalid target unit. Use 'solar' for solar masses or 'kg' for kilograms.")


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





def blackbody(nu, temp):
    """
    Planck's law for black body emission, per unit frequency. [B(nu_0,T) or B(v_0,T)]

    :param nu: Rest frame frequency in Hz.
    :param temp: Temperature in K.
    :return: Emission in units of W / (m^2 Hz)
    """
    return 2 * const.h * nu ** 3 / const.c ** 2 / (np.exp(const.h * nu / (const.k * temp)) - 1)




def kappa(nu_rest,beta):

    """

    :param nu_rest: Rest frame frequency in Hz.
    :param beta: Emissivity coefficient, dimensionless
    :return: mass absorption coefficient (kappa) in m^2 kg^−1
    """

    #Taken from Interferopy (Also see: https://iopscience.iop.org/article/10.3847/1538-4357/ab2beb/pdf)
    #kappa_ref = 2.64  # m**2/kg
    #kappa_nu_ref = const.c / 125e-6 # Hz
    #return kappa_ref * (nu_rest / kappa_nu_ref) ** beta

    #Saw this value in Decarli et al. 2018 (https://www.aanda.org/articles/aa/pdf/2022/09/aa43920-22.pdf)
    #kappa_ref = 0.77 * u.cm**2/u.g
    #kappa_ref = kappa_ref.to(u.m**2/u.kg).value
    #kappa_nu_ref = 352e9
    #return kappa_ref * (nu_rest / kappa_nu_ref) ** beta

    #Saw this value in Tripodi et al. 2022 (https://www.aanda.org/articles/aa/pdf/2022/09/aa43920-22.pdf)
    #kappa_ref = 0.45 * u.cm**2/u.g
    #kappa_ref = kappa_ref.to(u.m**2/u.kg).value
    #kappa_nu_ref = 250e9
    #return kappa_ref * (nu_rest / kappa_nu_ref) ** beta

    """
    #Taken from Decarli et al. 2023
    kv = []
    for i in nu_rest:
        kv.append(kappa_Draine_2003(nu_rest = i,beta = beta))

    kv = np.asarray(kv)
    return kv
    
    """





def kappa_Draine_2003(nu_rest,beta):

    """
     dust emissivity law following Draine et al. 2003 (https://arxiv.org/pdf/astro-ph/0304489)

    :param nu: Rest frame frequency in Hz.
    :param beta: Emissivity coefficient, dimensionless
    :return: mass absorption coefficient (kappa) in m^2 kg^−1
    """

    nu_rest_GHz = nu_rest * 1e-9  # Convert Hz to GHz
    nu_in_wavelength_micrometer = ghz_to_mum(nu_rest_GHz)


    if 6<nu_in_wavelength_micrometer<200: #or nu_GHz > 1500 GHz

        #Table 5 of Draine et al. 2003 (https://arxiv.org/pdf/astro-ph/0304489)
        wavelength_micrometer = np.array([6,7,8,9.7,12,14,18,25,40,60,100,140,200]) #in micrometer
        kappa_abs = np.array([4.52e2,3.85e2,7.29e2,2.1e2,9.37e2,4.92e2,7.76e2,4.79e2,2.36e2,8.7e1,2.71e1,1.39e1,6.37]) # in cm^2 g^-1


        spl_akima1d = Akima1DInterpolator(wavelength_micrometer,kappa_abs)
        spl_cubic = CubicSpline(wavelength_micrometer, kappa_abs)

        """
        xnew = np.linspace(6, 199, num=1001)
        plt.scatter(wavelength_micrometer,kappa_abs)
        plt.plot(xnew,spl_akima1d(xnew),'-.', label='Akima1DInterpolator')
        plt.plot(xnew, spl_cubic(xnew), '--', label='CubicSpline')
        plt.axvline(ghz_to_mum(1500),color='black',label='1500GHz')
        plt.legend()
        plt.show()
        """

        kappa_cm2_g_minus_1 = spl_akima1d(nu_in_wavelength_micrometer) * u.cm**2/u.g
        kappa_m2_kg_minus_1 = kappa_cm2_g_minus_1.to(u.m**2/u.kg).value

        return kappa_m2_kg_minus_1



    else: #nu_GHz < 1500 GHz
        #power-law extrapolation for nu_GHz < 1500 GHz
        #This equation is from Decarli et al. 2023 (A comprehensive view of the interstellar medium in a quasar host galaxy at z ≈ 6.4)

        kappa_cm2_g_minus_1 = 6.37 * ((nu_rest_GHz/1500)**beta) * u.cm**2/u.g
        kappa_m2_kg_minus_1 = kappa_cm2_g_minus_1.to(u.m ** 2 / u.kg).value

        return kappa_m2_kg_minus_1



def tau(nu_rest,beta,solid_angle,mass_dust,z):

    """

    :param nu_rest: Rest frame frequency in Hz.
    :param beta: Emissivity coefficient, dimensionless.
    :param solid_angle: Solid angle in steradians.
    :param mass_dust: Total dust mass in kg.
    :param z: redshift
    :return: optical depth (dimensionless Unit)
    """

    cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    dl = cosmo.luminosity_distance(z).to(u.m).value #Luminosity Distance in m
    da = (dl*((1+z)**-2))
    return (kappa(nu_rest = nu_rest,beta=beta) * mass_dust)/(solid_angle * (da**2))



def dust_s_obs(nu_obs, z, solid_angle, mass_dust, temp_dust, beta, cmb_contrast = True, cmb_heating = True,
               output_unit_mjy=False):

    """

    :param nu_obs: Observed frame frequency in Hz.
    :param z: Redshift
    :param solid_angle: Solid angle in arcsec^2 (arcsec squared)

    :param mass_dust: Total dust mass in kg.
    :param temp_dust: Intrinsic dust temperature in K (the source would have at z=0).
    :param beta: Emissivity coefficient, dimensionless.

    :param cmb_contrast: Correcting for cosmic microwave background contrast.
    :param cmb_heating: Correcting for cosmic microwave background heating (This is important at
    high redshifts (high-z) where the temperature of the CMB (T_CMB) ~ dust temperature (T_dust) )

    :param output_unit_mjy: Convert W/Hz/m^2 to mJy

    :return: Observed flux density in W/Hz/m^2 or mJy.
    """



    temp_cmb_0 = 2.73  # CMB temperature at z=0
    temp_cmb_z = (1 + z) * temp_cmb_0 ## CMB temperature at z

    nu_rest = nu_obs * (1 + z) #Convert to rest frequencies

    solid_angle_arcsec2 = solid_angle * u.arcsec**2
    solid_angle_steradians = solid_angle_arcsec2.to(u.steradian).value

    #Correct for CMB heating from da Cunha+2013 (See interferopy source code)

    if cmb_heating==True:
        temp_dust_z = (temp_dust ** (4 + beta) + temp_cmb_0 ** (4 + beta) * ((1 + z) ** (4 + beta) - 1)) ** (1 / (4 + beta))
    else:
        temp_dust_z = temp_dust


    #Correct for CMB contrast

    if cmb_contrast==True:
        f_cmb = blackbody(nu_rest, temp_dust_z) - blackbody(nu_rest, temp_cmb_z)
    else:
        f_cmb = 1

    t = tau(nu_rest = nu_rest, beta=beta, solid_angle = solid_angle_steradians,mass_dust = mass_dust,z = z)


    if output_unit_mjy==True:
        flux_obs = (solid_angle_steradians/((1+z)**3)) * f_cmb * (1-np.exp(-t)) * 1e29
    else:
        flux_obs = (solid_angle_steradians / ((1 + z) ** 3)) * f_cmb * (1 - np.exp(-t))



    return flux_obs





#print(kappa(nu=mum_to_ghz(10) * 1e9))

freq = np.linspace(1e1,1e4,10000)
mass = mass_kgs_to_solar(10**8.94,'solar')
z_qso = 6.4386
#print(ghz_to_mum(freq*(1+z_qso)))


s = dust_s_obs(freq*1e9, z = z_qso, solid_angle=0.155, mass_dust=mass, temp_dust=47, beta=1.84,output_unit_mjy=True)





decarli_freq = np.array([97.435,109.342,231.946, 239.520, 246.646, 255.459, 280.875, 292.989, 292.989, 331.688, 457.035, 469.093])
decarli_wave_mm = np.array([3.077, 2.742, 1.293,  1.252,  1.215, 1.174,  1.067,  1.023, 0.937, 0.904, 0.656,  0.639])
decarli_flux_mjy = np.array([0.244,0.334,  3.81, 3.75, 4.33,  4.42, 5.82,  6.12,  8.15,  8.63, 12.52, 12.51])




plt.scatter(decarli_freq,decarli_flux_mjy)
plt.plot(freq,s)
plt.xscale('log')
plt.yscale('log')
plt.ylim(1e-2,1e2)
plt.show()



