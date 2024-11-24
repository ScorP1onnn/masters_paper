import numpy as np
from networkx.algorithms.bipartite import color
from scipy.ndimage import label
import pandas as pd
from scipy.optimize import curve_fit
import interferopy.tools as iftools #(uses only numpy versions 1.x not 2.x)
import matplotlib.pyplot as plt
from scipy.constants import c
import scipy.constants as const
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import emcee
from scipy.signal import freqs
from uncertainties import ufloat

import utils as utils

"""

#This plot is similar to the one in Stacey et al. 2018 (https://arxiv.org/pdf/1705.10530) for
#PSSJ 2322+1944. Use it as reference

fig, ax1 = plt.subplots(1,1, figsize=(6, 6), tight_layout=True)
dust_mass = 1e39
dust_temp = 47 #K
dust_beta = 1.6
z = 4.12

xxx = np.linspace(10, 1e5, 10000) #micrometre
freqs_xxx = c/(xxx/1e6)/1e9

r = iftools.dust_sobs(freqs_xxx * 1e9,z, dust_mass, dust_temp, dust_beta ) * 1e26
ax1.plot(xxx/(1+z),r, 'blue')
ax1.set_xlabel(r"Rest Wavelength [$\mu$m]") #rest wavelength = observed wavelength/(1+z)
ax1.set_ylabel("Flux [Jy]")
ax1.set_yscale('log')
ax1.set_ylim(1.5*1e-6, 1e0)
ax1.set_xscale('log')


ax1_1 = ax1.twiny()
ax1_1.plot(freqs_xxx,r,color='green', label = fr'T = {dust_temp}, $\beta$ = {dust_beta}')
ax1_1.set_xlabel("Observed Frequency [GHz]")
ax1_1.invert_xaxis()
ax1_1.set_xscale('log')
plt.legend()
plt.show()


exit()


"""
def plot_sed_freq(freqs,fluxes,freqs_extra = [],fluxes_extra = []):
    plt.scatter(freqs, fluxes)

    if len(freqs_extra)!=0 and len(fluxes_extra)!=0:
        plt.scatter(freqs_extra, fluxes_extra,label='Extra')


    # Set x-axis to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')
    # Add labels and title
    plt.xlabel("Observed Frequency")
    plt.ylabel("Flux Density (mJy)")
    #plt.legend()

    plt.xlim(1e0, 1e6)
    plt.gca().invert_xaxis()

    plt.ylim(1e-6, 1e2)


    # Show plot
    plt.show()


def plot_sed_wave(wave,fluxes,wave_extra = [],fluxes_extra = []):
    plt.scatter(wave, fluxes)

    if len(wave_extra)!=0 and len(fluxes_extra)!=0:
        plt.scatter(wave_extra, fluxes_extra,label='Extra')


    # Set x-axis to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')
    # Add labels and title
    plt.xlabel("Observed Wavelength")
    plt.ylabel("Flux Density (mJy)")
    #plt.legend()

    plt.xlim(0.5, 1e6)
    #plt.gca().invert_xaxis()

    #plt.ylim(1e-6, 1e0)


    # Show plot
    plt.show()




#SDSS from NED
df = pd.read_csv(r'NED_photo_points/sdss_j2054_0005_table_photandseds_NED.csv')
print(df.columns)

#df.drop([0,2,4,6],inplace=True) #Dropping Pan-STARRS1 Observations and K(keck)

df.drop([0,2,3,4,5,6],inplace=True) #Dropping Pan-STARRS1 Observations and K(keck)

freq_ned = df['Frequency'].to_numpy()/1e9 #Convert Frequncies to GHz
wave_ned = utils.ghz_to_mum(freq_ned) #Wavelength in micrometers
flux_ned = df['Flux Density'].to_numpy()*1e3 #Convert to mJy
flux_err_ned = df['Upper limit of uncertainty'].to_numpy()*1e3 #Convert to mJy
references_ned = df['Refcode']



#plot_sed_wave(wave=wave_ned,fluxes=flux_ned)


#Wise W1 value (Blain et al. 2013)
w1_wave = np.array([3.4]) #micrometers
w1_freq = utils.mum_to_ghz(w1_wave)
w1_mag = ufloat(18.017,0.337)
#Convert mag to flux for WISE: https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux
w1_flux = (309.540* 10**((w1_mag)/-2.5)) * 1e3 #Flux in mJy


# Leipski et al. 2014 (https://iopscience.iop.org/article/10.1088/0004-637X/785/2/154/pdf)

leipski_wave = np.array([100,160,250,350]) #Wavelengths in micrometer
leipski_freq = utils.mum_to_ghz(leipski_wave)
leipski_flux = np.array([3.1,10.5,15.2,12.0]) #Flux in mJy
leipski_flux_err = np.array([1.0,2.0,5.4,4.9]) #Flux in mJy


#Wang et al. 2011, Wang et al. 2008, Tripodi et al 2024,

wwt_freq = np.array([1.4,92.26,250, 262.6, 263.93, 488.31 , 674.97])
wwt_wave = utils.ghz_to_mum(wwt_freq)
wwt_flux = np.array([17e-3, 0.082, 2.38,2.93,3.08,11.71,9.87]) #Flux in mJy
wwt_flux_err = np.array([23e-3,0.009,0.53,0.07,0.03,0.11,0.94]) #Flux in mJy


#Hashimoto et al. 2018 (https://arxiv.org/pdf/1811.00030)
#Our data probe dust continuum emission at the rest-frame wavelength, λ_rest, of ≈ 87 μm
hash_wave = np.array([87*(1+6.0391)]) #In micrometers
has_freq = utils.mum_to_ghz(hash_wave)
hash_flux = ufloat(10.35,0.15) #Flux in mJy


#Salak et al. 2024 (https://iopscience.iop.org/article/10.3847/1538-4357/ad0df5/pdf)
#λ_rest, of ≈ 123 μm
salak_wave = np.array([123*(1+6.0391)]) #In micrometers
salak_freq = utils.mum_to_ghz(salak_wave)
salak_flux = ufloat(5.723 , 0.009) #Flux in mJy


#Our Value
sai_freq = np.array([1461.134/(1+6.0391)]) #in GHz
sai_wave = utils.ghz_to_mum(sai_freq)
sai_flux = ufloat(0.8,0.1) #lux in mJy


plt.scatter(wave_ned,flux_ned,label='SDSS-NED')
plt.scatter(w1_wave,w1_flux.n,label='W1-Blain et al. 2013')
plt.scatter(leipski_wave,leipski_flux,label='Leipski et al. 2014')
plt.scatter(wwt_wave,wwt_flux,label='WWT')
plt.scatter(hash_wave,hash_flux.n,label='Hashimoto et al. 2018')
plt.scatter(salak_wave,salak_flux.n,label='Salak et al. 2024')

plt.scatter(sai_wave,sai_flux.n,label='Our Value',marker='*',s=90,color='blue')

plt.xscale('log')
plt.yscale('log')
#plt.axvline(205*(1+6.0391))
plt.xlabel("Observed Wavelength")
plt.ylabel("Flux Density (mJy)")
plt.legend()
plt.show()

