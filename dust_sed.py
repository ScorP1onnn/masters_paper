import numpy as np
from networkx.algorithms.bipartite import color
from scipy.ndimage import label
#import pandas as pd
from scipy.optimize import curve_fit
import interferopy.tools as iftools
import matplotlib.pyplot as plt
from scipy.constants import c
import scipy.constants as const
from astropy.cosmology import FlatLambdaCDM
import astropy.units as u
import emcee


"""

#This plot is similar to the one in Stacey et al. 2018 (https://arxiv.org/pdf/1705.10530) for
#PSSJ 2322+1944. Use it as refernce

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
def plot_sed(freqs,fluxes,freqs_extra = [],fluxes_extra = []):
    plt.scatter(freqs, fluxes)

    if len(freqs_extra)!=0 and len(fluxes_extra)!=0:
        plt.scatter(freqs_extra, fluxes_extra,label='Extra')

    plt.scatter(285.928, 17.4e-3, label='Our Value')

    # Set x-axis to logarithmic scale
    plt.xscale('log')
    plt.yscale('log')
    # Add labels and title
    plt.xlabel("Frequency")
    plt.ylabel("Flux Density (Jy)")
    plt.legend()

    plt.xlim(1e0, 1e4)
    plt.gca().invert_xaxis()

    plt.ylim(1e-6, 1e0)


    # Show plot
    plt.show()

"""fluxes = np.asarray([0.0137, 0.0434, 0.075, 0.0225 , 0.024, 0.0096, 0.0075, 0.00064, 9e-5,9.8e-5]) #Jy
flux_err = np.asarray([0.0061, 0.0084, 0.019, 0.0025, 0.002, 0.0005, 0.0013, 0,0,1.5e-5]) # Jy
freqs = np.asarray([4300, 1875, 660, 353, 353, 231, 225, 90, 5, 1.4]) #Ghz"""

#These Values are from  Stacey et al. 2018 (https://arxiv.org/pdf/1705.10530)
fluxes = np.asarray([0.075, 0.0225 , 0.024, 0.0096, 0.0075, 0.00064,9e-5,9.8e-5]) #Jy
flux_err = np.asarray([ 0.019, 0.0025, 0.002, 0.0005, 0.0013,0,0, 1.5e-5]) # Jy
freqs = np.asarray([ 660, 353, 353, 231, 225,  90,5, 1.4 ]) #Ghz

wave = (c/(freqs*1e9))*1e6


#These are from NED
data = pd.read_csv(r"C:\Users\kolup\Downloads\table_photandseds.csv")
df = data.iloc[:13]

df_1 = df[df['Frequency']<1500e9]




plot_sed(df_1['Frequency']/1e9,df_1['Flux Density'], freqs_extra=freqs,fluxes_extra=fluxes)









# Create a plot



print(df_1[['Frequency', 'Flux Density', 'Upper limit of uncertainty']])
print()


fluxes = np.asarray([0.075, 0.0225 , 0.024, 0.0096, 0.0075, 0.00064, 0.0790,0.0096]) #Jy
flux_err = np.asarray([ 0.019, 0.0025, 0.002, 0.0005, 0.0013,0.00064*(10/100), 0.0110, 0.0005]) # Jy
freqs = np.asarray([ 660, 353, 353, 231, 225, 90,8.570000e+11/1e9, 2.500000e+11/1e9 ]) #Ghz

plot_sed(freqs,fluxes)


fluxes_with_our_value = np.asarray([0.075, 0.0225 , 0.024, 0.0096, 0.0075, 0.00064, 0.0790,0.0096, 17.4e-3]) #Jy
flux_err_with_our_value = np.asarray([ 0.019, 0.0025, 0.002, 0.0005, 0.0013,0.00064*(10/100), 0.0110, 0.0005, 2.6e-3]) # Jy
freqs_with_our_value = np.asarray([ 660, 353, 353, 231, 225, 90,8.570000e+11/1e9, 2.500000e+11/1e9, 285.928 ]) #Ghz



fluxes = fluxes * 1e-26  # to W/Hz/m2 (since Flux Density is in jansky)
fluxes_err =flux_err * 1e-26  # to W/Hz/m2 (since Flux Density is in jansky)
freqs = freqs * 1e9

"""
fluxes_with_our_value = fluxes_with_our_value * 1e-26  # to W/Hz/m2 (since Flux Density is in jansky)
fluxes_err_with_our_value =flux_err_with_our_value * 1e-26  # to W/Hz/m2 (since Flux Density is in jansky)
freqs_with_our_value = freqs_with_our_value * 1e9
"""
z = 4.12



dust_mass = 1e37  # in kilograms
dust_temp = 47  # T_dust in Kelvins
dust_beta = 1.6  # modified black body exponent

popt, pcov = curve_fit(lambda freqs, dust_mass: iftools.dust_sobs(freqs, z, dust_mass, dust_temp, dust_beta),
                        freqs, fluxes, p0=(dust_mass), sigma=fluxes_err, absolute_sigma=True)

dust_mass = popt[0]
dust_mass_err = np.diagonal(pcov)[0]


print("Dust Mass:",dust_mass,dust_mass_err)


theta = dust_mass
plt.scatter(freqs/1e9,fluxes * 1e26)
xxx = np.linspace(np.min(freqs) / 5, np.max(freqs) * 5, 1000)/1e9 #GHz
r = iftools.dust_sobs(xxx * 1e9,z, dust_mass, dust_temp, dust_beta ) * 1e26
plt.plot(xxx,r)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e0, 1e4)
plt.gca().invert_xaxis()
plt.ylim(1e-6, 1e0)
plt.show()


lum_fir, lum_tir, sfr_K98, sfr_k12 = iftools.dust_cont_integrate(dust_mass=theta, dust_temp=dust_temp,dust_beta=dust_beta, print_to_console=True)



print("Next")
print("")
#exit()




fluxes_with_our_value = fluxes_with_our_value * 1e-26  # to W/Hz/m2 (since Flux Density is in jansky)
fluxes_err_with_our_value =flux_err_with_our_value * 1e-26  # to W/Hz/m2 (since Flux Density is in jansky)
freqs_with_our_value = freqs_with_our_value * 1e9

dust_mass = 1e37  # in kilograms
dust_temp = 47  # T_dust in Kelvins
dust_beta = 1.6  # modified black body exponent


popt, pcov = curve_fit(lambda freqs_with_our_value, dust_mass: iftools.dust_sobs(freqs_with_our_value, z, dust_mass, dust_temp, dust_beta),
                        freqs_with_our_value, fluxes_with_our_value, p0=(dust_mass), sigma=fluxes_err_with_our_value, absolute_sigma=True)


dust_mass_with_our_value = popt[0]
dust_mass_err_with_our_value = np.diagonal(pcov)[0]

print("Dust Mass:",dust_mass_with_our_value,dust_mass_err_with_our_value)



plt.scatter(freqs_with_our_value/1e9,fluxes_with_our_value * 1e26,color='green',label='Our Value')
plt.scatter(freqs/1e9,fluxes * 1e26,color='blue')
xxx = np.linspace(np.min(freqs) / 5, np.max(freqs) * 5, 1000)/1e9 #GHz
r = iftools.dust_sobs(xxx * 1e9,z, dust_mass_with_our_value, dust_temp, dust_beta ) * 1e26
plt.plot(xxx,r)
plt.xscale('log')
plt.yscale('log')
plt.xlim(1e0, 1e4)
plt.gca().invert_xaxis()
plt.ylim(1e-6, 1e0)
plt.legend()
plt.show()

lum_fir, lum_tir, sfr_K98, sfr_k12 = iftools.dust_cont_integrate(dust_mass=dust_mass_with_our_value, dust_temp=dust_temp,dust_beta=dust_beta, print_to_console=True)

print(31.0/5.3)
