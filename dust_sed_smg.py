from cProfile import label
from csv import unix_dialect

import numpy as np
import pandas as pd
import pylab as pl
from scipy.optimize import curve_fit
import interferopy.tools as iftools #(uses only numpy versions 1.x not 2.x)
import matplotlib.pyplot as plt
from uncertainties import ufloat
import utils as utils
import astropy.units as u
from lmfit import Model

from utils import mum_to_ghz

"""
walter_freq = np.array([307.383,111.835,37.286])
walter_flux = np.array([6.8,0.13,30e-3])
walter_flux_err = np.array([0.8,0.3,np.nan])
walter_wave = utils.ghz_to_mum(walter_freq)

downes1999_wave = np.array([1300])
downes1999_flux = np.array([2.2])
downes1999_flux_err = np.array([0.3])
downes1999_freq = utils.mum_to_ghz(downes1999_wave)

wave = np.linspace(1e1,1e4,10000)
s_inter = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 5.183,
                            mass_dust=utils.mass_kgs_solar_conversion(2.75e8,'solar'),
                            temp_dust=35,
                            beta=1.6) *  1e26 * 1e3

plt.scatter(walter_wave,walter_flux)
plt.scatter(downes1999_wave,downes1999_flux)
plt.plot(wave,s_inter)

plt.xlim(1e2,1e4)
plt.ylim(1e-4, 1e2)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.show()
"""

#https://arxiv.org/pdf/1206.2641
walter_freq = np.array([307.383,111.835,37.286])
walter_flux = np.array([6.8,0.13,30e-3])
walter_flux_err = np.array([0.8,0.3,np.nan])
walter_wave = utils.ghz_to_mum(walter_freq)

downes1999_wave = np.array([1300])
downes1999_flux = np.array([2.2])
downes1999_flux_err = np.array([0.3])
downes1999_freq = utils.mum_to_ghz(downes1999_wave)

#https://iopscience.iop.org/article/10.3847/1538-4357/aa60bb/pdf : Table 6
cowie2017_wave = np.array([450])
cowie2017_flux = np.array([13]) #mJy
cowie2017_flux_err = np.array([2.7])
cowie2017_freq = utils.mum_to_ghz(cowie2017_wave)

#https://academic.oup.com/mnras/article/398/4/1793/982311 : Table A3, AzTEC ID 14. 1.1mm point not taken due to its low S/N
chapin2009_wave = np.array([850])
chapin2009_flux = np.array([5.88])
chapin2009_flux_err = np.array([0.33])
chapin2009_freq = utils.mum_to_ghz(chapin2009_wave)

#https://www.aanda.org/articles/aa/pdf/2014/02/aa22528-13.pdf : Table 3. 158 mum continuum flux density
neri2014_wave = np.array([158*(1+5.183)])
neri2014_flux = np.array([4.6])
neri2014_flux_err = np.array([np.nan])
neri2014_freq = utils.mum_to_ghz(neri2014_wave)

#https://iopscience.iop.org/article/10.1088/0004-637X/790/1/77/pdf Table 1. GDF-2000.6 is HDF850.1
staguhn2014_wave = np.array([2000])
staguhn2014_flux = np.array([0.42])
staguhn2014_flux_err = np.array([0.13])
staguhn2014_freq = utils.mum_to_ghz(staguhn2014_wave)


hdf_freq_ghz = np.concatenate([np.array([307.383,111.835]),downes1999_freq])* 1e9
hdf_flux = np.array([6.8,0.13,2.2]) * 1e-29
hdf_flux_err = np.array([0.8,0.3,0.3]) * 1e-29

z = 5.183

dust_mass = utils.mass_kgs_solar_conversion(2.75e8,unit_of_input_mass='solar') # in kilograms
dust_temp = 35  # T_dust in Kelvins
dust_beta = 2.5  # modified black body exponent

dust_mass_err = 0
dust_temp_err = 0
dust_beta_err = 0

popt, pcov = curve_fit(lambda hdf_freq_ghz, dust_mass: iftools.dust_sobs(hdf_freq_ghz, z, dust_mass, dust_temp, dust_beta),hdf_freq_ghz, hdf_flux, p0=(dust_mass), sigma=hdf_flux_err, absolute_sigma=True)
dust_mass_out = popt[0]
dust_mass_err_out = np.diagonal(pcov)[0]
print(dust_mass_out)


popt, pcov = curve_fit(lambda hdf_freq_ghz, dust_temp: iftools.dust_sobs(hdf_freq_ghz, z, dust_mass, dust_temp, dust_beta),hdf_freq_ghz, hdf_flux, p0=(dust_temp), sigma=hdf_flux_err, absolute_sigma=True)
dust_temp_out = popt[0]
dust_temp_err_out = np.diagonal(pcov)[0]
print(dust_temp_out)

"""

popt, pcov = curve_fit(lambda hdf_freq_ghz, dust_beta: iftools.dust_sobs(hdf_freq_ghz, z, dust_mass, dust_temp, dust_beta),hdf_freq_ghz, hdf_flux, p0=(dust_beta), sigma=hdf_flux_err, absolute_sigma=True)
dust_beta_out = popt[0]
dust_beta_err_out = np.diagonal(pcov)[0]
print(dust_beta_out)
"""




wave = np.linspace(1e1,1e4,10000)
s_inter_mine_mass = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 5.183,
                            mass_dust=dust_mass_out,
                            temp_dust=35,
                            beta=2.5) *  1e26 * 1e3

s_inter_mine_temp = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 5.183,
                            mass_dust=utils.mass_kgs_solar_conversion(2.5e8,unit_of_input_mass='solar'),
                            temp_dust=dust_temp_out,
                            beta=2.5) *  1e26 * 1e3


s_inter_walter = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 5.183,
                            mass_dust=utils.mass_kgs_solar_conversion(2.5e8,unit_of_input_mass='solar'),
                            temp_dust=35,
                            beta=2.5) *  1e26 * 1e3



plt.scatter(walter_wave[0:2],walter_flux[0:2],label='Walter+12')
plt.scatter(downes1999_wave,downes1999_flux,label='Downes+99')
plt.plot(wave,s_inter_mine_mass,label=f'My FIT (Dust Mass = {np.round(utils.mass_kgs_solar_conversion(dust_mass_out,"kgs")/1e8,2)}x 10^8 M_solar)',color='blue')
plt.plot(wave,s_inter_mine_temp,label=f'My FIT (Dust Temp = {np.round(dust_temp_out,1)} K)',color='grey',ls='--')
plt.plot(wave,s_inter_walter,label='Walter+12',color='green')

plt.xlim(1e2,1e4)
plt.ylim(1e-4, 10**2.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("HDF850.1")
plt.legend()
plt.show()



plt.scatter(walter_wave[0:2],walter_flux[0:2],label='Walter+12')
plt.scatter(downes1999_wave,downes1999_flux,label='Downes+99')
plt.scatter(cowie2017_wave,cowie2017_flux,label='Cowie+17')
plt.scatter(chapin2009_wave,chapin2009_flux,label='Chapin+09')
plt.scatter(neri2014_wave,neri2014_flux,label='Neri+14')
plt.scatter(staguhn2014_wave,staguhn2014_flux,label='Staguhn+14')

plt.xlim(1e2,1e4)
plt.ylim(1e-4, 10**2.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.legend()
plt.show()

"""
print(dust_mass_err_out)
iftools.dust_cont_integrate(dust_mass_out, dust_temp, dust_beta,print_to_console=True)


gm = Model(iftools.dust_sobs)

params = gm.make_params(
    z=5.183,  # Example value for z
    mass_dust=1e39,  # Initial guess for dust_mass
    temp_dust=35.0,  # Example value for dust_temp
    beta=2.5  # Example value for beta
)

# Fix the parameters you don't want to vary
params["z"].vary = False
params["temp_dust"].vary = False
params["beta"].vary = False

result = gm.fit(hdf_flux, params, nu_obs=hdf_freq_ghz)
print(result.fit_report())


s_inter_mine_mass_lmfit = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 5.183,
                            mass_dust=result.params["mass_dust"].value,
                            temp_dust=35,
                            beta=2.5) *  1e26 * 1e3


plt.scatter(walter_wave[0:2],walter_flux[0:2],color='black')
plt.scatter(downes1999_wave,downes1999_flux,color='black')
plt.plot(wave, s_inter_mine_mass_lmfit, label=f'lmfit: (Dust Mass = {np.round(utils.mass_kgs_solar_conversion(result.params["mass_dust"].value,"kgs")/1e8,2)}x 10^8 M_solar)')
plt.plot(wave,s_inter_mine_mass,label=f'Curve FIT (Dust Mass = {np.round(utils.mass_kgs_solar_conversion(dust_mass_out,"kgs")/1e8,2)}x 10^8 M_solar)')
plt.plot(wave,s_inter_walter,label='Walter+12')

plt.xlim(1e2,1e4)
plt.ylim(1e-4, 10**2.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.legend()
plt.show()

"""

#########################################################################################33

my_value_ghz = np.array([1461.134/(1+5.1853)])
my_value_wave = utils.ghz_to_mum(my_value_ghz)
my_value_flux = np.array([2.0])
my_value_flux_err = np.array([0.2])
print(my_value_wave)

#np.array([307.383,111.835]),

hdf_freq_ghz = np.concatenate([walter_freq[:-1],downes1999_freq,cowie2017_freq,chapin2009_freq,neri2014_freq,staguhn2014_freq,my_value_ghz])* 1e9
#hdf_freq_ghz = np.concatenate([hdf_freq_ghz,my_value_ghz])
hdf_flux = np.concatenate([walter_flux[:-1],downes1999_flux,cowie2017_flux,chapin2009_flux,neri2014_flux,staguhn2014_flux,my_value_flux]) * 1e-29
hdf_flux_err = np.concatenate([walter_flux_err[:-1],downes1999_flux_err,cowie2017_flux_err,chapin2009_flux_err,neri2014_flux_err,staguhn2014_flux_err,my_value_flux_err]) * 1e-29
hdf_wave_mum = utils.ghz_to_mum(hdf_freq_ghz/1e9)


weights = 1 /hdf_flux_err

gm = Model(iftools.dust_sobs)

params = gm.make_params(
    z=5.183,  # Example value for z
    mass_dust=utils.mass_kgs_solar_conversion(2.5e8,unit_of_input_mass='solar'),  # Initial guess for dust_mass
    temp_dust=35.0,  # Example value for dust_temp
    beta=2.5  # Example value for beta
)

# Fix the parameters you don't want to vary
params["z"].vary = False
params['mass_dust'].vary = True
params["temp_dust"].vary = True
params["beta"].vary = False
result_my_value_vary_mass = gm.fit(hdf_flux, params, nu_obs=hdf_freq_ghz,cmb_contrast=True,cmb_heating=True,weights=weights,nan_policy='omit')
print(result_my_value_vary_mass.fit_report())
dust_mass_my_value = utils.mass_kgs_solar_conversion(ufloat(result_my_value_vary_mass.params["mass_dust"].value,
                                                            result_my_value_vary_mass.params["mass_dust"].stderr),unit_of_input_mass='kg')


params["z"].vary = False
params['mass_dust'].vary = True
params["temp_dust"].vary = True
params["beta"].vary = True
params['beta'].min=2
result_my_value_vary_all = gm.fit(hdf_flux, params, nu_obs=hdf_freq_ghz,cmb_contrast=True,cmb_heating=True,weights=weights,nan_policy='omit',
                                  )
print(result_my_value_vary_all.fit_report())
dust_mass_vary_all = utils.mass_kgs_solar_conversion(ufloat(result_my_value_vary_all.params["mass_dust"].value,
                                                            result_my_value_vary_all.params["mass_dust"].stderr),unit_of_input_mass='kg')

"""
Note: result_my_value_vary_all gives a fit where the error on the dust mass is larger than the mean dust mass. Rejecting this and using the fit where beta=2.5
"""



s_my_value_vary_mass = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 5.183,
                            mass_dust=result_my_value_vary_mass.params["mass_dust"].value,
                            temp_dust=result_my_value_vary_mass.params["temp_dust"].value,
                            beta=result_my_value_vary_mass.params["beta"].value) *  1e26 * 1e3

s_my_value_vary_all = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 5.183,
                            mass_dust=result_my_value_vary_all.params["mass_dust"].value,
                            temp_dust=result_my_value_vary_all.params["temp_dust"].value,
                            beta=result_my_value_vary_all.params["beta"].value) *  1e26 * 1e3


#result_my_value_vary_mass.params["temp_dust"].stderr
utils.dust_integrated_luminsoity(dust_mass=utils.mass_kgs_solar_conversion(dust_mass_my_value,'solar'),
                                 dust_temp=ufloat(result_my_value_vary_mass.params["temp_dust"].value,result_my_value_vary_mass.params["temp_dust"].stderr),
                                 dust_beta=2.5,
                                 lum='both',
                                 gmf=1.7,
                                 print_to_console=True)


utils.dust_integrated_luminsoity(dust_mass=utils.mass_kgs_solar_conversion(dust_mass_my_value,'solar'),
                                 dust_temp=ufloat(35,5),
                                 dust_beta=2.5,
                                 lum='both',
                                 gmf=1.7,
                                 print_to_console=True)



print("")
iftools.dust_cont_integrate(dust_mass=utils.mass_kgs_solar_conversion(dust_mass_my_value.n,'solar'),
                            dust_temp=result_my_value_vary_mass.params["temp_dust"].value,
                            dust_beta=2.5,
                            print_to_console=True)



"""
utils.dust_integrated_luminsoity(dust_mass=utils.mass_kgs_solar_conversion(dust_mass_vary_all,'solar'),
                                 dust_temp=ufloat(result_my_value_vary_all.params["temp_dust"].value,result_my_value_vary_all.params["temp_dust"].stderr),
                                 dust_beta=ufloat(result_my_value_vary_all.params["beta"].value,result_my_value_vary_all.params["beta"].stderr),
                                 lum='both',
                                 gmf=1.7,
                                 print_to_console=True)

"""

s = 1#/(1+5.183)
plt.scatter(hdf_wave_mum*s,hdf_flux * 1e29,color='black')
plt.scatter(my_value_wave*s,my_value_flux,color='red',label='Our Value')
plt.plot(wave*s, s_my_value_vary_mass,label='Best Fit: '
                                f'\nDust Mass = {dust_mass_my_value} $L_\odot$'
                                          f'\nDust Temp = {ufloat(result_my_value_vary_mass.params["temp_dust"].value,result_my_value_vary_mass.params["temp_dust"].stderr)} K'
                                          f'\nBeta = {ufloat(result_my_value_vary_mass.params["beta"].value,result_my_value_vary_mass.params["beta"].stderr)} (FIXED)')
"""
plt.plot(wave, s_my_value_vary_all,label='Best Fit: '
                                f'\nDust Mass = {dust_mass_vary_all} $L_\odot$'
                                          f'\nDust Temp = {ufloat(result_my_value_vary_all.params["temp_dust"].value,result_my_value_vary_all.params["temp_dust"].stderr)} K'
                                          f'\nBeta = {ufloat(result_my_value_vary_all.params["beta"].value,result_my_value_vary_all.params["beta"].stderr)}')

"""
plt.plot(wave*s,s_inter_walter,label='Walter et al. 2012')

plt.xlim(1e2,1e4*s)
plt.ylim(1e-4, 10**2.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.legend()
plt.title("HDF850.1")
plt.show()


exit()
########################################################################################################################

print()
print()

#ID141

#https://iopscience.iop.org/article/10.1088/0004-637X/740/2/63/pdf
id141_wave = np.array([250,350,500,870,880,1200,1950,2750,3000,3290])
id141_flux = np.array([115,192,204,102,90,36,9.7,1.8,1.6,1.2])
id141_flux_err = np.array([19,30,32,8.8,5,2,0.9,0.3,0.2,0.1])
id141_freq_ghz = utils.mum_to_ghz(id141_wave)

"""
cheng2019_freq = np.array([1461.134/(1+4.24)])
cheng2019_flux = np.array([52])
cheng2019_flux_err = np.array([5.2])


id141_freq_ghz = np.concatenate([id141_freq_ghz,cheng2019_freq]) * 1e9 #Convert to Hz
id141_flux = np.concatenate([id141_flux,cheng2019_flux]) * 1e-29 #Convert to to W/Hz/m2
id141_flux_err = np.concatenate([id141_flux_err,cheng2019_flux_err]) * 1e-29  #Convert to to W/Hz/m2
id141_wave = np.concatenate([id141_wave,utils.ghz_to_mum(cheng2019_freq)])"""


my_value_freq = np.array([1461.134/(1+4.24)])
my_value_flux = np.array([57])
my_value_flux_err = np.array([8.6])

id141_freq_ghz = np.concatenate([id141_freq_ghz,my_value_freq]) * 1e9 #Convert to Hz
id141_flux = np.concatenate([id141_flux,my_value_flux]) * 1e-29 #Convert to to W/Hz/m2
id141_flux_err = np.concatenate([id141_flux_err,my_value_flux_err]) * 1e-29  #Convert to to W/Hz/m2
id141_wave = np.concatenate([id141_wave,utils.ghz_to_mum(my_value_freq)])

"""
plt.scatter(id141_wave,id141_flux)
plt.xlim(1e2,5e3)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.axvline(utils.ghz_to_mum(cheng2019_freq))
plt.legend()
plt.show()
"""

gm = Model(iftools.dust_sobs)

params = gm.make_params(
    z=4.24,  # Example value for z
    mass_dust=1e39,  # Initial guess for dust_mass
    temp_dust=35.0,  # Example value for dust_temp
    beta=1.8  # Example value for beta
)


# Fix the parameters you don't want to vary
params["z"].vary = False
#params["temp_dust"].vary = False
params["beta"].vary = False

# Include weights (1/errors)
weights = 1/id141_flux_err


result_id141 = gm.fit(id141_flux, params, nu_obs=id141_freq_ghz,cmb_contrast=True,cmb_heating=True, weights=weights)
print(result_id141.fit_report())




wave = np.linspace(1e1,1e4,10000)
s_id141 = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 4.24,
                            mass_dust=result_id141.params["mass_dust"].value,
                            temp_dust=result_id141.params["temp_dust"].value,
                            beta=1.8) *  1e26 * 1e3




dust_mass_solar_id141 = ufloat(utils.mass_kgs_solar_conversion(result_id141.params["mass_dust"].value,unit_of_input_mass='kg'),
                               utils.mass_kgs_solar_conversion(result_id141.params["mass_dust"].stderr,unit_of_input_mass='kg'))

dust_temp_id141 = ufloat(result_id141.params["temp_dust"].value,result_id141.params["temp_dust"].stderr)



print("ID141 log(Dust mass) = ",np.round(np.log10(dust_mass_solar_id141.n),2),"+/-",np.round(utils.log10_of_error(dust_mass_solar_id141.n,dust_mass_solar_id141.s),2))
print("ID141 Dust Temperature (K) = ",np.round(result_id141.params["temp_dust"].value,2), "+/-", np.round(result_id141.params["temp_dust"].stderr,2))
print()
#iftools.dust_cont_integrate(dust_mass=result_id141.params["mass_dust"].value,dust_temp=result_id141.params["temp_dust"].value,dust_beta = 1.8,print_to_console=True)


mu_tir_id141,tir,_,_=utils.dust_integrated_luminsoity(dust_mass=ufloat(result_id141.params["mass_dust"].value,result_id141.params["mass_dust"].stderr),
                                 dust_temp=ufloat(result_id141.params["temp_dust"].value,result_id141.params["temp_dust"].stderr),
                                 dust_beta=1.8,
                                 lum='both',
                                 gmf=5.8,
                                 print_to_console=True)

print(np.log10(71.2e12))
#ID141 Values are well within the errors


plt.scatter(id141_wave,id141_flux * 1e29,color='black')
plt.scatter(utils.ghz_to_mum(my_value_freq), my_value_flux,color='red',label='Our Value')
plt.plot(wave,s_id141,label=f'Best Fit:'
                            f'\nDust Mass = {dust_mass_solar_id141} L⊙'
                            f'\nDust Temp = {dust_temp_id141} K'
                            f'\nBeta = {1.8} (Fixed)'
                            f'\nμL_IR = {mu_tir_id141}  L⊙')

plt.xlim(1e2,5e3)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.legend()
plt.title("ID141 Fit")
plt.show()

#eit()
########################################################################################################################

#Literature values for GN20

gn20_wave_mum = np.array([100,160,250,350,500,850,880,1100,2200,3300,3050,1860])
gn20_flux = np.array([0.7,5.4,18.6,41.3,39.7,20.3,16,10.7,0.9,0.33,0.36,2.8])
gn20_flux_err = np.array([0.4,1.0,2.7,5.2,6.1,2.0,1.0,1.0,0.15,0.06,0.05,0.13])
gn20_freq = utils.mum_to_ghz(gn20_wave_mum)

my_value_gn20_freq = np.array([1461.134/(1+4.0553)])
my_value_gn20_flux = np.array([16.2])
my_value_gn20_flux_err = np.array([2.5])
my_value_gn20_wave_mum = utils.ghz_to_mum(my_value_gn20_freq)

plt.scatter(gn20_wave_mum,gn20_flux)
plt.scatter(my_value_gn20_wave_mum,my_value_gn20_flux,color='red')

plt.xlim(1e1,1e4)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.legend()
plt.title("GN20")
plt.show()


#Optically Thin Scenario

gn20_freq_hz = np.concatenate([gn20_freq,my_value_gn20_freq]) * 1e9 #Convert to Hz
gn20_flux = np.concatenate([gn20_flux,my_value_gn20_flux]) * 1e-29 #Convert to to W/Hz/m2
gn20_flux_err = np.concatenate([gn20_flux_err,my_value_gn20_flux_err]) * 1e-29 #Convert to to W/Hz/m2
gn20_wave_mum = np.concatenate([gn20_wave_mum,my_value_gn20_wave_mum])

gm_gn20 = Model(iftools.dust_sobs)

params = gm.make_params(
    z=4.0553,  # Example value for z
    mass_dust=1e39,  # Initial guess for dust_mass
    temp_dust=35.0,  # Example value for dust_temp
    beta=1.95  # Example value for beta
)

# Fix the parameters you don't want to vary
params["z"].vary = False
#params["temp_dust"].vary = False
params["beta"].vary = False

# Include weights (1/errors)
weights = 1/gn20_flux_err


result_gn20 = gm.fit(gn20_flux, params, nu_obs=gn20_freq_hz,cmb_contrast=True,cmb_heating=True, weights=weights)
print(result_gn20.fit_report())

dust_mass_solar_gn20 = ufloat(utils.mass_kgs_solar_conversion(result_gn20.params["mass_dust"].value,unit_of_input_mass='kg'),
                               utils.mass_kgs_solar_conversion(result_gn20.params["mass_dust"].stderr,unit_of_input_mass='kg'))

dust_temp_gn20 = ufloat(result_gn20.params["temp_dust"].value,result_gn20.params["temp_dust"].stderr)

wave = np.linspace(1e1,1e4,10000)
s_gn20 = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 4.0553,
                            mass_dust=result_gn20.params["mass_dust"].value,
                            temp_dust=result_gn20.params["temp_dust"].value,
                            beta=result_gn20.params["beta"].value) *  1e26 * 1e3 #mJy


tir_gn20,fir= utils.dust_integrated_luminsoity(dust_mass=utils.mass_kgs_solar_conversion(dust_mass_solar_gn20,unit_of_input_mass='solar'),
                                          dust_temp=dust_temp_gn20,
                                          dust_beta=result_gn20.params["beta"].value,
                                          lum='both',
                                          print_to_console=True)



#Cortzen et al. 2020 (https://www.aanda.org/articles/aa/pdf/2020/02/aa37217-19.pdf)
#Optical thin regime
print(f"Dust Temp Cortzen+20 = {ufloat(33,2)}")
print(f"Dust Mass Cortzen+20 = {10**ufloat(9.59,0.1)}")
print(f"L_IR Cortzen+20 = {10**ufloat(13.15,0.04)}")


plt.scatter(gn20_wave_mum,gn20_flux*1e29,color='black')
plt.scatter(my_value_gn20_wave_mum,my_value_gn20_flux,color='red',label='Our Value')
plt.plot(wave,s_gn20,label=f'Best Fit:'
                            f'\nDust Mass = {dust_mass_solar_gn20 } L⊙'
                            f'\nDust Temp = {dust_temp_gn20} K'
                            f'\nBeta = {1.95} (Fixed)'
                            f'\nμL_IR = {tir_gn20}  L⊙')

plt.xlim(1e1,1e4)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.legend()
plt.title("GN20 Fit")
plt.show()




exit()


print("GN20 log(Dust mass) = ",np.round(np.log10(dust_mass_solar_gn20.n),2),"+/-",np.round(utils.log10_of_error(dust_mass_solar_gn20.n,dust_mass_solar_gn20.s),2))
print("GN20 Dust Temperature (K) = ",np.round(result_gn20.params["temp_dust"].value,2), "+/-", np.round(result_gn20.params["temp_dust"].stderr,2))
print()

iftools.dust_cont_integrate(dust_mass=result_gn20.params["mass_dust"].value,dust_temp=result_gn20.params["temp_dust"].value,
                            dust_beta = 1.9,print_to_console=True)

print("")

#GN20 Values are well within the errors