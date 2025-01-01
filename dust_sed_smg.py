from uncertainties import ufloat
from MBB import mbb_values, mbb_best_fit_flux
import numpy as np
import matplotlib.pyplot as plt
import utils
import interferopy.tools as iftools

print("MBB.py")




"""
#GN20
print("GN20")
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

print(utils.mass_kgs_solar_conversion(1e9,'solar'))

z_gn20 = 4.0553
x_stats = mbb_values(nu_obs=gn20_freq_hz,
           z=z_gn20,
           flux_obs=gn20_flux,
           flux_err=gn20_flux_err,
           dust_mass_fixed=0,
           dust_temp_fixed=0,
           dust_beta_fixed=1.95,
           nparams=2,
           params_type='mt',
           optically_thick_regime=False,
           dust_mass_limit=[1e8,1e11],
           dust_temp_limit=[25,40],
           initial_guess_values = [1e9,30],
           corner_plot=True,
           nsteps=1000,
           trace_plots=True)


wave = np.linspace(1e1,1e4,10000)
f = mbb_best_fit_flux(nu=utils.mum_to_ghz(wave)*1e9,z=z_gn20, stats=x_stats,dust_beta_default=1.95,
                      optically_thick_regime=False,output_unit_mjy=True)

dust_mass_median = x_stats['dust_mass']['median']
dust_mass_err = np.max(np.asarray([x_stats['dust_mass']['upper_1sigma'],x_stats['dust_mass']['lower_1sigma'] ]))

dust_temp_median =  x_stats['dust_temp']['median']
dust_temp_err = np.max(np.asarray([x_stats['dust_temp']['upper_1sigma'],x_stats['dust_temp']['lower_1sigma'] ]))

ir_median = x_stats['TIR x 10^13']['median']
ir_err = x_stats['TIR x 10^13']['upper_1sigma']


plt.scatter(gn20_wave_mum[:-1],gn20_flux[:-1]*1e29,color='black')
plt.scatter(my_value_gn20_wave_mum,my_value_gn20_flux,color='red',marker='*',label='Our Value',s=150)
plt.plot(wave,f,label=f'Best Fit:'
                            f'\nDust Mass = {ufloat(dust_mass_median,dust_mass_err)} L⊙'
                            f'\nDust Temp = {ufloat(dust_temp_median,dust_temp_err)} K'
                            f'\nBeta = {1.95} (Fixed)'
                            f'\nL_IR = {ufloat(ir_median,ir_err)*1e13} L⊙')

plt.xlim(1e1,1e4)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.legend()
plt.title("GN20")
plt.show()

exit()
"""

"""
print('ID141')
#https://iopscience.iop.org/article/10.1088/0004-637X/740/2/63/pdf
id141_wave = np.array([250,350,500,870,880,1200,1950,2750,3000,3290])
id141_flux = np.array([115,192,204,102,90,36,9.7,1.8,1.6,1.2])
id141_flux_err = np.array([19,30,32,8.8,5,2,0.9,0.3,0.2,0.1])
id141_freq_ghz = utils.mum_to_ghz(id141_wave)

my_value_freq = np.array([1461.134/(1+4.24)])
my_value_flux = np.array([57])
my_value_flux_err = np.array([8.6])

id141_freq_hz = np.concatenate([id141_freq_ghz,my_value_freq]) * 1e9 #Convert to Hz
id141_flux = np.concatenate([id141_flux,my_value_flux]) * 1e-29 #Convert to to W/Hz/m2
id141_flux_err = np.concatenate([id141_flux_err,my_value_flux_err]) * 1e-29  #Convert to to W/Hz/m2
id141_wave = np.concatenate([id141_wave,utils.ghz_to_mum(my_value_freq)])


plt.scatter(id141_wave,id141_flux * 1e29)
plt.xlim(1e2,5e3)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
#plt.axvline(utils.ghz_to_mum(cheng2019_freq))
plt.legend()
plt.show()


z_id141 = 4.24
x_stats = mbb_values(nu_obs=id141_freq_hz,
                     z=z_id141,
                     flux_obs=id141_flux,
                     flux_err=id141_flux_err,
                     gmf=5.8,
                     dust_mass_fixed=0,
                     dust_temp_fixed=0,
                     dust_beta_fixed=1.8,
                     nparams=2,
                     params_type='mt',
                     optically_thick_regime=False,
                     dust_mass_limit=[1e8,1e10],
                     dust_temp_limit=[30,45],
                     initial_guess_values = [1e9,38],
                     nsteps=1000,
                     flat_samples_discarded=300,
                     trace_plots=True,
                     corner_plot=True)



wave = np.linspace(1e1,1e4,10000)
f_id141 = mbb_best_fit_flux(nu=utils.mum_to_ghz(wave)*1e9,
                      z=z_id141,
                      stats=x_stats,
                      dust_beta_default=1.8,
                      optically_thick_regime=False,
                      output_unit_mjy=True)

dust_mass_median = x_stats['dust_mass']['median']
dust_mass_err = np.max(np.asarray([x_stats['dust_mass']['upper_1sigma'],x_stats['dust_mass']['lower_1sigma'] ]))

dust_temp_median =  x_stats['dust_temp']['median']
dust_temp_err = np.max(np.asarray([x_stats['dust_temp']['upper_1sigma'],x_stats['dust_temp']['lower_1sigma'] ]))

ir_median = x_stats['TIR x 10^13']['median']
ir_err = x_stats['TIR x 10^13']['upper_1sigma']

print(x_stats)

plt.scatter(id141_wave[:-1],id141_flux[:-1] * 1e29,color='black')
plt.scatter(utils.ghz_to_mum(my_value_freq), my_value_flux,color='red',marker='*',label='Our Value',s=150)
plt.plot(wave,f_id141,label=f'Best Fit:'
                            f'\nDust Mass = {ufloat(dust_mass_median,dust_mass_err)} L⊙'
                            f'\nDust Temp = {ufloat(dust_temp_median,dust_temp_err)} K'
                            f'\nBeta = {1.95} (Fixed)'
                            f'\nL_IR = {ufloat(ir_median,ir_err)*1e13} L⊙')

plt.xlim(1e2,5e3)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("ID141")
#plt.axvline(utils.ghz_to_mum(cheng2019_freq))
plt.legend()
plt.show()
"""


print("HDF850.1")

iftools.dust_cont_integrate(utils.mass_kgs_solar_conversion(0.72e9 ,'solar'),
                            35,
                            2.50,
                            True)

print("")
iftools.dust_cont_integrate(utils.mass_kgs_solar_conversion(1.11e9,'solar'),
                            30.72,
                            2.50,
                            True)

print("")
#exit()

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
cowie2017_flux_err = np.array([2.7]) #mJy
cowie2017_freq = utils.mum_to_ghz(cowie2017_wave)

#https://academic.oup.com/mnras/article/398/4/1793/982311 : Table A3, AzTEC ID 14. 1.1mm point not taken due to its low S/N
chapin2009_wave = np.array([850])
chapin2009_flux = np.array([5.88]) #mJy
chapin2009_flux_err = np.array([0.33]) #mJy
chapin2009_freq = utils.mum_to_ghz(chapin2009_wave)

#https://www.aanda.org/articles/aa/pdf/2014/02/aa22528-13.pdf : Table 3. 158 mum continuum flux density
neri2014_wave = np.array([158*(1+5.183)])
neri2014_flux = np.array([4.6]) #mJy
neri2014_flux_err = np.array([np.nan])
neri2014_freq = utils.mum_to_ghz(neri2014_wave)

#https://iopscience.iop.org/article/10.1088/0004-637X/790/1/77/pdf Table 1. GDF-2000.6 is HDF850.1
staguhn2014_wave = np.array([2000])
staguhn2014_flux = np.array([0.42]) #mJy
staguhn2014_flux_err = np.array([0.13]) #mJy
staguhn2014_freq = utils.mum_to_ghz(staguhn2014_wave)

z_hdf = 5.183
wave = np.linspace(1e1,1e4,10000)

s_inter_walter = iftools.dust_sobs(nu_obs=utils.mum_to_ghz(wave)*1e9,
                            z = 5.183,
                            mass_dust=utils.mass_kgs_solar_conversion(2.5e8,unit_of_input_mass='solar'),
                            temp_dust=35,
                            beta=2.5) *  1e26 * 1e3


plt.scatter(walter_wave[0:2],np.log10(walter_flux[0:2] * 1e-3),label='Walter+12')
plt.scatter(downes1999_wave,np.log10(downes1999_flux  * 1e-3),label='Downes+99')
plt.plot(wave,np.log10(s_inter_walter * 1e-3),label='Walter+12',color='green')

plt.xlim(1e2,1e4)
#plt.ylim(1e-4 , 10**2.5)
plt.ylim(-7 , -1.8)
plt.xscale('log')
#plt.yscale('log')
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


my_value_ghz = np.array([1461.134/(1+5.1853)])
my_value_wave = utils.ghz_to_mum(my_value_ghz)
my_value_flux = np.array([2.0])
my_value_flux_err = np.array([0.2])
print(my_value_wave)

#np.array([307.383,111.835]),



#hdf_freq_hz = np.concatenate([walter_freq[:-1],downes1999_freq,cowie2017_freq,chapin2009_freq,neri2014_freq,staguhn2014_freq,my_value_ghz])* 1e9
#hdf_flux = np.concatenate([walter_flux[:-1],downes1999_flux,cowie2017_flux,chapin2009_flux,neri2014_flux,staguhn2014_flux,my_value_flux]) * 1e-29
#hdf_flux_err = np.concatenate([walter_flux_err[:-1],downes1999_flux_err,cowie2017_flux_err,chapin2009_flux_err,neri2014_flux_err,staguhn2014_flux_err,my_value_flux_err]) * 1e-29
#hdf_wave_mum = utils.ghz_to_mum(hdf_freq_hz/1e9)

hdf_freq_hz = np.concatenate([walter_freq[:-1],downes1999_freq,cowie2017_freq,chapin2009_freq,staguhn2014_freq,my_value_ghz])* 1e9
hdf_flux = np.concatenate([walter_flux[:-1],downes1999_flux,cowie2017_flux,chapin2009_flux,staguhn2014_flux,my_value_flux]) * 1e-29
hdf_flux_err = np.concatenate([walter_flux_err[:-1],downes1999_flux_err,cowie2017_flux_err,chapin2009_flux_err,staguhn2014_flux_err,my_value_flux_err]) * 1e-29
hdf_wave_mum = utils.ghz_to_mum(hdf_freq_hz/1e9)

print('only dust_mass')
x_stats_m = mbb_values(nu_obs=hdf_freq_hz,
                     z=z_hdf,
                     gmf=1.7,
                     flux_obs=hdf_flux,
                     flux_err=hdf_flux_err,
                     dust_mass_fixed=0,
                     dust_temp_fixed=35,
                     dust_beta_fixed=2.5,
                     nparams=1,
                     optically_thick_regime=False,
                     dust_mass_limit=[1e7,1e10],
                     dust_temp_limit=[25,45],
                     initial_guess_values = [1e9,30],
                     nsteps=1000,
                     flat_samples_discarded=300,
                     trace_plots=True,
                     corner_plot=True)


print('dust_mass & dust_temp')
x_stats_mt = mbb_values(nu_obs=hdf_freq_hz,
                     z=z_hdf,
                     gmf=1.7,
                     flux_obs=hdf_flux,
                     flux_err=hdf_flux_err,
                     dust_mass_fixed=0,
                     dust_temp_fixed=35,
                     dust_beta_fixed=2.5,
                     nparams=2,
                     params_type='mt',
                     optically_thick_regime=False,
                     dust_mass_limit=[1e7,1e10],
                     dust_temp_limit=[25,45],
                     initial_guess_values = [1e9,35],
                     nsteps=1000,
                     flat_samples_discarded=300,
                     trace_plots=True,
                     corner_plot=True)

f_hdf_m = mbb_best_fit_flux(nu=utils.mum_to_ghz(wave)*1e9,
                          z=z_hdf,
                          stats=x_stats_m,
                          dust_temp_default=35,
                          dust_beta_default=2.5,
                          optically_thick_regime=False,
                          output_unit_mjy=True)


f_hdf_mt = mbb_best_fit_flux(nu=utils.mum_to_ghz(wave)*1e9,
                          z=z_hdf,
                          stats=x_stats_mt,
                          dust_beta_default=2.5,
                          optically_thick_regime=False,
                          output_unit_mjy=True)



plt.scatter(hdf_wave_mum,hdf_flux * 1e29,color='black')
plt.scatter(my_value_wave,my_value_flux,color='red',label='Our Value')
plt.plot(wave,f_hdf_m,label='only dust_mass')
plt.plot(wave,f_hdf_mt,label='dust_mass & dust_temp')

plt.plot(wave,s_inter_walter,label='Walter et al. 2012')

plt.xlim(1e2,1e4)
plt.ylim(1e-4, 10**2.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.legend()
plt.title("HDF850.1")
plt.show()

