import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import interferopy.tools as iftools #(uses only numpy versions 1.x not 2.x)
import matplotlib.pyplot as plt
from uncertainties import ufloat
import utils as utils
import astropy.units as u





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


############################################################################################

#J2054-0005

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

#print(freq_ned)
#print(wave_ned)
print(flux_ned,flux_err_ned)
#print(references_ned)




#plot_sed_wave(wave=wave_ned,fluxes=flux_ned)


#Wise W1 value (Blain et al. 2013)
w1_wave = np.array([3.4]) #micrometers
w1_freq = utils.mum_to_ghz(w1_wave)
w1_mag = ufloat(18.017,0.337)
#Convert mag to flux for WISE: https://wise2.ipac.caltech.edu/docs/release/allsky/expsup/sec4_4h.html#conv2flux
w1_flux = (309.540* 10**((w1_mag)/-2.5)) * 1e3 #Flux in mJy

print("WISE1 FLux = ", w1_flux)


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


#Plot SED
plt.scatter(wave_ned,flux_ned,label='SDSS-NED')
plt.scatter(w1_wave,w1_flux.n,label='W1-Blain et al. 2013')
plt.scatter(leipski_wave,leipski_flux,label='Leipski et al. 2014')
plt.scatter(wwt_wave,wwt_flux,label='WWT')
plt.scatter(hash_wave,hash_flux.n,label='Hashimoto et al. 2018')
plt.scatter(salak_wave,salak_flux.n,label='Salak et al. 2024')

plt.scatter(sai_wave,sai_flux.n,label='Our Value',marker='*',s=120,color='blue')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("J2054-0005")
plt.legend()
plt.show()




exit()



###################################################################################################

#J2310+1855



#PAN-STARR from NED (Check the reference paper on that wesbite)
"""
freq_ned = np.array([3.97e+14, 3.45e+14, 3.15e+14])/1e9 #Convert to GHz
wave_ned = utils.ghz_to_mum(freq_ned)
flux_ned = np.array([0.00000871, 0.0000506, 8.47e-5]) * 1e3 #Convert t0 mJy
flux_err_ned = np.array([8.02e-7, 1.86e-6, 3.12e-6]) * 1e3 #Convert t0 mJy

"""

freq_ned = np.array([3.97e+14])/1e9 #Convert to GHz
wave_ned = utils.ghz_to_mum(freq_ned)
flux_ned = np.array([0.00000871]) * 1e3 #Convert t0 mJy
flux_err_ned = np.array([8.02e-7]) * 1e3 #Convert t0 mJy




#Ross & Cross, 2019 (https://arxiv.org/abs/1906.06974v1)
df = pd.read_csv("NED_photo_points/VHzQs_ZYJHK_WISE_v3.dat",delimiter='\t')
#print(df.columns)

filtered_row = df[df['desig'] == 'J2310+1855']  # Ensure '2310+1855' matches the data type and formatting in the DataFrame
#print(filtered_row[['unW1mag', 'unW1err','unW2mag', 'unW2err','w3mpro', 'w3sig']])


W1_mag = ufloat(float(filtered_row['unW1mag'].values), float(filtered_row['unW1err'].values))
W2_mag = ufloat(float(filtered_row['unW2mag'].values), float(filtered_row['unW2err'].values))
W3_mag = ufloat(float(filtered_row['w3mpro'].values), float(filtered_row['w3sig'].values))

W1_mjy = 309.540 * 10**(W1_mag/-2.5) * 1e3 #Convert to mJy
W2_mjy = 171.787 * 10**(W2_mag/-2.5) * 1e3 #Convert to mJy
W3_mjy = 31.674 * 10**(W3_mag/-2.5) * 1e3 #Convert to mJy

wise_freq = np.array([8.94e+13,6.51e+13, 2.59e+13])/1e9 #Convert to GHz
wise_wave = utils.ghz_to_mum(wise_freq) #Convert to micrometer
wise_flux = np.array([W1_mjy.n,W2_mjy.n,W3_mjy.n])
wise_flux_err = np.array([W1_mjy.s,W2_mjy.s,W3_mjy.s])

print(W1_mjy,W2_mjy,W3_mjy)


#Wang et al. 2008
wang_freq = np.array([99,250]) # IN GHz
wang_wave = utils.ghz_to_mum(wang_freq) #convert to micrometer
wang_flux = np.array([0.4, 8.29]) #in mJy
wang_flux_err = np.array([0.05,0.63])#in mJy


#Hashimoto et al. 2018 (https://arxiv.org/pdf/1811.00030)
#Our data probe dust continuum emission at the rest-frame wavelength, λ_rest, of ≈ 87 μm
hash_wave = np.array([87*(1+6.0035)]) #In micrometers
hash_freq = utils.mum_to_ghz(hash_wave)
hash_flux = ufloat(24.89,0.21) #Flux in mJy


#Tripodi et al. 2022 (https://www.aanda.org/articles/aa/pdf/2022/09/aa43920-22.pdf)
tripodi_freq = np.array([91.5, 136.627, 140.995, 153.07, 263.315, 265.369, 284.988, 289.18, 344.185, 490.787]) # IN GHz
tripodi_wave = utils.ghz_to_mum(tripodi_freq) #convert to micrometer
tripodi_flux = np.array([0.29,1.29,1.40, 1.63, 7.73, 8.81, 11.05, 11.77, 14.63, 25.31]) #in mJy
tripodi_flux_err = np.array([0.01, 0.03, 0.02, 0.06, 0.31, 0.13, 0.16, 0.12, 0.34, 0.19]) #in mJy



"""
#Shao, Y., Wang, R., Carilli, C. L., et al. 2019, ApJ, 876, 99
#Herschel SPIRE and PACS
shao_freq = np.array([131, 135, 143,147,283, 295, 338, 350, 856.549, 1199.169 , 1873.703, 2997.924]) # IN GHz
shao_wave = utils.ghz_to_mum(shao_freq) #convert to micrometer
shao_flux = np.array([1.10, 1.24,1.41,1.59, 11.60, 11.56, 14.49, 14.54, 22, 19.9, 13.2, 6.5]) #in mJy
shao_flux_err = np.array([0.05, 0.04, 0.04, 0.04, 0.1,0.1,0.21,0.21, 6.9, 6.0, 2.8, 1.2])
"""

#Shao, Y., Wang, R., Carilli, C. L., et al. 2019, ApJ, 876, 99
#Herschel SPIRE and PACS
shao_freq = np.array([856.549, 1199.169 , 1873.703, 2997.924]) # IN GHz
shao_wave = utils.ghz_to_mum(shao_freq) #convert to micrometer
shao_flux = np.array([ 22, 19.9, 13.2, 6.5]) #in mJy
shao_flux_err = np.array([ 6.9, 6.0, 2.8, 1.2])



"""
#Butler et al. 2023 (https://iopscience.iop.org/article/10.3847/1538-4357/acad03/pdf)
#λ_rest, of ≈ 119 μm
butler_wave = np.array([119 * (1+6.0035)])
butler_freq = utils.mum_to_ghz(butler_wave)
butler_flux = ufloat(4.73, 0.0016) #in mJy
"""


#Our Value
sai_freq = np.array([1461.134/(1+6.0035)]) #in GHz
sai_wave = utils.ghz_to_mum(sai_freq)
sai_flux = ufloat(5.5,0.8) #flux in mJy




plt.scatter(wave_ned,flux_ned,label='NED')
plt.scatter(wise_wave,wise_flux,label='WISE')
plt.scatter(wang_wave,wang_flux,label='Wang et al. 2008')
plt.scatter(hash_wave,hash_flux.n,label='Hashimoto et al. 2018')
plt.scatter(tripodi_wave,tripodi_flux,label='Tripodi et al. 2022')
plt.scatter(shao_wave,shao_flux,label='Shao et al. 2019')
#plt.scatter(butler_wave,butler_flux.n,label='Butler et al. 2023')

plt.scatter(sai_wave,sai_flux.n,label='Our Value',marker='*',s=120,color='blue')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("J2310-1855")
plt.legend()
plt.show()





##################################################################################################


#PSSJ2322+1944
#Last two values are from Stacey et al. 2018 (https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.5075S/abstract)

j2322_freq = np.array([1.4,90, 201.78,225,231, 353, 660, 856.54, 1875, 4300]) #IN GHz
j2322_wave = utils.ghz_to_mum(j2322_freq)
j2322_flux = np.array([9.8e-5, 0.4e-3, 5.79e-3,0.0075,0.0096,0.0225,0.075 , 79e-3, 0.0434, 0.0137]) *1e3 #Converting to mJy
j2322_flux_err = np.array([1.5e-5, 0.25e-3, 0.77e-3,0.0013, 0.0005, 0.0025, 0.019, 11e-3, 0.0084, 0.0061]) *1e3 #Converting to mJy


#Our Value
j2322_sai_freq = np.array([1461.134/(1+4.12)]) #in GHz
j2322_sai_wave = utils.ghz_to_mum(sai_freq)
j2322_sai_flux = ufloat(17.4,2.6) #flux in mJy


plt.scatter(j2322_wave,j2322_flux)
#plt.scatter(utils.ghz_to_mum(96),0.31) #Ignore this point
plt.scatter(j2322_sai_wave,j2322_sai_flux.n,label='Our Value',marker='*',s=120,color='blue')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("PSSJ2322+1944")
plt.legend()
plt.show()

