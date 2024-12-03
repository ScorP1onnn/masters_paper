import os
from cProfile import label
from fileinput import filename

import matplotlib.pyplot as plt
import numpy as np
import interferopy.tools as tools
import astropy.units as u
from matplotlib.pyplot import tight_layout

import utils as utils
from utils import ghz_to_mum


def alma_filter(file_name,rest_freq=0.,z=0.,tuning_freq=[],middle_freq=0.,alma_bandwidth=7.5,bin=30.,plot=False):


    if middle_freq==0:
        tuning_freq = np.asarray(tuning_freq)
        central_freq = np.mean(tuning_freq)
    else:
        central_freq = middle_freq


    if rest_freq!=0 and z!=0:
        ref_freq = rest_freq / (1 + z)
    else:
        ref_freq = middle_freq

    #ref_freq = rest_freq/(1+z)
    freq_Step = tools.kms2ghz(bin,ref_freq)

    a = central_freq - ((alma_bandwidth / 2) - freq_Step)
    b = central_freq + ((alma_bandwidth / 2) + freq_Step)

    array_GHz = np.arange(a, b, freq_Step) * u.GHz
    array_angstroms = array_GHz.to(u.angstrom, equivalencies=u.spectral())

    transmission = np.zeros(len(array_angstroms))
    transmission[1:-1] = transmission[1:-1] + 1.000

    #print(array_GHz)
    #print(array_angstroms[0].value)
    #print(ghz_to_mum(array_GHz.value))






    #print(transmission)
    transmission_split = np.array_split(transmission[1:-1],60)
    array_angstroms_split = np.array_split(array_angstroms[1:-1].value,60)

    transmission_avg = np.array([np.mean(bin) for bin in transmission_split])
    array_angstroms_avg = np.array([np.mean(bin) for bin in array_angstroms_split])

    #for i in range(len(a)):
        #print(len(a[i]),len(b[i]))





    transmission_avg = np.insert(transmission_avg,0,transmission[0])
    transmission_avg = np.insert(transmission_avg,len(transmission_avg),transmission[len(transmission) - 1])

    array_angstroms_avg = np.insert(array_angstroms_avg, 0, array_angstroms[0].value)
    array_angstroms_avg = np.insert(array_angstroms_avg, len(array_angstroms_avg), array_angstroms[len(array_angstroms) - 1].value)


    transmission = transmission_avg
    array_angstroms = array_angstroms_avg
    array_GHz = (array_angstroms * u.angstrom).to(u.GHz, equivalencies=u.spectral())



    data = np.column_stack((array_angstroms[::-1], transmission))
    comments = (
        f"{file_name}",
        "photon",
        "boxcar shape filter"
    )
    header = "\n".join(comments)

    desktop_path = os.path.expanduser('~') + '/Desktop' + '/alma_iram_filters'
    file_path = os.path.join(desktop_path, file_name + '.dat')

    np.savetxt(file_path,data, fmt="%.6f", header=header, comments="# ")


    #Wavelength in Observed Frame (or Sky frame)
    #The vertical line is a bit off from the actual GHz axis but is exactly correct in the wavlength axis (i.e. when
    #you convert the GHz value to angstrom). The only thing that matters is that the value is within the profile
    if plot == True:

        fig,ax = plt.subplots(1,1,tight_layout=True)

        ax.plot(array_angstroms, transmission)
        ax.axvline(utils.ghz_to_mum(central_freq) *1e4, color='black',ls='--',label=f'freq_central = {central_freq} GHz')
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel("Transmission")
        plt.legend()

        ax_freq = ax.twiny()
        ax_freq.plot(array_GHz,transmission)
        ax_freq.invert_xaxis()
        ax_freq.set_xlabel('Observed Frequency [GHz]')

        plt.title(f"{file_name}")
        plt.show()

        """
        plt.plot(array_angstroms, transmission)
        plt.axvline(utils.ghz_to_mum(central_freq) *1e4, color='black',ls='--',label=f'freq_central = {central_freq} GHz')
        plt.xlabel(r"Wavelength [$\AA$]")
        plt.ylabel("Transmission")
        plt.legend()
        plt.show()"""







"""
#Hashimoto et al. 2019
restfreq = 3393.006244
z_j2054=6.0391
file = 'hashimoto_j2054'
alma_filter(rest_freq=restfreq,z=z_j2054,file_name=file,tuning_freq=[480.71,483.68],plot=True)

exit()

restfreq = 3393.006244
z_j2310=6.0035
file = 'hashimoto_j2310'
alma_filter(rest_freq=restfreq,z=z_j2310,file_name=file,tuning_freq=[483.19,486.31],plot=True)


#Sala et al. 2024
restfreq = 2437.3370569 #for rest wavelength of 123microns
z_j2054=6.0391
file_salak = 'salak_j2054'
alma_filter(rest_freq=restfreq,z=z_j2054,file_name=file_salak,tuning_freq=[344.2,346.1],alma_bandwidth=8,bin=35,plot=True)



#Tripodi et al. 2024
obs_freq_92 = 92.26 #Use it as middle freq
file_tripodi_2024_92 = 'tripodi_2024_92_GHz'
alma_filter(file_name=file_tripodi_2024_92,middle_freq=obs_freq_92,bin=50,plot=True)


obs_freq_262 = 262.6 #Use it as middle freq
file_tripodi_2024_262 = 'tripodi_2024_262_GHz'
alma_filter(file_name=file_tripodi_2024_262,middle_freq=obs_freq_262,bin=50,plot=True)

obs_freq_262 = 263.93 #Use it as middle freq
file_tripodi_2024_262 = 'tripodi_2024_263_GHz'
alma_filter(file_name=file_tripodi_2024_263,middle_freq=obs_freq_263,bin=50,plot=True)

obs_freq_488 = 488.31
file_tripodi_2024_488 = 'tripodi_2024_488_GHz'
alma_filter(file_name=file_tripodi_2024_488,middle_freq=obs_freq_488,bin=50,plot=True)

obs_freq_674 = 674.97
file_tripodi_2024_674 = 'tripodi_2024_674_GHz'
alma_filter(file_name=file_tripodi_2024_674,middle_freq=obs_freq_674,bin=50,plot=True)
"""


#J2054-0005

filters = [['sdss.ip', 'sdss.ip_err']]
flux = [[0.00174,0.000352]]


header = ("#id redshift "
          "sdss.ip sdss.ip_err"
          "WISE1 WISE1_err "
          "herschel_pacs_100 herschel_pacs_100_err "
          "herschel_pacs_160 herschel_pacs_160_err "
          "herschel_spire_psw herschel_spire_psw_err "
          "herschel_spire_pmw herschel_spire_pmw_err "
          "IRAM_MAMBO2.250GHz IRAM_MAMBO2.250GHz_err "
          "hashimoto_j2054 hashimoto_j2054_err "
          "salak_j2054 salak_j2054_err "
          "tripodi_2024_92_GHz tripodi_2024_92_GHz_err "
          "tripodi_2024_262_GHz tripodi_2024_262_GHz_err "
          "tripodi_2024_263_GHz tripodi_2024_263_GHz_err "
          "tripodi_2024_488_GHz tripodi_2024_488_GHz_err "
          "tripodi_2024_674_GHz tripodi_2024_674_GHz_err "
          )

data = ('J2054',6.0392,
        0.00174,0.000352,
        0.019,0.006,
        3.1,1.0,
        10.5,2.0,
        15.2,5.4,
        12.0,4.9,
        2.38,0.53,
        10.35,0.15,
        5.723,0.009,
        0.082,0.009,
        2.93,0.07,
        3.08, 0.03,
        11.71, 0.11,
        9.87, 0.94
        )


with open("/home/sai/source_data.txt", "w") as f:
    f.write(header + "\n")
    f.write(" ".join(map(str, data)) + "\n")
