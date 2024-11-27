import os
from cProfile import label
from fileinput import filename

import matplotlib.pyplot as plt
import numpy as np
import interferopy.tools as tools
import astropy.units as u
import utils as utils
from utils import ghz_to_mum


def alma_filter(rest_freq,z,file_name,tuning_freq=[],middle_freq=0,alma_bandwidth=7.5,bin=30,plot=False):


    if middle_freq==0:
        tuning_freq = np.asarray(tuning_freq)
        central_freq = np.mean(tuning_freq)
    else:
        central_freq = middle_freq


    ref_freq = rest_freq/(1+z)
    freq_Step = tools.kms2ghz(bin,ref_freq)

    a = central_freq - ((alma_bandwidth / 2) - freq_Step)
    b = central_freq + ((alma_bandwidth / 2) + freq_Step)

    array_GHz = np.arange(a, b, freq_Step) * u.GHz
    array_angstroms = array_GHz.to(u.angstrom, equivalencies=u.spectral())

    transmission = np.zeros(len(array_angstroms))
    transmission[1:-1] = transmission[1:-1] + 1

    #print(array_GHz)
    #print(array_angstroms[0].value)
    #print(ghz_to_mum(array_GHz.value))


    #Wavelength in Angstroms in observed frame (or sky frame)
    desktop_path = os.path.expanduser('~') + '/Desktop' + '/alma_iram_filters'
    file_path = os.path.join(desktop_path, file_name + '.dat')
    file_1 = open(file_path, "w+")
    file_1.write(f"# {file_name}\n# energy\n")
    file_1.write(f"# boxcar shape filter centered at {central_freq} GHz or {utils.ghz_to_mum(central_freq) *1e4} Angstroms (or {np.round(utils.ghz_to_mum(central_freq),2)} micrometer)")
    for i in range(len(array_angstroms)):
        file_1.write(f"\n")
        file_1.write(f"{np.round(array_angstroms[i].value,4)} {transmission[i]}")
    file_1.close()
    print(f"FILE {file_name + '.dat'} CREATED")





    #Wavelength in Observed Frame (or Sky frame)
    if plot == True:

        fig,ax = plt.subplots(1,1)

        ax.plot(array_angstroms, transmission)
        ax.axvline(utils.ghz_to_mum(central_freq) *1e4, color='black',ls='--',label=f'freq_central = {central_freq} GHz')
        ax.set_xlabel(r"Wavelength [$\AA$]")
        ax.set_ylabel("Transmission")
        plt.legend()

        ax_freq = ax.twiny()
        ax_freq.plot(array_GHz,transmission)
        ax_freq.invert_xaxis()
        ax_freq.set_xlabel('Observed Frequency [GHz]')
        plt.show()

        """
        plt.plot(array_angstroms, transmission)
        plt.axvline(utils.ghz_to_mum(central_freq) *1e4, color='black',ls='--',label=f'freq_central = {central_freq} GHz')
        plt.xlabel(r"Wavelength [$\AA$]")
        plt.ylabel("Transmission")
        plt.legend()
        plt.show()"""








#Hashimoto et al. 2019
restfreq = 3393.006244
z_j2054=6.0391
file = 'hashimoto_j2054'
alma_filter(rest_freq=restfreq,z=z_j2054,file_name=file,tuning_freq=[480.71,483.68],plot=True)

restfreq = 3393.006244
z_j2310=6.0035
file = 'hashimoto_j2310'
alma_filter(rest_freq=restfreq,z=z_j2310,file_name=file,tuning_freq=[483.19,486.31],plot=True)


