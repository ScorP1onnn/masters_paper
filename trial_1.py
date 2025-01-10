import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import utils
import scipy.constants as const


"""
r = 2.64  * u.m**2/u.kg
print(r.to(u.cm**2/u.g))

kappa_ref = 0.45 * u.cm**2/u.g
kappa_ref = kappa_ref.to(u.m**2/u.kg)

print(kappa_ref)
"""




class EmissionLine:
    # Class-level dictionary storing the predefined emission lines
    predefined_lines = {

        #Fine-Structure Lines (Carilli & Walter 2013) [Decarli23: Decarli et al. 2023]
        "[OI]63": {'excitation_potential': 228 * u.K, "wavelength_micron": 63.18 * u.micron, "frequency_ghz": 4744.8 * u.GHz, "einstein_A_coefficients": 9e-5 * 1/u.s, "critical_density_CW13": 4.7e5*1/(u.cm ** 3)},
        "[OI]145": {'excitation_potential': 329, "wavelength_micron": 145.53 * u.micron, "frequency_ghz": 2060.1 * u.GHz, "einstein_A_coefficients": 1.7e-5 * 1/u.s, "critical_density_CW13": 9.4e4*1/(u.cm ** 3)},

        "[OIII]52": {'excitation_potential': 440 * u.K, "wavelength_micron": 51.82 * u.micron, "frequency_ghz": 5785.9 * u.GHz, "einstein_A_coefficients": 9.8e-5 * 1/u.s, "critical_density_CW13": 3.6e3*1/(u.cm ** 3),
                     "critical_density_Decarli23": {5000: 2227*1/(u.cm ** 3), 10000:3048*1/(u.cm ** 3), 20000:4163*1/(u.cm ** 3)}},
        "[OIII]88": {'excitation_potential': 163 * u.K, "wavelength_micron": 88.36 * u.micron, "frequency_ghz": 3393.0 * u.GHz, "einstein_A_coefficients": 2.6e-5 * 1/u.s, "critical_density_CW13": 510*1/(u.cm ** 3),
                     "critical_density_Decarli23": {5000:1307*1/(u.cm ** 3), 10000:1798*1/(u.cm ** 3), 20000:2496*1/(u.cm ** 3)}},

        "[CII]158": {'excitation_potential': 91 * u.K, "wavelength_micron": 157.74 * u.micron, "frequency_ghz": 1900.5 * u.GHz, "einstein_A_coefficients": 2.1e-6 * 1/u.s, "critical_density_CW13_neutral": 2.8e3*1/(u.cm ** 3),
                     "critical_density_CW13_ionized": 50*1/(u.cm ** 3), "critical_density_Decarli23_ionized":{5000:41*1/(u.cm ** 3), 10000:54*1/(u.cm ** 3), 20000:71*1/(u.cm ** 3)}},

        "[NII]122": {'excitation_potential': 188 * u.K, "wavelength_micron": 121.90 * u.micron, "frequency_ghz": 2459.4 * u.GHz, "einstein_A_coefficients": 7.5e-6 * 1/u.s, "critical_density_CW13": 310*1/(u.cm ** 3),
                     "critical_density_Decarli23": {5000:199*1/(u.cm ** 3), 10000:260*1/(u.cm ** 3), 20000:333*1/(u.cm ** 3)}},
        "[NII]205": {'excitation_potential': 70 * u.K, "wavelength_micron": 205.18 * u.micron, "frequency_ghz": 1461.1 * u.GHz,"einstein_A_coefficients": 2.1e-6 * 1/u.s, "critical_density_CW13": 48*1/(u.cm ** 3),
                     "critical_density_Decarli23": {5000:127*1/(u.cm ** 3), 10000:169*1/(u.cm ** 3), 20000:222*1/(u.cm ** 3)}},

        "[CI]370": {'excitation_potential': 63 * u.K, "wavelength_micron": 370.42 * u.micron, "frequency_ghz": 809.34 * u.GHz,"einstein_A_coefficients": 2.7e-7 * 1/u.s, "critical_density_CW13": 1.2e3*1/(u.cm ** 3)},
        "[CI]609": {'excitation_potential': 24 * u.K, "wavelength_micron": 609.14 * u.micron, "frequency_ghz": 492.16 * u.GHz,"einstein_A_coefficients": 7.9e-8 * 1/u.s, "critical_density_CW13": 470*1/(u.cm ** 3)},

        #CO
        "CO_10": {'excitation_potential': 5.5 * u.K, "wavelength_micron": 2601 * u.micron, "frequency_ghz": 115.27 * u.GHz,"einstein_A_coefficients": 7.2e-8 * 1/u.s, "critical_density_CW13": 2.1e3*1/(u.cm ** 3)},
        "CO_21": {'excitation_potential': 16.6 * u.K,"wavelength_micron": 1300 * u.micron, "frequency_ghz": 230.54 * u.GHz,"einstein_A_coefficients": 6.9e-7 * 1/u.s, "critical_density_CW13": 1.1e4*1/(u.cm ** 3)},
        "CO_32": {'excitation_potential': 33.2 * u.K,"wavelength_micron": 867 * u.micron, "frequency_ghz": 345.80 * u.GHz,"einstein_A_coefficients": 2.5e-6 * 1/u.s, "critical_density_CW13": 3.6e4*1/(u.cm ** 3)},
        "CO_43": {'excitation_potential': 55.3 * u.K,"wavelength_micron": 650.3 * u.micron, "frequency_ghz": 461.04 * u.GHz,"einstein_A_coefficients": 6.1e-6 * 1/u.s, "critical_density_CW13": 8.7e4*1/(u.cm ** 3)},
        "CO_54": {'excitation_potential': 83.0 * u.K, "wavelength_micron": 520.2 * u.micron, "frequency_ghz": 576.27 * u.GHz,"einstein_A_coefficients": 1.2e-5 * 1/u.s, "critical_density_CW13": 1.7e5*1/(u.cm ** 3)},
        "CO_65": {'excitation_potential': 116.2 * u.K, "wavelength_micron": 433.6 * u.micron, "frequency_ghz": 691.47 * u.GHz,"einstein_A_coefficients": 2.1e-5 * 1/u.s, "critical_density_CW13": 2.9e5*1/(u.cm ** 3)},
        "CO_76": {'excitation_potential': 154.9 * u.K, "wavelength_micron": 371.7 * u.micron, "frequency_ghz": 806.65 * u.GHz,"einstein_A_coefficients": 3.4e-5 * 1/u.s, "critical_density_CW13": 4.5e5*1/(u.cm ** 3)},
        "CO_87": {'excitation_potential': 199.1 * u.K, "wavelength_micron": 325.2 * u.micron, "frequency_ghz": 921.80 * u.GHz,"einstein_A_coefficients": 5.1e-5 * 1/u.s, "critical_density_CW13": 6.4e5*1/(u.cm ** 3)},
        "CO_98": {'excitation_potential': 248.9 * u.K, "wavelength_micron": 289.1 * u.micron, "frequency_ghz": 1036.9 * u.GHz,"einstein_A_coefficients": 7.3e-5 * 1/u.s, "critical_density_CW13": 8.7e5*1/(u.cm ** 3)},
        "CO_109": {'excitation_potential':  304.2 * u.K, "wavelength_micron": 260.2 * u.micron, "frequency_ghz": 1152.0 * u.GHz,"einstein_A_coefficients": 1.0e-4 * 1/u.s, "critical_density_CW13": 1.1e6*1/(u.cm ** 3)},

    }

    alma_bands = {

        "1": {"frequency_range": np.array([35,50]) * u.GHz, "wavelength_range": np.array([6,8.6]) * u.mm},
        "2": {"frequency_range": np.array([67, 116]) * u.GHz, "wavelength_range": np.array([2.6, 4,5]) * u.mm},
        "3": {"frequency_range": np.array([84, 116]) * u.GHz, "wavelength_range": np.array([2.6, 3.6]) * u.mm},
        "4": {"frequency_range": np.array([125, 163]) * u.GHz, "wavelength_range": np.array([1.8, 2.4]) * u.mm},
        "5": {"frequency_range": np.array([163, 211]) * u.GHz, "wavelength_range": np.array([1.4, 1.8]) * u.mm},
        "6": {"frequency_range": np.array([211, 275]) * u.GHz, "wavelength_range": np.array([1.1, 1.4]) * u.mm},
        "7": {"frequency_range": np.array([275, 373]) * u.GHz, "wavelength_range": np.array([0.8, 1.1]) * u.mm},
        "8": {"frequency_range": np.array([385, 500]) * u.GHz, "wavelength_range": np.array([0.6, 0.8]) * u.mm},
        "9": {"frequency_range": np.array([602, 720]) * u.GHz, "wavelength_range": np.array([0.4, 0.5]) * u.mm},
        "10": {"frequency_range": np.array([787, 950]) * u.GHz, "wavelength_range": np.array([0.3, 0.4]) * u.mm},

    }

    def __init__(self, name):
        """
        Initialize an EmissionLine object using the predefined line information.
        """
        if name not in EmissionLine.predefined_lines:
            raise ValueError(f"Emission line '{name}' not found in predefined lines.")

        line_info = EmissionLine.predefined_lines[name]
        self.name = name

        if self.name == "[CII]158":
            self.excitation_potential = line_info["excitation_potential"]
            self.wavelength = line_info["wavelength_micron"]
            self.frequency = line_info["frequency_ghz"]
            self.einstein_A_coefficients = line_info["einstein_A_coefficients"]
            self.critical_density_CW13_neutral = line_info["critical_density_CW13_neutral"]
            self.critical_density_CW13_ionized = line_info["critical_density_CW13_ionized"]
            self.critical_density_Decarli23_ionized = line_info["critical_density_Decarli23_ionized"]

        else:
            self.excitation_potential = line_info["excitation_potential"]
            self.wavelength = line_info["wavelength_micron"]
            self.frequency = line_info["frequency_ghz"]
            self.einstein_A_coefficients = line_info["einstein_A_coefficients"]
            self.critical_density_CW13 = line_info["critical_density_CW13"]

            if self.name == "[OIII]52" or "[OIII]88" or "[NII]122" or "[NII]205":
                self.critical_density_Decarli23 = line_info["critical_density_Decarli23"]


    def observed_frequency(self, redshift:float):
        """
        Calculate the observed frequency of the emission line for a given redshift.
        """
        return self.frequency / (1 + redshift)

    def observed_wavelength(self, redshift:float):
        """
        Calculate the observed wavelength of the emission line for a given redshift.
        """
        return self.wavelength * (1 + redshift)

    def to_unit(self, value: u.Quantity, output_unit: u.Unit = None):
        if output_unit is None:
            raise ValueError("Output unit must be specified.")
        if not value.unit.is_equivalent(output_unit):
            raise u.UnitsError(f"Units {value.unit} and {output_unit} are not equivalent.")
        return value.to(output_unit)

    def to_wavelength(self,value: u.Quantity, output_unit: u.Unit = None):
        if output_unit is None:
            output_unit = u.GHz  # Default output unit for frequency
        if not value.unit.is_equivalent(u.m):
            raise u.UnitsError(f"Unit should be of wavelength units.")
        return value.to(output_unit, equivalencies=u.spectral())

    def to_frequency(self,value: u.Quantity, output_unit: u.Unit = None):
        if output_unit is None:
            output_unit = u.micron  # Default output unit for wavelength
        if not value.unit.is_equivalent(u.Hz):
            raise u.UnitsError(f"Unit should be of Frequency units.")
        return value.to(output_unit, equivalencies=u.spectral())

    def alma_declination_range(self):
        print(f"ALMA can observe targets within +40 deg and -70 deg, corresponding to a maximum elevation of 25 deg at the ALMA site can be observed.\n"
              f"Use 'http://catserver.ing.iac.es/staralt/index.php' if the see if the source is visible to ALMA or not")

    def alma_band_observe(self,z:float=0.,observed_frequency_in_GHz:float=None):
        """
        Determine the ALMA band that can observe the line for a specific redshift or a given observed frequency.
        :param z: Redshift
        :param observed_frequency_in_GHz: Observed frequency in GHz
        :return: ALMA band(s) that can observe the line or given frequency.
        """
        bands = []
        if z!=0.:
            obs_freq = self.observed_frequency(z)
            for band, info in self.alma_bands.items():
                freq_range = info["frequency_range"]
                if freq_range[0] <= obs_freq <= freq_range[1]:
                    bands.append(band)

        elif z==0. and observed_frequency_in_GHz is not None:
            if isinstance(observed_frequency_in_GHz, u.Quantity) == False:
                observed_frequency_in_GHz = observed_frequency_in_GHz * u.GHz

            for band, info in self.alma_bands.items():
                freq_range = info["frequency_range"]
                if freq_range[0] <= observed_frequency_in_GHz <= freq_range[1]:
                    bands.append(band)

        if bands:
            print(f"{self.name} can be observed with ALMA band(s) {bands}")
        else:
            if z!=0:
                print(f"The observed frequency ({obs_freq:.2f}) for line {self.name} at z = {z} is not within the range of any ALMA band.")
            elif z==0 and observed_frequency_in_GHz is not None:
                print(f"The frequency {observed_frequency_in_GHz:.2f} is not within the range of any ALMA band.")


    """
    #@classmethod
    def plot_example_figure(self):
        #Plot a straight line as an example figure.
        
        z = np.linspace(0,10,101)

        for i in range(1,11):
            plt.fill_between(z, self.alma_bands[f'{i}'].get('frequency_range')[0].value,
                             self.alma_bands[f'{i}'].get('frequency_range')[1].value,color='skyblue', alpha=0.1)
            plt.axhline(y=self.alma_bands[f'{i}'].get('frequency_range')[0].value, color='black', linestyle='-',linewidth=0.5)
            plt.axhline(y=self.alma_bands[f'{i}'].get('frequency_range')[1].value, color='black', linestyle='-',linewidth=0.5)

            middle = (self.alma_bands[f'{i}'].get('frequency_range')[0].value + self.alma_bands[f'{i}'].get('frequency_range')[1].value)/2
            if i==1:
                plt.text(x=0.5, y=middle - 2, s=f'ALMA {i}', color='black', fontsize=7, ha='center', va='center')
            elif i == 2:
                plt.text(x=0.5, y=middle - 18, s=f'ALMA {i}', color='black', fontsize=7, ha='center', va='center')
            else:
                plt.text(x=0.5,y=middle - 4, s=f'ALMA {i}',color='black',fontsize=7,ha='center',va='center')

        plt.xlim(z[0],z[-1])
        plt.ylim(10,1300)
        plt.yscale('log')
        plt.show()
        exit()
        y = [i * 2 for i in x]  # y = 2x
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, label="y = 2x", color="blue", linestyle="--")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        #plt.title(f"Example Plot for {self.name}")
        plt.legend()
        plt.grid(True)
        plt.show()
        """


line = EmissionLine("[NII]205")



exit()




#print(line.wavelength)
a = line.convert_wavelength_frequency(line.wavelength)
print(a)
line.alma_declination_range()
#line.alma_band_observe(observed_frequency_in_GHz=600)
exit()
# Example usage
# Create objects for predefined emission lines
nii_205 = EmissionLine("[NII]205")
cii_158 = EmissionLine("[CII]158")
oiii_88 = EmissionLine("[OIII]88")

# Print information for all lines
lines = [nii_205, cii_158, oiii_88]
for line in lines:
    print(f"Name: {line.name}")
    print(f"Wavelength: {line.wavelength.value} µm")
    print(f"Frequency: {line.frequency.value} GHz")
    print(f"Energy Potential: {line.excitation_potential.value} K")
    print(f"Observed Frequency at z=1: {line.observed_frequency(1).value:.2f} GHz")
    print(f"Observed wavelength at z=1: {line.observed_wavelength(1).value:.2f} GHz")
    print("-" * 40)






exit()
# Plot an example figure for one of the lines
cii_158.plot_example_figure()

print(nii_205.excitation_potential)
exit()

"""
To account for 15% calibration error in the flux error, the formula used is

corrected/accounted_error = SQRT( (error ** 2) + ((0.15 * flux_value)**2) )

"""
