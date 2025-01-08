import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt
import utils


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
        "[NII]205": {'excitation_potential': 70 * u.K, "wavelength_micron": 205.80 * u.micron, "frequency_ghz": 1461.1 * u.GHz,"einstein_A_coefficients": 2.1e-6 * 1/u.s, "critical_density_CW13": 48*1/(u.cm ** 3),
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

    def frequency_conversion(self,unit:str):
        """
        :param unit: Unit to convert to (e.g. Hz, MHz)
        :return: Converted frequency with the new unit
        """
        return self.frequency.to(getattr(u, unit))

    def wavelength_conversion(self,unit:str):
        """
        :param unit: Unit to convert to (e.g., 'angstrom', 'nm', 'micron', 'm')
        :return: Converted wavelength with the new unit
        """
        return self.wavelength.to(getattr(u, unit))

    @classmethod
    def plot_example_figure(cls):
        """
        Plot a straight line as an example figure.
        """
        x = [0, 1, 2, 3, 4, 5]
        y = [i * 2 for i in x]  # y = 2x
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, label="y = 2x", color="blue", linestyle="--")
        plt.xlabel("X-axis")
        plt.ylabel("Y-axis")
        #plt.title(f"Example Plot for {self.name}")
        plt.legend()
        plt.grid(True)
        plt.show()



EmissionLine.plot_example_figure()
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
