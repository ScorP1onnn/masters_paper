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

    noema_bands = {
        #https://www.craf.eu/radio-observatories-in-europe/noema/
        "1": {"frequency_range": np.array([72,116]) * u.GHz, "wavelength_range": np.round(utils.ghz_to_mum(np.array([72,116])),2) * u.micron},
        "2": {"frequency_range": np.array([127,179]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([127,179])),2) * u.micron},
        "3": {"frequency_range": np.array([200,276]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([200,276])),2) * u.micron},
        "4": {"frequency_range": np.array([275,373]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([275,373])),2) * u.micron},
    }

    vla_bands = {
        #https://science.nrao.edu/facilities/vla/docs/manuals/oss2016A/performance/bands
        "vla_4": {"frequency_range": np.array([0.058,0.084]) * u.GHz, "wavelength_range": np.round(utils.ghz_to_mum(np.array([0.058,0.084])),2) * u.micron},
        "vla_p": {"frequency_range": np.array([0.23, 0.47]) * u.GHz, "wavelength_range": np.round(utils.ghz_to_mum(np.array([0.23, 0.47])), 2) * u.micron},
        "vla_l": {"frequency_range": np.array([1.0,2.0]) * u.GHz, "wavelength_range": np.round(utils.ghz_to_mum(np.array([1.0,2.0])), 2) * u.micron},
        "vla_s": {"frequency_range": np.array([2.0,4.0]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([2.0,4.0])), 2) * u.micron},
        "vla_c": {"frequency_range": np.array([4.0,8.0]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([4.0,8.0])), 2) * u.micron},
        "vla_x": {"frequency_range": np.array([8.0,12.0]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([8.0,12.0])), 2) * u.micron},
        "vla_ku": {"frequency_range": np.array([12.0,18.0]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([12.0,18.0])), 2) * u.micron},
        "vla_k": {"frequency_range": np.array([18.0,26.5]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([18.0,26.5])), 2) * u.micron},
        "vla_ka": {"frequency_range": np.array([26.5,40.0]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([26.5,40.0])), 2) * u.micron},
        "vla_q": {"frequency_range": np.array([40.0,50.0]) * u.GHz,"wavelength_range": np.round(utils.ghz_to_mum(np.array([40.0,50.0])), 2) * u.micron},

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

            if self.name in ["[OIII]52","[OIII]88","[NII]122","[NII]205"]:
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

    def to_unit(self, attribute: str, output_unit: u.Unit = None, save=False):
        """
        Convert the specified attribute (frequency or wavelength) to another unit.

        :param attribute: The attribute to convert ('frequency' or 'wavelength').
        :param output_unit: Output unit to be converted into.
        :param save: Save the converted value back to the attribute if True.
        :return: Return converted value if save is False.
        """
        if output_unit is None:
            raise ValueError("Output unit must be specified.")

        # Determine the type of unit and attribute
        if attribute == 'frequency':
            if not output_unit.is_equivalent(u.Hz):
                raise u.UnitsError(f"The specified unit '{output_unit}' is not a frequency unit.")
            value = self.frequency
        elif attribute == 'wavelength':
            if not output_unit.is_equivalent(u.m):
                raise u.UnitsError(f"The specified unit '{output_unit}' is not a wavelength unit.")
            value = self.wavelength
        else:
            raise AttributeError(f"Attribute '{attribute}' is not supported. Use 'frequency' or 'wavelength'.")

        # Perform conversion
        converted_value = value.to(output_unit)

        if save:
            setattr(self, attribute, converted_value)
        else:
            return converted_value

    def declination_range(self,telescope:str='alma'):
        if telescope.lower() == 'alma':
            print(f"ALMA can observe targets within +40 deg and -70 deg, corresponding to a maximum elevation of 25 deg at the ALMA site can be observed.\n"
              f"Use 'http://catserver.ing.iac.es/staralt/index.php' if the see if the source is visible to ALMA or not")
        elif telescope.lower() == 'noema':
            #https://www.iram.fr/GENERAL/calls/s20/NOEMACapabilities.pdf Section:2.8
            print(f"NOEMA can observe targets within +90 deg and -30 deg. sources between declinations of −30 and −25 degrees"
                  f"\nare very difficult to observe and they do not rise much above 10 degrees in elevation"
                  f"\nand suffer from heavy shadowing in the compact configurations")


    def alma_band_observe(self,z:float=None,telescope:str='alma',observed_frequency_in_GHz:u.Unit = None):
        """
        Determine the ALMA band that can observe the line for a specific redshift or a given observed frequency.
        :param z: Redshift
        :param observed_frequency_in_GHz: Observed frequency in GHz
        :return: ALMA band(s) that can observe the line or given frequency.
        """
        if telescope.lower()=='alma':
            telescope_data = self.alma_bands.items()
        elif telescope.lower() == 'noema':
            telescope_data = self.noema_bands.items()
        elif telescope.lower() == 'vla':
            telescope_data = self.vla_bands.items()
        else:
            raise ValueError(f"Telescope not found. \nTelesclopes available: 'ALMA', 'NOEMA', and 'VLA'." )


        bands = []
        if z!=None:
            obs_freq = self.observed_frequency(z)
            for band, info in telescope_data:
                freq_range = info.get("frequency_range")
                if freq_range[0] <= obs_freq <= freq_range[1]:
                    bands.append(band)

        elif z==None and observed_frequency_in_GHz is not None:
            if not observed_frequency_in_GHz.unit.is_equivalent(u.Hz):
                raise u.UnitsError(f"The specified unit is not a frequency unit.")
            obs_freq = observed_frequency_in_GHz.to(u.GHz)
            for band, info in telescope_data:
                freq_range = info.get("frequency_range")
                if freq_range[0] <= observed_frequency_in_GHz <= freq_range[1]:
                    bands.append(band)

        if bands:
            if z!=None:
                print(rf"{self.name} at z={z} (v_obs = {obs_freq:.2f}) can be observed with {telescope.upper()} band: {', '.join(bands)}")
            elif z == None and observed_frequency_in_GHz is not None:
                print(f"The frequency {obs_freq:.2f} can be observed with {telescope.upper()} band: {', '.join(bands)}")
        else:
            if z!=None:
                print(f"The observed frequency ({obs_freq:.2f}) for {self.name} at z={z} is not within the range of any {telescope.upper()} band.")
            elif z==None and observed_frequency_in_GHz is not None:
                print(f"The frequency {obs_freq:.2f} is not within the range of any {telescope.upper()} band.")



#Some checks

line = EmissionLine("[NII]205")
line.declination_range(telescope='alma')
line.declination_range(telescope='noema')
print("")
line.alma_band_observe(z=4, telescope='alma')
line.alma_band_observe(z=4, telescope='noema')
line.alma_band_observe(z=4, telescope='vla')

print("")
print(line.frequency)
converted_freq = line.to_unit(attribute='frequency', output_unit=u.kHz)
print(converted_freq)
line.to_unit(attribute='frequency', output_unit=u.kHz, save=True)  # Save back to `obj.frequency`.
print(line.frequency)


print("")
print(line.wavelength)
converted_wave = line.to_unit(attribute='wavelength', output_unit=u.nm)
print(converted_wave)
line.to_unit(attribute='wavelength', output_unit=u.nm, save=True)  # Save back to `obj.wavelength`.
print(line.wavelength)


exit()
