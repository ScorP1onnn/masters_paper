import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import axvline

import utils as utils
import emcee
from scipy import integrate, constants
import astropy.units as u
import corner
import interferopy.tools as iftools
from uncertainties import ufloat
from tqdm import tqdm
import pandas as pd


def optically_thick_integral(z,dust_mass,dust_temp,dust_beta,solid_angle,lower_limit,upper_limit):

    num_points = 10001
    integral = integrate.simpson(y = utils.dust_s_obs(np.linspace(lower_limit, upper_limit, num_points),
                                                   z=z,
                                                   dust_mass=dust_mass,
                                                   dust_temp=dust_temp,
                                                   beta=dust_beta,
                                                   optically_thick_regime=True,
                                                   solid_angle=solid_angle,
                                                   output_unit_mjy=True),
                                 x = np.linspace(lower_limit, upper_limit, num_points))

    """
    #Takes too much time
    integral = integrate.quad(lambda x: utils.dust_s_obs(x,
                                              z=z,
                                              dust_mass=dust_mass,
                                              dust_temp=dust_temp,
                                              beta=dust_beta,
                                              optically_thick_regime=True,
                                              solid_angle=solid_angle,
                                              output_unit_mjy=True), lower_limit, upper_limit)[0]  # mJy * Hz)
    """

    # Taken from line 558 of https://github.com/roberta96/EOS-Dustfit/blob/main/colddust_sed_models/cdsed_modelling/results_plot.py
    integral = (integral * u.mJy * u.Hz).to(u.mJy * u.GHz)  # Convert to mJy*GHz
    dl = (utils.luminosity_distance(z) * u.Mpc).to(u.cm)
    integral = (4 * np.pi * (dl.value ** 2) * integral.value * 1e-26 * 1e9)  # (cm2)*(mJy*GHz)*1e-26*1e9 = erg/s
    L_sun = 3.826 * 1e33  # erg/s
    return integral / L_sun  # Lsun


def optically_thin_integral(dust_mass,dust_temp,dust_beta,lower_limit,upper_limit):
    num_points = 10001
    integral = integrate.simpson(y=utils.dust_luminosity_one_freq_value(np.linspace(lower_limit, upper_limit, num_points),
                                                                        mass_dust=dust_mass,
                                                                        temp_dust=dust_temp,
                                                                        beta=dust_beta),
                                 x = np.linspace(lower_limit, upper_limit, num_points))
    """
    integral = integrate.quad(lambda x: utils.dust_luminosity_one_freq_value(x,
                                                                             mass_dust=dust_mass,
                                                                             temp_dust=dust_temp,
                                                                             beta=dust_beta),lower_limit,upper_limit)[0]
    """

    return integral * u.W.to(u.solLum) # Lsun



def dust_integrated_luminosity(z,dust_mass,dust_temp,dust_beta,solid_angle,optically_thick_regime=False,lum='both'):


    if optically_thick_regime == True:

        tir_lower_limit = constants.c / (1000e-6)  # Convert to frequency for integration
        tir_upper_limit = constants.c / (8e-6)
        tir_lum = optically_thick_integral(z,dust_mass,dust_temp,dust_beta,solid_angle,tir_lower_limit,tir_upper_limit) # Lsun

        """
        # Using limit 42.5-122.5 gives a L_FIR value of order 10^10, which makes no sense. On the other hand, 
        # adopting limit 42.5-1000 gives an answer very similar to L_IR.
        # The FIR limit 42.0-1000 is taken from https://github.com/roberta96/EOS-Dustfit/blob/main/colddust_sed_models/cdsed_modelling/results_plot.py#L377 Line 477
        """
        
        fir_lower_limit = constants.c / (1000e-6) 
        fir_upper_limit = constants.c / (42.5e-6)
        fir_lum = optically_thick_integral(z, dust_mass, dust_temp, dust_beta, solid_angle, fir_lower_limit,fir_upper_limit) # Lsun


        #fir_lum = np.NaN #

    else:

        tir_lower_limit = constants.c / (1000e-6)  # Convert to frequency for integration
        tir_upper_limit = constants.c / (8e-6)

        fir_lower_limit = constants.c / (122.5e-6)  # Convert to frequency for integration
        fir_upper_limit = constants.c / (42.5e-6)

        tir_lum = optically_thin_integral(dust_mass,dust_temp,dust_beta,tir_lower_limit,tir_upper_limit) # Lsun
        fir_lum = optically_thin_integral(dust_mass, dust_temp, dust_beta, fir_lower_limit, fir_upper_limit) # Lsun

    if lum.lower() == 'tir' or lum.lower() == 'ir':
        return tir_lum
    elif lum.lower() == 'fir':
        return fir_lum
    elif lum.lower() == 'both':
        return tir_lum, fir_lum




def prior_distribution(param, limits,dist: str):
    if dist.lower() == 'flat':
        """
        Flat (uniform) prior.
        """
        if limits[0]<param<limits[1]:
            return 0.0
        return -np.inf

    if dist.lower() == 'gauss' or 'gaussian':
        """
        Gaussian prior.
        """
        mean = limits[0]
        std = limits[1]
        if mean - std <= param <=mean + std:
            return -0.5 * ((param - mean) ** 2 / std ** 2) - np.log(std * np.sqrt(2 * np.pi))
        return -np.inf

    if dist.lower() == 'logn' or 'log_n' or 'log_normal':
        """
        Log-normal prior.
        """
        mean = limits[0]
        std = limits[1]
        if mean - std <= param <= mean + std and param > 0: # Ensure param is positive
            log_param = np.log(param)
            return -0.5 * ((log_param - mean) ** 2 / std ** 2) - log_param - np.log(std * np.sqrt(2 * np.pi))
        return -np.inf



def mbb_emcee(nu_obs, z, flux_obs, flux_err, dust_mass, dust_temp, dust_beta,
            nparams, params_type,
            solid_angle, optically_thick_regime,


            dust_mass_prior_distribution,
            dust_temp_prior_distribution,
            dust_beta_prior_distribution,
            solid_angle_prior_distribution,


            dust_mass_limit,
            dust_temp_limit,
            dust_beta_limit,
            solid_angle_limit,

            nwalkers,
            initial_guess_values,
            nsteps,
            flat_samples_discarded,
            trace_plots,
            corner_plot,
            corner_kwargs
            ):

    def log_likelihood(params):

        # Number of parameters = 1
        if nparams == 1:
            dust_m = params
            if dust_m <= 0:
                return -np.inf
            #print(dust_m)
            flux_model = utils.dust_s_obs(nu_obs=nu_obs,
                                          z=z,
                                          dust_mass=utils.mass_kgs_solar_conversion(dust_m,unit_of_input_mass='solar'),
                                          dust_temp=dust_temp,
                                          beta=dust_beta,
                                          solid_angle=solid_angle,
                                          optically_thick_regime=optically_thick_regime)

        # Number of parameters = 2
        elif nparams == 2:
            if params_type.lower() == 'mt':
                dust_m, dust_t = params
                if dust_m <= 0 or dust_t <= 0:
                    return -np.inf
                flux_model = utils.dust_s_obs(nu_obs=nu_obs,
                                              z=z,
                                              dust_mass=utils.mass_kgs_solar_conversion(dust_m,unit_of_input_mass='solar'),
                                              dust_temp=dust_t,
                                              beta=dust_beta,
                                              solid_angle=solid_angle,
                                              optically_thick_regime=optically_thick_regime)
            elif params_type.lower() == 'mb':
                dust_m, dust_b = params
                if dust_m <= 0 or dust_b <= 0:
                    return -np.inf
                flux_model = utils.dust_s_obs(nu_obs=nu_obs,
                                              z=z,
                                              dust_mass=utils.mass_kgs_solar_conversion(dust_m,unit_of_input_mass='solar'),
                                              dust_temp=dust_temp,
                                              beta=dust_b,
                                              solid_angle=solid_angle,
                                              optically_thick_regime=optically_thick_regime)
            else:
                raise ValueError(f"Unsupported params_type: {params_type}")

        # Number of parameters = 3
        elif nparams == 3:
            dust_m, dust_t, dust_b = params
            if dust_m <= 0 or dust_t <= 0 or dust_b <= 0:
                return -np.inf
            flux_model = utils.dust_s_obs(nu_obs=nu_obs,
                                          z=z,
                                          dust_mass=utils.mass_kgs_solar_conversion(dust_m,unit_of_input_mass='solar'),
                                          dust_temp=dust_t,
                                          beta=dust_b,
                                          solid_angle=solid_angle,
                                          optically_thick_regime=optically_thick_regime)

        # Number of parameters = 4
        elif nparams == 4:
            dust_m, dust_t, dust_b, dust_sa = params
            if dust_m <= 0 or dust_t <= 0 or dust_b <= 0 or dust_sa <= 0:
                return -np.inf
            flux_model = utils.dust_s_obs(nu_obs=nu_obs,
                                          z=z,
                                          dust_mass=utils.mass_kgs_solar_conversion(dust_m,unit_of_input_mass='solar'),
                                          dust_temp=dust_t,
                                          beta=dust_b,
                                          solid_angle=dust_sa,
                                          optically_thick_regime=optically_thick_regime)

        else:
            raise ValueError(f"Unsupported nparams value: {nparams}")

        chi2 = np.sum(((flux_obs - flux_model) / flux_err) ** 2)
        return -0.5 * chi2

    def log_prior(params):

        # Number of parameters = 1
        if nparams == 1:
            dust_m = params
            lp_m = prior_distribution(dust_m,dust_mass_limit,dust_mass_prior_distribution)
            return lp_m

        elif nparams == 2:
            if params_type.lower() == 'mt':
                dust_m, dust_t = params
                lp_m = prior_distribution(dust_m,dust_mass_limit,dust_mass_prior_distribution)
                lp_t = prior_distribution(dust_t,dust_temp_limit,dust_temp_prior_distribution)
                if not np.isfinite(lp_m) or not np.isfinite(lp_t):
                    return -np.inf  # At least one prior is invalid
                return lp_m + lp_t

            elif params_type.lower() == 'mb':
                dust_m, dust_b = params
                lp_m = prior_distribution(dust_m, dust_mass_limit, dust_mass_prior_distribution)
                lp_b = prior_distribution(dust_b, dust_beta_limit, dust_beta_prior_distribution)
                if not np.isfinite(lp_m) or not np.isfinite(lp_b):
                    return -np.inf  # At least one prior is invalid
                return lp_m + lp_b

        # Number of parameters = 3
        elif nparams == 3:
            dust_m, dust_t, dust_b = params
            lp_m = prior_distribution(dust_m, dust_mass_limit, dust_mass_prior_distribution)
            lp_t = prior_distribution(dust_t, dust_temp_limit, dust_temp_prior_distribution)
            lp_b = prior_distribution(dust_b, dust_beta_limit, dust_beta_prior_distribution)
            if not np.isfinite(lp_m) or not np.isfinite(lp_t) or not np.isfinite(lp_b):
                return -np.inf  # At least one prior is invalid
            return lp_m + lp_t + lp_b

        # Number of parameters = 4
        elif nparams == 4:
            dust_m, dust_t, dust_b, dust_sa = params
            lp_m = prior_distribution(dust_m, dust_mass_limit, dust_mass_prior_distribution)
            lp_t = prior_distribution(dust_t, dust_temp_limit, dust_temp_prior_distribution)
            lp_b = prior_distribution(dust_b, dust_beta_limit, dust_beta_prior_distribution)
            lp_sa = prior_distribution(dust_sa, solid_angle_limit, solid_angle_prior_distribution)
            if not np.isfinite(lp_m) or not np.isfinite(lp_t) or not np.isfinite(lp_b) or not np.isfinite(lp_sa):
                return -np.inf  # At least one prior is invalid
            return lp_m + lp_t + lp_b + lp_sa

    def log_posterior(params):
        lp = log_prior(params)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(params)

    # Set up the MCMC sampler
    ndim = nparams
    dust_mass_rescale = int((np.log10(dust_mass_limit[0]) + np.log10(dust_mass_limit[1]))/2)
    if params_type == 'mb':
        initial_guesses = np.asarray([initial_guess_values[0],initial_guess_values[2]])
        trace_label = ['dust_mass','dust_beta']
        corner_labels = [r"log($M_{\mathrm{dust}}$) [$M_\odot$]", r"$\beta_{\mathrm{dust}}$"]
    else:
        #initial_guesses = np.asarray(initial_guess_values[:ndim])
        initial_guesses = np.concatenate( (np.asarray([initial_guess_values[0] / 10**dust_mass_rescale]), np.asarray(initial_guess_values[1:ndim])),axis=0)
        trace_label = ['dust_mass', 'dust_temp','dust_beta', 'solid_angle'][:ndim]
        corner_labels = [r"log($M_{\mathrm{dust}}$) [$M_\odot$]", r"$T_{\mathrm{dust}}$ [K]",
                         r"$\beta_{\mathrm{dust}}$", r"$\Omega_{\mathrm{S}}$"][:ndim]

    pos = initial_guesses + 1e-2 * np.random.randn(nwalkers, ndim) # Initial positions
    pos[:,0] = pos[:,0] * 10**dust_mass_rescale

    # Initialize the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior) #, args=(nu_obs,flux_obs, flux_err)

    # Run the MCMC
    sampler.run_mcmc(pos, nsteps, progress=True)

    # Check 1: Autocorrelation time
    try:
        tau = sampler.get_autocorr_time(discard=flat_samples_discarded, quiet=True)
        print(f"Autocorrelation time: {tau}")
        if nsteps < 10 * np.max(tau):
            print("Warning: nsteps may not be sufficient. Consider increasing it.")
    except emcee.autocorr.AutocorrError:
        print("Autocorrelation time could not be estimated. Chains may not have converged.")

    if trace_plots == True:
        # Check 2: Trace plots
        print("Generating trace plots...")
        samples = sampler.get_chain()  # Get the raw samples
        fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
        labels = trace_label
        if ndim==1:
            for walker in samples[:, :, 0].T:  # Loop over walkers
                plt.plot(walker, alpha=0.5)
            plt.ylabel(labels[0])
            plt.xlabel("Step number")
            plt.show()
        else:

            for i in range(ndim):
                ax = axes[i]
                for walker in samples[:, :, i].T:  # Loop over walkers
                    ax.plot(walker, alpha=0.5)
                ax.set_ylabel(labels[i])
            axes[-1].set_xlabel("Step number")
            plt.show()

    if corner_plot == True and nparams>1:
        print("Generating corner plots...")
        corner_flat_samples = sampler.get_chain(discard=flat_samples_discarded, thin=10, flat=True).copy()
        corner_flat_samples[:, 0] = np.log10(corner_flat_samples[:, 0])  # np.log(dust_mass)
        corner_kwargs = corner_kwargs or {"labels": corner_labels, "show_titles": True, "plot_datapoints": True,  "title_fmt": ".2f"}
        corner.corner(corner_flat_samples, quantiles=[0.16, 0.5, 0.84], **corner_kwargs)
        plt.show()

    return sampler


def mbb_values(nu_obs, z, flux_obs, flux_err,

        gmf: float = 1.,
        dust_mass_fixed: float = 1e8,
        dust_temp_fixed: float = 37,
        dust_beta_fixed: float = 1.6,

        nparams: int = 3, params_type: str = 'mt',
        solid_angle:float = 0., optically_thick_regime: bool = False,


        dust_mass_prior_distribution: str = 'flat',
        dust_temp_prior_distribution: str = 'flat',
        dust_beta_prior_distribution: str = 'flat',
        solid_angle_prior_distribution: str = 'flat',


        dust_mass_limit = None,
        dust_temp_limit = None,
        dust_beta_limit = None,
        solid_angle_limit = None,

        nwalkers: int = 50,
        initial_guess_values = None,
        nsteps: int = 1000,
        flat_samples_discarded: int = 200,
        trace_plots: bool = True,
        corner_plot: bool = False,
        corner_kwargs: dict = None):

    """
    Estimates the MBB parameters (dust_mass, dust_temp, dust_beta, solid_angle) from emcee fitting. emcee fitting
    takes into account the dependency between variables, providing a more precise estimate on the TIR and FIR value and errors
    :param nu_obs: Observed frame frequency in Hz.
    :param z: Redshift
    :param flux_obs: Observed Flux in units W/Hz/m^2
    :param flux_err: Observed Flux error in units W/Hz/m^2

    :param dust_mass_fixed: Fixed dust mass value in solar masses (solar masses will be converted to kg in mbb.emcee function)
                            Keep it to 0 if you want to use dust_mass as a variable parameter
    :param dust_temp_fixed: Fixed dust temperature value in K
                            Keep it to 0 if you want to use dust_mass as a variable parameter
    :param dust_beta_fixed: Fixed beta value
                            Keep it to 0 if you want to use dust_mass as a variable parameter

    :param nparams: Number of variable parameters.
                    1: Only dust mass
                    2: ['mt' : dust mass and dust temperature]; ['mb': dust mass and beta]
                    3: dust mass, dust temperature, beta
                    4: dust mass, dust temperature, beta, solid angle
    :param params_type: 'mt' or 'mb' (only important when nparams = 2)
    :param solid_angle: solid_angle value
    :param optically_thick_regime: IF True:- Calculations done assuming optically thick regime (tau is not ignored)
                                   IF False:- Calculations done assuming optically thin regime (tau is ignored)

    :param dust_mass_prior_distribution: 'flat' or 'gaussian' or 'log_normal' distribution
    :param dust_temp_prior_distribution: 'flat' or 'gaussian' or 'log_normal' distribution
    :param dust_beta_prior_distribution: 'flat' or 'gaussian' or 'log_normal' distribution
    :param solid_angle_prior_distribution: 'flat' or 'gaussian' or 'log_normal' distribution

    :param dust_mass_limit: limits within which emcee should search for dust mass
                            'flat': [lower limit, upper limit]
                            'gauss' or 'gaussian': [mean, std (or width)]
                            'log_n' or 'log_normal': [mean, std (or width)]
    :param dust_temp_limit: limits within which emcee should search for dust temperature
                            'flat': [lower limit, upper limit]
                            'gauss' or 'gaussian': [mean, std (or width)]
                            'log_n' or 'log_normal': [mean, std (or width)]
    :param dust_beta_limit: limits within which emcee should search for beta
                            'flat': [lower limit, upper limit]
                            'gauss' or 'gaussian': [mean, std (or width)]
                            'log_n' or 'log_normal': [mean, std (or width)]
    :param solid_angle_limit: limits within which emcee should search for solid angle
                            'flat': [lower limit, upper limit]
                            'gauss' or 'gaussian': [mean, std (or width)]
                            'log_n' or 'log_normal': [mean, std (or width)]

    :param nwalkers: Number of walkers
    :param initial_guess_values: initial guess priors
    :param nsteps: Number of iterations
    :param corner_plot: Plot Corner plot
    :return: Dictionary containing MBB derived values and their 1-sigma posterior limits, TIR and FIR and their limits
    """

    if dust_mass_limit is None:
        dust_mass_limit = [1e6, 1e11]
    if not isinstance(dust_mass_limit, list):
        raise TypeError("dust_mass_limit must be a list.")

    if dust_temp_limit is None:
        dust_temp_limit = [25.,40.]
    if not isinstance(dust_temp_limit, list):
        raise TypeError("dust_temp_limit must be a list.")

    if dust_beta_limit is None:
        dust_beta_limit = [1.5, 2.5]
    if not isinstance(dust_beta_limit, list):
        raise TypeError("dust_beta_limit must be a list.")

    if solid_angle_limit is None:
        solid_angle_limit = [0.,1.]
    if not isinstance(solid_angle_limit, list):
        raise TypeError("solid_angle_limit must be a list.")

    if initial_guess_values is None:
        initial_guess_values = [1e8, 30, 2, 0.1]
    if not isinstance(initial_guess_values, list):
        raise TypeError("initial_guess_values must be a list.")

    sampler = mbb_emcee(nu_obs, z, flux_obs, flux_err, dust_mass_fixed, dust_temp_fixed, dust_beta_fixed,
            nparams, params_type, solid_angle, optically_thick_regime,

            dust_mass_prior_distribution,
            dust_temp_prior_distribution,
            dust_beta_prior_distribution,
            solid_angle_prior_distribution,
            dust_mass_limit,
            dust_temp_limit,
            dust_beta_limit,
            solid_angle_limit,

            nwalkers,
            initial_guess_values,
            nsteps,
            flat_samples_discarded,
            trace_plots,
            corner_plot,
            corner_kwargs)




    flat_samples = sampler.get_chain(discard=flat_samples_discarded, thin=10, flat=True)

    log_probs = sampler.get_log_prob(discard=flat_samples_discarded, thin=10, flat=True)
    chi2_values = -2 * log_probs
    min_chi2 = np.min(chi2_values)


    percentiles = [16, 50, 84]  # 1-sigma percentiles
    stats = {}

    stats['min_chi2'] = {
        "median": min_chi2,
        "lower_1sigma": np.NAN,
        "upper_1sigma": np.NAN,
        }


    # Calculate marginalized L_IR
    L_TIR_samples = []
    L_FIR_samples = []




    if nparams == 1:
        param_names = [r"dust_mass", 'μTIR x 10^13', 'μFIR x 10^13']
        for sample in tqdm(flat_samples, desc="Computing (F)IR Luminosity...."):
            L_TIR_samples.append(dust_integrated_luminosity(z=z,dust_mass=utils.mass_kgs_solar_conversion(sample[0],'solar'),dust_temp=dust_temp_fixed,dust_beta=dust_beta_fixed,
                                                            solid_angle=solid_angle,optically_thick_regime=optically_thick_regime,lum='tir'))
            L_FIR_samples.append(dust_integrated_luminosity(z=z,dust_mass=utils.mass_kgs_solar_conversion(sample[0],'solar'),dust_temp=dust_temp_fixed,dust_beta=dust_beta_fixed,
                                                            solid_angle=solid_angle,optically_thick_regime=optically_thick_regime,lum='fir'))

    elif nparams == 2:
        if params_type.lower() == 'mt':
            param_names = ["dust_mass", "dust_temp", 'μTIR x 10^13', 'μFIR x 10^13']
            for sample in tqdm(flat_samples, desc="Computing (F)IR Luminosity...."):
                L_TIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=dust_beta_fixed,
                                                                solid_angle=solid_angle,optically_thick_regime=optically_thick_regime, lum='tir'))
                L_FIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=dust_beta_fixed,
                                                                solid_angle=solid_angle,optically_thick_regime=optically_thick_regime, lum='fir'))

        elif params_type.lower() == 'mb':
            param_names = ["dust_mass", "dust_beta", 'μTIR x 10^13', 'μFIR x 10^13']
            for sample in tqdm(flat_samples, desc="Computing (F)IR Luminosity...."):
                L_TIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=dust_temp_fixed, dust_beta=sample[1],
                                                                solid_angle=solid_angle,optically_thick_regime=optically_thick_regime, lum='tir'))
                L_FIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=dust_temp_fixed, dust_beta=sample[1],
                                                                solid_angle=solid_angle,optically_thick_regime=optically_thick_regime, lum='fir'))
    elif nparams == 3:
        param_names = ["dust_mass", "dust_temp", "dust_beta", 'μTIR x 10^13', 'μFIR x 10^13']
        for sample in tqdm(flat_samples, desc="Computing (F)IR Luminosity...."):
            L_TIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=sample[2],
                                                            solid_angle=solid_angle, optically_thick_regime=optically_thick_regime,lum='tir'))
            L_FIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=sample[2],
                                                            solid_angle=solid_angle, optically_thick_regime=optically_thick_regime,lum='fir'))
    elif nparams == 4:
        param_names = ["dust_mass", "dust_temp", "dust_beta", 'solid_angle', 'μTIR x 10^13', 'μFIR x 10^13']
        for sample in tqdm(flat_samples, desc="Computing (F)IR Luminosity...."):
            L_TIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),
                                                            dust_temp=sample[1], dust_beta=sample[2],solid_angle=sample[3], optically_thick_regime=optically_thick_regime,lum='tir'))
            L_FIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=sample[2],
                                                            solid_angle=sample[3], optically_thick_regime=optically_thick_regime,lum='fir'))


    flat_samples = np.column_stack((flat_samples, np.asarray(L_TIR_samples) / 1e13, np.asarray(L_FIR_samples) / 1e13))
    for i, name in enumerate(param_names):
        param_samples = flat_samples[:, i]  # Extract samples for parameter
        p16, median, p84 = np.percentile(param_samples, percentiles)  # Compute percentiles
        stats[name] = {
            "median": median,
            "lower_1sigma": median - p16,
            "upper_1sigma": p84 - median,
        }


    print("")
    print("EMCEE Fit Values")
    for name, values in stats.items():
        if name == 'min_chi2':
            print(f"{name}: {values['median']:.2f}")
        elif name.lower() == 'dust_mass':
            print(f"{name} x 10^9: {values['median']/1e9:.2f} (+{values['upper_1sigma']/1e9:.2f}, -{values['lower_1sigma']/1e9:.2f})")
        else:
            print(f"{name}: {values['median']:.2f} (+{values['upper_1sigma']:.2f}, -{values['lower_1sigma']:.2f})")


    if gmf>1.:
        flat_samples_gmf_ir = np.column_stack(((np.asarray(L_TIR_samples) / 1e13)/gmf, (np.asarray(L_FIR_samples) / 1e13)/gmf))
        gmf_ir_names = ['TIR x 10^13', 'FIR x 10^13']
        for i, name in enumerate(gmf_ir_names):
            param_samples = flat_samples_gmf_ir[:, i]  # Extract samples for parameter
            p16, median, p84 = np.percentile(param_samples, percentiles)  # Compute percentiles
            stats[name] = {
                "median": median,
                "lower_1sigma": median - p16,
                "upper_1sigma": p84 - median,
            }
        print("")
        print(f'μ = {gmf}')
        for name, values in stats.items():
            if name == 'TIR x 10^13' or name == 'FIR x 10^13':
                print(f"{name}: {values['median']:.2f} (+{values['upper_1sigma']:.2f}, -{values['lower_1sigma']:.2f})")

    print("")
    return stats



def mbb_best_fit_flux(nu,z,stats: dict = None, dust_mass_default: float = 1e8, dust_temp_default: float= 35, dust_beta_default: float=1.6,
            solid_angle_default: float = 0.0, optically_thick_regime=False,output_unit_mjy=True):

    #I'm using the 50th percentile values instead of samples[np.argmax(log_probabilities)] since the
    # posterior distribution is fairly symmetrical in nature (close to a gaussian) and, hence, the values will be similar
    # to each other

    # Retrieve values from stats or use defaults
    dust_mass_median = stats.get('dust_mass', {}).get('median', dust_mass_default)
    dust_temp_median = stats.get('dust_temp', {}).get('median', dust_temp_default)
    dust_beta_median = stats.get('dust_beta', {}).get('median', dust_beta_default)
    solid_angle_median = stats.get('solid_angle', {}).get('median', solid_angle_default)

    print("")
    print("Parameters Used for Best Fit Flux")
    print(f"Dust Mass: {dust_mass_median/1e9:.2f} x 10^9 M_solar (default: {dust_mass_default})")
    print(f"Dust Temperature: {dust_temp_median:.2f} K (default: {dust_temp_default})")
    print(f"Dust Beta: {dust_beta_median:.2f} (default: {dust_beta_default})")
    if optically_thick_regime:
        print(f"Solid Angle: {solid_angle_median:.3f} arcsec^2 (default: {solid_angle_default})")


    flux_fit = utils.dust_s_obs(nu_obs=nu,
                                  z=z,
                                  dust_mass=utils.mass_kgs_solar_conversion(dust_mass_median, unit_of_input_mass='solar'),
                                  dust_temp=dust_temp_median,
                                  beta=dust_beta_median,
                                  solid_angle=solid_angle_median,
                                  optically_thick_regime=optically_thick_regime,
                                  output_unit_mjy=output_unit_mjy)

    return flux_fit








print("MBB.py")

"""

print("PSSJ2322")

#PSSJ2322+1944
#Last two values are from Stacey et al. 2018 (https://ui.adsabs.harvard.edu/abs/2018MNRAS.476.5075S/abstract)

j2322_freq = np.array([1.4,90, 201.78,225,231, 353, 660, 856.54, 1875, 4300]) #IN GHz
j2322_wave = utils.ghz_to_mum(j2322_freq)
j2322_flux = np.array([9.8e-5, 0.4e-3, 5.79e-3,0.0075,0.0096,0.0225,0.075 , 79e-3, 0.0434, 0.0137]) *1e3 #Converting to mJy
j2322_flux_err = np.array([1.5e-5, 0.25e-3, 0.77e-3,0.0013, 0.0005, 0.0025, 0.019, 11e-3, 0.0084, 0.0061]) *1e3 #Converting to mJy


#Our Value
j2322_sai_freq = np.array([1461.134/(1+4.12)]) #in GHz
j2322_sai_wave = utils.ghz_to_mum(j2322_sai_freq)
j2322_sai_flux =17.4 #flux in mJy
j2322_sai_flux_err = 2.6

plt.scatter(j2322_wave,j2322_flux)
#plt.scatter(utils.ghz_to_mum(96),0.31) #Ignore this point
plt.scatter(j2322_sai_wave,j2322_sai_flux,label='Our Value',marker='*',s=120,color='blue')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("PSSJ2322+1944")
plt.legend()
plt.show()


#For fit
#PSSJ2322+1944
z_pssj=4.12

pssj_freq_ghz = np.array([90,96,201.78,225,231,350,utils.mum_to_ghz(450),utils.mum_to_ghz(350)]) #All obs freq
pssj_freq_wave = utils.mum_to_ghz(pssj_freq_ghz)
pssj_flux = np.array([0.4, 0.31,5.79,7.5,9.6,22.5,75,79])
pssj_flux_err = np.array([0.25,0.08,0.77,1.3,0.5,2.5,19,11])

print(pssj_freq_wave)

my_value_freq_ghz = np.array([1461.134/(1+z_pssj)])
my_value_flux = np.array([17.4])
my_value_flux_err = np.array([2.6])
my_value_wave = utils.ghz_to_mum(my_value_freq_ghz)


pssj_freq_hz = np.concatenate([pssj_freq_ghz,my_value_freq_ghz]) * 1e9
pssj_freq_wave = np.concatenate([pssj_freq_wave,my_value_wave])
pssj_flux = np.concatenate([pssj_flux,my_value_flux]) * 1e-29
pssj_flux_err = np.concatenate([pssj_flux_err,my_value_flux_err]) * 1e-29


x_stats_pssj = mbb_values(nu_obs=pssj_freq_hz,
                     z=z_pssj,
                     gmf=5.3,
                     flux_obs=pssj_flux,
                     flux_err=pssj_flux_err,
                     dust_mass_fixed=0,
                     dust_temp_fixed=0,
                     dust_beta_fixed=1.6,
                     nparams=2,
                     params_type='mt',
                     optically_thick_regime=False,
                     dust_mass_limit=[1e7,1e10],
                     dust_temp_limit=[25,55],
                     initial_guess_values = [1e9,40],
                     nsteps=2000,
                     flat_samples_discarded=300,
                     trace_plots=False,
                     corner_plot=True)


dust_mass_median = x_stats_pssj['dust_mass']['median']
dust_mass_err = np.max(np.asarray([x_stats_pssj['dust_mass']['upper_1sigma'],x_stats_pssj['dust_mass']['lower_1sigma'] ]))

dust_temp_median =  x_stats_pssj['dust_temp']['median']
dust_temp_err = np.max(np.asarray([x_stats_pssj['dust_temp']['upper_1sigma'],x_stats_pssj['dust_temp']['lower_1sigma'] ]))

ir_median = x_stats_pssj['μTIR x 10^13']['median']
ir_err = x_stats_pssj['μTIR x 10^13']['upper_1sigma']

fir_median = x_stats_pssj['FIR x 10^13']['median']
fir_err = x_stats_pssj['FIR x 10^13']['upper_1sigma']

wave = np.linspace(1e1,1e4,10000)
f_pssj = mbb_best_fit_flux(nu=utils.mum_to_ghz(wave)*1e9,
                           z=z_pssj,
                           stats=x_stats_pssj,
                           dust_beta_default=1.6,
                           optically_thick_regime=False,
                           output_unit_mjy=True)


plt.scatter(pssj_freq_wave[:-1], pssj_flux[:-1] * 1e29,color='black')
plt.scatter(my_value_wave, my_value_flux,color='red',marker='*',label='Our Value',s=150)
plt.plot(wave, f_pssj, label=f'Best Fit:'
                            f'\nDust Mass = {ufloat(dust_mass_median,dust_mass_err)} L⊙'
                            f'\nDust Temp = {ufloat(dust_temp_median,dust_temp_err)} K'
                            f'\nBeta = {1.6} (Fixed)'
                            f'\nL_FIR = {ufloat(fir_median,fir_err)*1e13} L⊙')

plt.xlim(1e1,1e4)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("PSSJ2322+1944")
plt.legend()
plt.show()
"""


"""
print("J2310+1855")
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

#Shao, Y., Wang, R., Carilli, C. L., et al. 2019, ApJ, 876, 99
#Herschel SPIRE and PACS
shao_freq = np.array([856.549, 1199.169 , 1873.703, 2997.924]) # IN GHz
shao_wave = utils.ghz_to_mum(shao_freq) #convert to micrometer
shao_flux = np.array([ 22, 19.9, 13.2, 6.5]) #in mJy
shao_flux_err = np.array([ 6.9, 6.0, 2.8, 1.2])

#Our Value
sai_freq = np.array([1461.134/(1+6.0035)]) #in GHz
sai_wave = utils.ghz_to_mum(sai_freq)
sai_flux = np.array([5.5]) #flux in mJy
sai_flux_err = np.array([0.8])#flux in mJy

plt.scatter(wang_wave/(1+6.0035),wang_flux,label='Wang et al. 2008')
plt.scatter(hash_wave/(1+6.0035),hash_flux.n,label='Hashimoto et al. 2018')
plt.scatter(tripodi_wave/(1+6.0035),tripodi_flux,label='Tripodi et al. 2022')
plt.scatter(shao_wave/(1+6.0035),shao_flux,label='Shao et al. 2019')
#plt.scatter(butler_wave,butler_flux.n,label='Butler et al. 2023')

plt.scatter(sai_wave/(1+6.0035),sai_flux,label='Our Value',marker='*',s=120,color='blue')

plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Rest Wavelength [$\mu$m]") #Remove /(1+6.0035) for Observed frame
plt.ylabel("Flux Density [mJy]")
plt.title("J2310-1855")
plt.legend()
plt.show()


j2310_hz = np.concatenate([wang_freq,tripodi_freq,sai_freq]) * 1e9
j2310_wave = np.concatenate([wang_wave,tripodi_wave,sai_wave])
j2310_flux = np.concatenate([wang_flux,tripodi_flux,sai_flux]) * 1e-29
j2310_flux_err = np.concatenate([wang_flux_err,tripodi_flux_err,sai_flux_err])* 1e-29

#For MBB fit
z_j2310 = 6.0035
size = np.mean(np.array([0.261 * 0.171, 0.345 * 0.212, 0.263 * 0.212, 0.214 * 0.189, 0.190 * 0.180,  0.456 * 0.422, 0.233 * 0.220, 0.330 * 0.246, 0.289 * 0.229,  0.318 *0.229]))

x_stats_j2310 = mbb_values(nu_obs=j2310_hz,
                     z=z_j2310,
                     gmf=1,
                     flux_obs=j2310_flux,
                     flux_err=j2310_flux_err,
                     solid_angle=size,
                     dust_mass_fixed=0,
                     dust_temp_fixed=0,
                     dust_beta_fixed=0,
                     nparams=3,
                     params_type='mt',
                     optically_thick_regime=True,
                     dust_mass_limit=[1e7,1e10],
                     dust_temp_limit=[50,80],
                     dust_beta_limit=[1.5,2.5],
                     initial_guess_values = [1e8,67,1.8],
                     nsteps=500,
                     flat_samples_discarded=300,
                     trace_plots=True,
                     corner_plot=True)


wave = np.linspace(1e1,1e4,10000)
f_j2310 = mbb_best_fit_flux(nu=utils.mum_to_ghz(wave)*1e9,
                           z=z_j2310,
                           stats=x_stats_j2310,
                           solid_angle_default=size,
                           optically_thick_regime=True,
                           output_unit_mjy=True)



plt.scatter(wang_wave,wang_flux,label='Wang et al. 2008')
plt.scatter(hash_wave,hash_flux.n,label='Hashimoto et al. 2018')
plt.scatter(tripodi_wave,tripodi_flux,label='Tripodi et al. 2022')
plt.scatter(shao_wave,shao_flux,label='Shao et al. 2019')
plt.scatter(sai_wave,sai_flux,color='red',marker='*',label='Our Value',s=150)
plt.plot(wave,f_j2310)

plt.xlim(1e1,1e4)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("J2310+1855")
plt.legend()
plt.show()


plt.scatter(j2310_wave[:-1], j2310_flux[:-1] * 1e29,color='black')
plt.scatter(sai_wave,sai_flux,color='red',marker='*',label='Our Value',s=150)
plt.plot(wave,f_j2310)

plt.xlim(1e1,1e4)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("J2310+1855")
plt.legend()
plt.show()
"""


print("J2054-0005")


#J2054-0005

#SDSS from NED
df = pd.read_csv(r'NED_photo_points/sdss_j2054_0005_table_photandseds_NED.csv')
print(df.columns)

#df.drop([0,2,4,6],inplace=True) #Dropping Pan-STARRS1 Observations and K(keck)

df.drop([0,2,4,5,6],inplace=True) #Dropping Pan-STARRS1 Observations and K(keck)


freq_ned = df['Frequency'].to_numpy()/1e9 #Convert Frequncies to GHz
wave_ned = utils.ghz_to_mum(freq_ned) #Wavelength in micrometers
flux_ned = df['Flux Density'].to_numpy()*1e3 #Convert to mJy
flux_err_ned = df['Upper limit of uncertainty'].to_numpy()*1e3 #Convert to mJy
references_ned = df['Refcode']


print(df[['Observed Passband','Photometry Measurement','Uncertainty','Flux Density']])
#print(freq_ned)
#print(wave_ned)
print(ufloat(flux_ned[0],flux_err_ned[0]))
print(ufloat(flux_ned[1],flux_err_ned[1]))
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

"""
#Wang et al. 2011, Wang et al. 2008, Tripodi et al 2024,

wwt_freq = np.array([1.4,92.26,250, 262.6, 263.93, 488.31 , 674.97])
wwt_wave = utils.ghz_to_mum(wwt_freq)
wwt_flux = np.array([17e-3, 0.082, 2.38,2.93,3.08,11.71,9.87]) #Flux in mJy
wwt_flux_err = np.array([23e-3,0.009,0.53,0.07,0.03,0.11,0.94]) #Flux in mJy
"""

#Wang et al. 2011, Wang et al. 2008, Tripodi et al 2024,

wwt_freq = np.array([92.26,250, 262.6, 263.93, 674.97])
wwt_wave = utils.ghz_to_mum(wwt_freq)
wwt_flux = np.array([ 0.082, 2.38,2.93,3.08,9.87]) #Flux in mJy
wwt_flux_err = np.array([0.009,0.53,0.07,0.03,0.94]) #Flux in mJy

#Hashimoto et al. 2018 (https://arxiv.org/pdf/1811.00030)
#Our data probe dust continuum emission at the rest-frame wavelength, λ_rest, of ≈ 87 μm
hash_wave = np.array([87*(1+6.0391)]) #In micrometers
hash_freq = utils.mum_to_ghz(hash_wave)
hash_flux = np.array([10.35])#Flux in mJy
hash_flux_err = np.array([0.15])#Flux in mJy



#Salak et al. 2024 (https://iopscience.iop.org/article/10.3847/1538-4357/ad0df5/pdf)
#λ_rest, of ≈ 123 μm
salak_wave = np.array([123*(1+6.0391)]) #In micrometers
salak_freq = utils.mum_to_ghz(salak_wave)
salak_flux = np.array([5.723]) #Flux in mJy
salak_flux_err = np.array([0.009]) #Flux in mJy


#Our Value
sai_freq = np.array([1461.134/(1+6.0391)]) #in GHz
sai_wave = utils.ghz_to_mum(sai_freq)
sai_flux = np.array([0.8]) #flux in mJy
sai_flux_err = np.array([0.1]) #flux in mJy


#Plot SED
plt.scatter(wave_ned,flux_ned,label='SDSS-NED')
plt.scatter(w1_wave,w1_flux.n,label='W1-Blain et al. 2013')
plt.scatter(leipski_wave,leipski_flux,label='Leipski et al. 2014')
plt.scatter(wwt_wave,wwt_flux,label='WWT')
plt.scatter(hash_wave,hash_flux,label='Hashimoto et al. 2018')
plt.scatter(salak_wave,salak_flux,label='Salak et al. 2024')

plt.scatter(sai_wave,sai_flux,label='Our Value',marker='*',s=120,color='blue')

axvline(63 * (1+6.0391))
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("J2054-0005")
plt.legend()
plt.show()


z_j2054 = 6.0391

#size: Hashimoto et al. 2019, Wang et al. 2013, Salak et al. 2024, ISHII et al. 2024 (https://arxiv.org/pdf/2408.09944)
J2054_size = np.mean(np.array([0.23*0.15,0.27*0.26,0.1567*0.1321,0.42*0.23]))

J2054_hz = np.concatenate([wwt_freq,hash_freq,salak_freq,sai_freq]) * 1e9
J2054_wave = np.concatenate([wwt_wave,hash_wave,salak_wave,sai_wave])
J2054_flux = np.concatenate([wwt_flux,hash_flux,salak_flux,sai_flux]) * 1e-29
J2054_flux_err = np.concatenate([wwt_flux_err,hash_flux_err,salak_flux_err,sai_flux_err])* 1e-29

x_stats_j2054 = mbb_values(nu_obs=J2054_hz,
                     z=6.0391,
                     gmf=1,
                     flux_obs=J2054_flux,
                     flux_err=J2054_flux_err,
                     solid_angle=J2054_size,
                     dust_mass_fixed=0,
                     dust_temp_fixed=0,
                     dust_beta_fixed=0,
                     nparams=3,
                     params_type='mt',
                     optically_thick_regime=True,
                     dust_mass_limit=[1e7,1e9],
                     dust_temp_limit=[50,80],
                     dust_beta_limit=[1.5,3.0],
                     initial_guess_values = [1e8,64,2.1],
                     nsteps=2000,
                     flat_samples_discarded=300,
                     trace_plots=True,
                     corner_plot=True)

dust_mass_median = x_stats_j2054['dust_mass']['median']
dust_mass_err = np.max(np.asarray([x_stats_j2054['dust_mass']['upper_1sigma'],x_stats_j2054['dust_mass']['lower_1sigma'] ]))

dust_temp_median =  x_stats_j2054['dust_temp']['median']
dust_temp_err = np.max(np.asarray([x_stats_j2054['dust_temp']['upper_1sigma'],x_stats_j2054['dust_temp']['lower_1sigma'] ]))

dust_beta_median = x_stats_j2054['dust_beta']['median']
dust_beta_err = np.max(np.asarray([x_stats_j2054['dust_beta']['upper_1sigma'],x_stats_j2054['dust_beta']['lower_1sigma'] ]))

ir_median = x_stats_j2054['μTIR x 10^13']['median']
ir_err = x_stats_j2054['μTIR x 10^13']['upper_1sigma']


wave = np.linspace(1e1,1e4,10000)
f_j2310 = mbb_best_fit_flux(nu=utils.mum_to_ghz(wave)*1e9,
                           z=z_j2054,
                           stats=x_stats_j2054,
                           solid_angle_default=J2054_size,
                           optically_thick_regime=True,
                           output_unit_mjy=True)



plt.scatter(wwt_wave,wwt_flux,label='WWT')
plt.scatter(hash_wave,hash_flux,label='Hashimoto et al. 2018')
plt.scatter(salak_wave,salak_flux,label='Salak et al. 2024')
plt.scatter(sai_wave,sai_flux,label='Our Value',marker='*',s=120,color='red')

plt.plot(wave,f_j2310,label=f'Best Fit:'
                            f'\nDust Mass = {ufloat(dust_mass_median,dust_mass_err)} M⊙'
                            f'\nDust Temp = {ufloat(dust_temp_median,dust_temp_err)} K'
                            f'\nBeta = {ufloat(dust_beta_median,dust_beta_err)}'
                            f'\nL_IR = {ufloat(ir_median,ir_err)*1e13} L⊙')

plt.xlim(1e1,1e4)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
plt.title("J2054-0005")
plt.legend()
plt.show()

#exit()
