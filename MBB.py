import numpy as np
import matplotlib.pyplot as plt
import utils as utils
import emcee
from scipy import integrate, constants
import astropy.units as u
import corner
import interferopy.tools as iftools

def optically_thick_integral(z,dust_mass,dust_temp,dust_beta,solid_angle,lower_limit,upper_limit):


    integral = integrate.quad(lambda x: utils.dust_s_obs(x,
                                              z=z,
                                              dust_mass=dust_mass,
                                              dust_temp=dust_temp,
                                              beta=dust_beta,
                                              optically_thick_regime=True,
                                              solid_angle=solid_angle,
                                              output_unit_mjy=True), lower_limit, upper_limit)[0]  # mJy * Hz)
    # Taken from line 558 of https://github.com/roberta96/EOS-Dustfit/blob/main/colddust_sed_models/cdsed_modelling/results_plot.py
    integral = (integral * u.mJy * u.Hz).to(u.mJy * u.GHz)  # Convert to mJy*GHz
    dl = (utils.luminosity_distance(z) * u.Mpc).to(u.cm)
    integral = (4 * np.pi * (dl.value ** 2) * integral.value * 1e-26 * 1e9)  # (cm2)*(mJy*GHz)*1e-26*1e9 = erg/s
    L_sun = 3.826 * 1e33  # erg/s
    return integral / L_sun  # Lsun


def optically_thin_integral(dust_mass,dust_temp,dust_beta,lower_limit,upper_limit):

    integral = integrate.quad(lambda x: utils.dust_luminosity_one_freq_value(x,
                                                                             mass_dust=dust_mass,
                                                                             temp_dust=dust_temp,
                                                                             beta=dust_beta),lower_limit,upper_limit)[0]
    return integral * u.W.to(u.solLum) # Lsun



def dust_integrated_luminosity(z,dust_mass,dust_temp,dust_beta,solid_angle,optically_thick_regime=False,lum='both'):
    tir_lower_limit = constants.c / (1000e-6)  # Convert to frequency for integration
    tir_upper_limit = constants.c / (8e-6)

    fir_lower_limit = constants.c / (122.5e-6)  # Convert to frequency for integration
    fir_upper_limit = constants.c / (42.5e-6)

    if optically_thick_regime == True:
        tir_lum = optically_thick_integral(z,dust_mass,dust_temp,dust_beta,solid_angle,tir_lower_limit,tir_upper_limit) # Lsun
        fir_lum = optically_thick_integral(z, dust_mass, dust_temp, dust_beta, solid_angle, fir_lower_limit,fir_upper_limit) # Lsun
    else:
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
            trace_plots
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
    else:
        #initial_guesses = np.asarray(initial_guess_values[:ndim])
        initial_guesses = np.concatenate( (np.asarray([initial_guess_values[0] / 10**dust_mass_rescale]), np.asarray(initial_guess_values[1:ndim])),axis=0)
        trace_label = ['dust_mass', 'dust_temp','dust_beta', 'solid_angle'][:ndim]

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
        corner_plot: bool = False
        ):

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
            trace_plots)

    flat_samples = sampler.get_chain(discard=flat_samples_discarded, thin=10, flat=True)

    log_probs = sampler.get_log_prob(discard=flat_samples_discarded, thin=10, flat=True)
    chi2_values = -2 * log_probs
    min_chi2 = np.min(chi2_values)


    percentiles = [16, 50, 84]  # 1-sigma percentiles
    stats = {}

    stats['min_chi2'] = {
        "median": min_chi2,
        "lower_1sigma": 0,
        "upper_1sigma": 0,
        }


    # Calculate marginalized L_IR
    L_TIR_samples = []
    L_FIR_samples = []

    if corner_plot == True and nparams>1:
        print("Generating corner plots...")

    if nparams == 1:
        param_names = [r"dust_mass", 'μTIR x 10^13', 'μFIR x 10^13']
        for sample in flat_samples:
            L_TIR_samples.append(dust_integrated_luminosity(z=z,dust_mass=utils.mass_kgs_solar_conversion(sample[0],'solar'),dust_temp=dust_temp_fixed,dust_beta=dust_beta_fixed,
                                                            solid_angle=solid_angle,optically_thick_regime=optically_thick_regime,lum='tir'))
            L_FIR_samples.append(dust_integrated_luminosity(z=z,dust_mass=utils.mass_kgs_solar_conversion(sample[0],'solar'),dust_temp=dust_temp_fixed,dust_beta=dust_beta_fixed,
                                                            solid_angle=solid_angle,optically_thick_regime=optically_thick_regime,lum='fir'))

    elif nparams == 2:
        if params_type.lower() == 'mt':
            param_names = ["dust_mass", "dust_temp", 'μTIR x 10^13', 'μFIR x 10^13']
            corner_labels = [r"log($M_{\mathrm{dust}}$) [$M_\odot$]", r"$T_{\mathrm{dust}}$ [K]"]
            for sample in flat_samples:
                L_TIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=dust_beta_fixed,
                                                                solid_angle=solid_angle,optically_thick_regime=optically_thick_regime, lum='tir'))
                L_FIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=dust_beta_fixed,
                                                                solid_angle=solid_angle,optically_thick_regime=optically_thick_regime, lum='fir'))

        elif params_type.lower() == 'mb':
            param_names = ["dust_mass", "dust_beta", 'μTIR x 10^13', 'μFIR x 10^13']
            corner_labels = [r"log($M_{\mathrm{dust}}$) [$M_\odot$]", r"$\beta_{\mathrm{dust}}$"]
            for sample in flat_samples:
                L_TIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=dust_temp_fixed, dust_beta=sample[1],
                                                                solid_angle=solid_angle,optically_thick_regime=optically_thick_regime, lum='tir'))
                L_FIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=dust_temp_fixed, dust_beta=sample[1],
                                                                solid_angle=solid_angle,optically_thick_regime=optically_thick_regime, lum='fir'))
    elif nparams == 3:
        param_names = ["dust_mass", "dust_temp", "dust_beta", 'μTIR x 10^13', 'μFIR x 10^13']
        corner_labels = [r"log($M_{\mathrm{dust}}$) [$M_\odot$]", r"$T_{\mathrm{dust}}$ [K]", r"$\beta_{\mathrm{dust}}$"]
        for sample in flat_samples:
            L_TIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=sample[2],
                                                            solid_angle=solid_angle, optically_thick_regime=optically_thick_regime,lum='tir'))
            L_FIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=sample[2],
                                                            solid_angle=solid_angle, optically_thick_regime=optically_thick_regime,lum='fir'))
    elif nparams == 4:
        param_names = ["dust_mass", "dust_temp", "dust_beta", 'solid_angle', 'μTIR x 10^13', 'μFIR x 10^13']
        corner_labels = [r"log($M_{\mathrm{dust}}$) [$M_\odot$]", r"$T_{\mathrm{dust}}$ [K]", r"$\beta_{\mathrm{dust}}$", r"$\Omega_{\mathrm{S}}$"]
        for sample in flat_samples:
            L_TIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),
                                                            dust_temp=sample[1], dust_beta=sample[2],solid_angle=sample[3], optically_thick_regime=optically_thick_regime,lum='tir'))
            L_FIR_samples.append(dust_integrated_luminosity(z=z, dust_mass=utils.mass_kgs_solar_conversion(sample[0], 'solar'),dust_temp=sample[1], dust_beta=sample[2],
                                                            solid_angle=sample[3], optically_thick_regime=optically_thick_regime,lum='fir'))

    if corner_plot == True and nparams>1:
        corner_flat_samples = flat_samples.copy()
        corner_flat_samples[:, 0] = np.log10(corner_flat_samples[:, 0])  # np.log(dust_mass)
        corner.corner(corner_flat_samples, labels=corner_labels, show_titles=True, plot_datapoints=True,quantiles=[0.16, 0.5, 0.84], title_fmt=".2f")
        plt.show()

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
    # Retrieve values from stats or use defaults
    dust_mass_median = stats.get('dust_mass', {}).get('median', dust_mass_default)
    dust_temp_median = stats.get('dust_temp', {}).get('median', dust_temp_default)
    dust_beta_median = stats.get('dust_beta', {}).get('median', dust_beta_default)
    solid_angle_median = stats.get('solid_angle', {}).get('median', solid_angle_default)

    print("")
    print("Parameters Used for Best Fit Flux")
    print(f"Dust Mass: {dust_mass_median/1e9:.2f} x 10^9 M_solar")
    print(f"Dust Temperature: {dust_temp_median:.2f} K")
    print(f"Dust Beta: {dust_beta_median:.2f}")
    if optically_thick_regime:
        print(f"Solid Angle: {solid_angle_median:.3f} arcsec^2")


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
#GN20
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


x_stats = mbb_values(nu_obs=gn20_freq_hz,
           z=4.0553,
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
f = mbb_best_fit_flux(nu=utils.mum_to_ghz(wave)*1e9,z=4.0553, stats=x_stats,dust_beta_default=1.95,
                      optically_thick_regime=False,output_unit_mjy=True)


plt.scatter(gn20_wave_mum,gn20_flux*1e29,color='black')
plt.scatter(my_value_gn20_wave_mum,my_value_gn20_flux,color='red',label='Our Value')
plt.plot(wave,f,label='EMCEE')

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


plt.scatter(id141_wave,id141_flux * 1e29)
plt.scatter(utils.ghz_to_mum(my_value_freq), my_value_flux,color='red',label='Our Value')
plt.plot(wave,f_id141)

plt.xlim(1e2,5e3)
plt.ylim(1e-4, 10**3)
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r"Observed Wavelength [$\mu$m]")
plt.ylabel("Flux Density [mJy]")
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
exit()

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
plt.title("HDF850.1 Fit")
plt.show()