import emcee
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from interferopy.cube import Cube,MultiCube
import interferopy.tools as iftools
from scipy.constants import c
from lmfit import Model
from uncertainties import ufloat

def only_spectrum():
    filename=(r"/home/sai/Desktop/data_cubes/id141.fits")
    cub = Cube(filename)
    ra,dec=(350.5299004,19.7393507)
    r=0
    x= cub.spectrum(ra=ra, dec=dec, radius=r, calc_error=True)
    flux=x[0]
    err=x[1]

    fig, ax = plt.subplots(figsize=(4.8, 3))
    ax.set_title("Integrated aperture spectrum")
    ax.plot(cub.freqs, flux , color="black", drawstyle='steps-mid', lw=0.75, label="Spectrum within r=" + str(r) + '"')
    ax.plot(cub.freqs, err, color="gray", ls=":", label=r"1$\sigma$ error")  # 1sigma error
    ax.tick_params(direction='in', which="both")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Aperture flux density (mJy)")
    ax.legend(frameon=False)
    plt.show()

#print(only_spectrum())


yat="Flux [mJy]"
xat=r"Velocity [km s$^{-1}$]"
xat2="Observed Frequency [GHz]"


def model(theta,vel):
    F, I_1, x_0_1, fwhm_1 = theta

    x=vel

    s_1 = fwhm_1 / 2.35


    m = I_1 * np.exp(-0.5 * (((x - x_0_1) / s_1) ** 2))
    n = (s_1 * np.sqrt(2 * np.pi))


    y = F + (m / n)

    return y

def gauss_2(theta,vel):
    F, I_1, x_0_1, fwhm_1, I_2, x_0_2, fwhm_2 = theta

    x=vel

    s_1 = fwhm_1 / 2.35
    s_2 = fwhm_2 / 2.35

    m = I_1 * np.exp(-0.5 * (((x - x_0_1) / s_1) ** 2))
    n = (s_1 * np.sqrt(2 * np.pi))

    m_2 = I_2 * np.exp(-0.5 * (((x - x_0_2) / s_2) ** 2))
    n_2 = (s_2 * np.sqrt(2 * np.pi))

    y = F + (m / n) + (m_2 / n_2)

    return y


def guass(x, F, I, x_0, s):
    m = I * np.exp(-0.5 * (((x - x_0) / s) ** 2))
    n = (s * np.sqrt(2 * np.pi))
    y = F + (m / n)
    return y


def spectrum(path):
    f = open(path, "r")
    l = f.readlines()

    vel=[]
    freq=[]
    flux=[]
    for i in range(1,len(l)):
        z=l[i].split()
        vel.append(float(z[0]))
        freq.append(float(z[1]))
        flux.append(float(z[2]))


    f.close()
    return np.asarray(vel),np.asarray(freq),np.asarray(flux)


def mean_spectrum_err(err):
    cal_err = 0.15 #15% Calibration Error
    return np.sqrt((err**2) + (err*cal_err)**2)


vel_gn20,freq_gn20,flux_gn20 = spectrum("saimurali/veb2-gn20.dat")
vel_id141,freq_id141,flux_id141 = spectrum("saimurali/vcb2-id141.dat")
vel_hdf,freq_hdf,flux_hdf = spectrum("saimurali/vdb2-hdf850.dat")

vel_2322,freq_2322,flux_2322 = spectrum("saimurali/vbb2-wbb7-2322.dat")
vel_2054,freq_2054,flux_2054 = spectrum("saimurali/w16ef002-2054.dat")
vel_2310,freq_2310,flux_2310 = spectrum("saimurali/w16ef001-2310.dat")


gm =Model(guass)


params_velocity_pssj = gm.make_params(F=0.01 * 1e3, I=1.7, x_0=0, s=50)

out_velocity_gn20 = gm.fit(flux_gn20, params_velocity_pssj, method='Leastsq', x=vel_gn20)
out_velocity_2322 = gm.fit(flux_2322, params_velocity_pssj, method='Leastsq', x=vel_2322)


""""plt.step(vel_2322,flux_2322)
plt.plot(vel_2322,out_velocity_2322.best_fit)
plt.show()"""


theta_id141 = [  57.09613459, 7889.48188901, -186.98081971,  478.57906311, 1899.91708063,293.5149512,   152.33310762]
best_fit_model = gauss_2(theta_id141,vel_id141)
"""plt.step(vel_id141,flux_id141)
plt.plot(vel_id141,best_fit_model,label='Highest Likelihood Model')
plt.xlabel("Velocity")
plt.ylabel("Flux [mJy]")
plt.show()"""

err = 1.4
print(err + (0.15*err))
print(mean_spectrum_err(err))

#exit()



fig, ((ax1, ax2, ax3), (ax4,ax5,ax6)) = plt.subplots(2,3, figsize=(8, 8), tight_layout=True)


ax1.fill_between(vel_gn20,flux_gn20,16.2,step="pre", alpha=0.2,color="orange")
ax1.step(vel_gn20,flux_gn20,drawstyle="steps-pre",color="orange")
ax1.errorbar(vel_gn20[0],21,yerr=1.8 ,capsize=3,color='grey',marker='o')
ax1.set_ylabel("Flux [mJy]",size=15)
ax1.plot(vel_gn20,model([  16.06781383, 2415.0972588 ,   -4.24083208,  720.66451899],vel_gn20),color='darkblue')
ax1_1 = ax1.twiny()
ax1_1.step(freq_gn20,flux_gn20,color="orange")
ax1_1.invert_xaxis()
ax1_1.set_xlabel(xat2,size=15)
ax1_1.set_box_aspect(1)
ax1.text(-2100,21.3,"GN20",size=15)



ax2.fill_between(vel_id141,flux_id141, 57.09613459,step="pre", alpha=0.2,color="orange")
ax2.step(vel_id141,flux_id141,drawstyle="steps-pre",color="orange")
ax2.errorbar(vel_id141[0],74,yerr= 3.3 ,capsize=3,color='grey',marker='o')
ax2.plot(vel_id141,best_fit_model,color='darkblue')
ax2_1=ax2.twiny()
ax2_1.step(freq_id141,flux_id141,color="orange")
ax2_1.invert_xaxis()
ax2_1.set_xlabel(xat2,size=15)
ax2_1.set_box_aspect(1)
ax2.text(-1800,72.5,"ID141",size=15)


ax3.fill_between(vel_hdf,flux_hdf,2.0,step="pre", alpha=0.2,color="orange")
ax3.step(vel_hdf,flux_hdf,drawstyle="steps-pre",color="orange")
ax3.errorbar(vel_hdf[0],5,yerr=1.1,capsize=3,color='grey',marker='o')
ax3.errorbar(0,-0.7,xerr=300,capsize=15,color='black',elinewidth=4)
ax3.axhline(2.0,linestyle='--', color='darkblue')
ax3_1=ax3.twiny()
ax3_1.step(freq_hdf,flux_hdf,color='orange')
ax3_1.invert_xaxis()
ax3_1.set_xlabel(xat2,size=15)
ax3_1.set_box_aspect(1)
ax3.text(-2900,5.2,"HDF850.1",size=15)


#qso

ax4.fill_between(vel_2322,flux_2322,out_velocity_2322.params['F'].value,step="pre", alpha=0.2,color="red")
ax4.step(vel_2322,flux_2322,drawstyle="steps-pre",color="red")
ax4.errorbar(vel_2322[0],24,yerr=2.8,capsize=3,color='grey',marker='o')
ax4.plot(vel_2322,out_velocity_2322.best_fit,color='darkblue')
ax4.set_xlabel(xat,size=15)
ax4.set_ylabel(yat,size=15)
ax4_1=ax4.twiny()
ax4_1.step(freq_2322,flux_2322,color='red')
ax4_1.invert_xaxis()
ax4_1.set_box_aspect(1)
ax4.text(-2250,26,"PSSJ2322+1944",size=15)


ax5.fill_between(vel_2054,flux_2054,0.8,step="pre", alpha=0.2,color="red")
ax5.step(vel_2054,flux_2054,color='red')
ax5.errorbar(vel_2054[3],3.1,yerr=1.1,capsize=3,color='grey',marker='o')
ax5.errorbar(0,-2,xerr=400,capsize=15,color='black',elinewidth=4)
ax5.axhline(0.8,linestyle='--', color='darkblue')
ax5.set_xlabel(xat,size=15)
ax5_1=ax5.twiny()
ax5_1.step(freq_2054,flux_2054,color='red')
ax5_1.invert_xaxis()
ax5_1.set_box_aspect(1)
ax5.text(-1800,3.8,"J2054-0005",size=15)

ax6.fill_between(vel_2310,flux_2310,5.5,step="pre", alpha=0.2,color="red")
ax6.step(vel_2310,flux_2310,color='red')
ax6.errorbar(vel_2310[0],8,yerr=1.4,capsize=3,color='grey',marker='o')
ax6.errorbar(0,2,xerr=200,capsize=15,color='black',elinewidth=4)
ax6.axhline(5.5,linestyle='--', color='darkblue')
ax6.set_xlabel(xat,size=15)
ax6_1=ax6.twiny()
ax6_1.step(freq_2310,flux_2310,color='red')
ax6_1.invert_xaxis()
ax6_1.set_box_aspect(1)
ax6.text(-3400,9,"J2310+1855",size=15)


plt.show()
























exit()

def guass_2(x, F, I_1, x_0_1, fwhm_1,I_2, x_0_2, fwhm_2):


    s_1 = fwhm_1/2.35
    s_2 = fwhm_2/2.35

    m = I_1 * np.exp(-0.5 * (((x - x_0_1) / s_1) ** 2))
    n = (s_1 * np.sqrt(2 * np.pi))

    m_2 = I_2 * np.exp(-0.5 * (((x - x_0_2) / s_2) ** 2))
    n_2 = (s_2 * np.sqrt(2 * np.pi))

    y = F + (m / n) + (m_2/n_2)
    return y

def spectrum(path):
    f = open(path, "r")
    l = f.readlines()

    vel=[]
    freq=[]
    flux=[]
    for i in range(1,len(l)):
        z=l[i].split()
        vel.append(float(z[0]))
        freq.append(float(z[1]))
        flux.append(float(z[2]))


    f.close()
    return np.asarray(vel),np.asarray(freq),np.asarray(flux)


vel_id141,freq_id141,flux_id141 = spectrum("/home/sai/saimurali/vcb2-id141.dat")

plt.step(vel_id141,flux_id141)
plt.show()

flux_id141 = flux_id141 * 1

def model(theta,vel):
    F, I_1, x_0_1, fwhm_1, I_2, x_0_2, fwhm_2 = theta

    x=vel

    s_1 = fwhm_1 / 2.35
    s_2 = fwhm_2 / 2.35

    m = I_1 * np.exp(-0.5 * (((x - x_0_1) / s_1) ** 2))
    n = (s_1 * np.sqrt(2 * np.pi))

    m_2 = I_2 * np.exp(-0.5 * (((x - x_0_2) / s_2) ** 2))
    n_2 = (s_2 * np.sqrt(2 * np.pi))

    y = F + (m / n) + (m_2 / n_2)

    return y


def lnlike(theta, x, y, yerr):
    return -0.5 * np.sum(((y - model(theta, x))/yerr) ** 2)



def lnprior(theta):
    F, I_1, x_0_1, fwhm_1, I_2, x_0_2, fwhm_2 = theta
    if 50<F<60 and I_1>1 and -500<x_0_1<0 and 200<fwhm_1<600 and I_2>0 and 10<x_0_2<500 and 100<fwhm_2<300:
        return 0.0
    return -np.inf

def lnprob(theta, x, y, yerr):
    lp = lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)


rms_id141 = np.ones(len(flux_id141)) * 3.3

data = (vel_id141,flux_id141,rms_id141)
nwalkers = 128
niter = 500
initial = np.array([55,2.0,-200,300, 2,200,150])
ndim = len(initial)
p0 = [np.array(initial) + 1e-7 * np.random.randn(ndim) for i in range(nwalkers)]

def main(p0,nwalkers,niter,ndim,lnprob,data):
    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=data)

    print("Running burn-in...")
    p0, _, _ = sampler.run_mcmc(p0, 100)
    sampler.reset()

    print("Running production...")
    pos, prob, state = sampler.run_mcmc(p0, niter)

    return sampler, pos, prob, state


sampler, pos, prob, state = main(p0,nwalkers,niter,ndim,lnprob,data)


def plotter(sampler,vel=vel_id141,flux=flux_id141):
    plt.step(vel,flux)
    samples = sampler.flatchain
    for theta in samples[np.random.randint(len(samples), size=100)]:
        plt.plot(vel, model(theta, vel), color="r", alpha=0.1)
    plt.xlabel("Velocity")
    plt.ylabel("Flux [mJy]")
    plt.show()

plotter(sampler)


samples = sampler.flatchain
theta_max  = samples[np.argmax(sampler.flatlnprobability)]
best_fit_model = model(theta_max,vel_id141)

plt.step(vel_id141,flux_id141)
plt.plot(vel_id141,best_fit_model,label='Highest Likelihood Model')
plt.xlabel("Velocity")
plt.ylabel("Flux [mJy]")
plt.show()

flat_samples = sampler.get_chain(discard=100, thin=15, flat=True)

print(theta_max)

labels = ['F', "I_1", "x_0_1", "fwhm_1", "I_2", "x_0_2", "fwhm_2"]

fig = corner.corner(flat_samples,show_titles=True,labels=labels,plot_datapoints=True,quantiles=[0.16, 0.5, 0.84])
plt.show()







