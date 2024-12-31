import numpy as np
from interferopy.cube import Cube
from matplotlib import colors
import matplotlib.pyplot as plt
import utils


def spectrum(cube, ra: float = None, dec: float = None, radius: float = 0.0,
             contour_mask = None, px: int = None, py: int = None,channel: int = None,
             freq: float = None, calc_error=False):

    #self = Cube(cube)
    self = cube
    if px is None or py is None:
        px, py = self.radec2pix(ra, dec)
    if freq is not None:
        channel = self.freq2pix(freq)


    print("Extracting aperture spectrum.")



    # select pixels within the aperture
    if contour_mask is not None:
        w = contour_mask
    else:
        distances = cube.distance_grid(px, py) * cube.pixsize
        w = distances <= radius

    if channel is not None:
        npix = np.array([np.sum(np.isfinite(self.im[:, :, channel][w]))])
        flux = np.array([np.nansum(self.im[:, :, channel][w]) / self.beamvol[channel]])
        peak_sb = np.nanmax(self.im[:, :, channel][w])
        if calc_error:
            err = np.array(self.rms[channel] * np.sqrt(npix / self.beamvol[channel]))
        else:
            err = np.array([np.nan])

    else:
        flux = np.zeros(self.nch)
        peak_sb = np.zeros(self.nch)
        npix = np.zeros(self.nch)
        for i in range(self.nch):
            flux[i] = np.nansum(self.im[:, :, i][w]) / self.beamvol[i]
            peak_sb[i] = np.nanmax(self.im[:, :, i][w])
            npix[i] = np.sum(np.isfinite(self.im[:, :, i][w]))
            if calc_error:
                err = np.array(self.rms * np.sqrt(npix / self.beamvol))
            else:
                err = np.full_like(flux, np.nan)


    return flux, err










cub_2 = Cube('/home/sai/Downloads/Pisco.cube.50kms.image.fits')
ra, dec = (205.533741, 9.477317341)

mask = utils.create_contour_mask(image='/home/sai/Downloads/Pisco.cii.455kms.image.fits',
                    ra=ra,
                    dec=dec,
                    sigma=2,
                    plot=True)

flux,err = spectrum(cube=cub_2, ra=ra,dec=dec,radius=0.8,calc_error=True)

flux_mask,err_mask = spectrum(cube=cub_2, ra=ra,dec=dec,radius=0., contour_mask= mask,calc_error=True)

fig, (ax1,ax2) = plt.subplots(1,2)

ax1.step(cub_2.freqs, flux * 1e3, label='Round Aperture')
ax1.step(cub_2.freqs,err * 1e3, color='grey')
ax1.set_ylabel("Flux [mJy]")
ax1.legend()
ax2.step(cub_2.freqs,flux_mask * 1e3, label='2-sigma contour')
ax2.step(cub_2.freqs,err_mask * 1e3, color='grey')
ax2.legend()
plt.show()


x=cub_2.spectrum(ra=ra,dec=dec,radius=0.8,calc_error=True)

f=x[0]
e=x[1]

plt.step(cub_2.freqs,f * 1e3)
plt.step(cub_2.freqs,e * 1e3)
plt.show()





exit()



