import interferopy.tools as iftools
from interferopy.cube import Cube
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Ellipse
import numpy as np
from interferopy.cube import Cube
from matplotlib import colors
from matplotlib.path import Path


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




def create_contour_mask(image,ra: float = None, dec: float = None, sigma: float = 1.0,
                        px: int = None, py: int = None, plot=True):

    """

    :param image: Path for line or continuum image
    :param ra: Right ascention in degrees.
    :param dec: Declination in degrees.
    :param sigma: depth of contour (sigma * rms)
    :param px: Right ascention pixel coord.
    :param py: Declination pixel coord.
    :param plot: plot the contour around the source and the corresponding mask
    :return: mask of the contour (used for contour spectral extraction)
    """

    self = Cube(image)
    scale = 1e3
    subim = self.im[:, :, 0] * scale
    if px is None or py is None:
        px, py = self.radec2pix(ra, dec)

    #edgera, edgedec = cub.pix2radec([0, self.shape[0]], [0, self.shape[1]])
    extent = [0, subim.shape[0], 0, subim.shape[1]]
    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)

    rms = self.rms[0] * scale


    fig, ax = plt.subplots(1, 1)
    contour_set = ax.contour(subim.T, extent=extent, levels=np.array([sigma]) * rms)
    plt.close()


    contour_paths = [path for path in contour_set.collections[0].get_paths()]

    min_distance = float('inf')
    nearest_path = None

    for path in contour_paths:
        # Find the minimum distance from (px, py) to the current path vertices
        # px,py = 0,0
        vertices = path.vertices
        distances = np.sqrt((vertices[:, 0] - px) ** 2 + (vertices[:, 1] - py) ** 2)
        if np.min(distances) < min_distance:
            min_distance = np.min(distances)
            nearest_path = path

    mask = np.zeros_like(subim, dtype=bool)

    x = np.linspace(extent[0], extent[1], subim.shape[0])
    y = np.linspace(extent[2], extent[3], subim.shape[0])
    X, Y = np.meshgrid(x, y)

    if nearest_path is not None:
        # Transform image indices to the plot coordinates
        pixel_coords = np.column_stack((X.ravel(), Y.ravel()))
        inside = nearest_path.contains_points(pixel_coords).reshape(subim.shape)
        mask[inside] = True


    if plot == True:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
        ax1.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)
        ax1.contour(subim.T, extent=extent, colors="red", levels=np.array([sigma]) * rms, zorder=1, linewidths=0.5,linestyles="-")
        # ax1.plot(px, py, 'bo', label='Target Pixel')
        ax1.set_title(fr"Original Image with {sigma}$\sigma$ Contours")

        ax2.imshow(mask, extent=extent, origin='lower', cmap='gray')
        ax2.set_title(fr"Masked Array ({sigma}$\sigma$ contours around source)")
        plt.show()

    return mask











cub_2 = Cube('/home/sai/Downloads/Pisco.cube.50kms.image.fits')
ra, dec = (205.533741, 9.477317341)

mask = create_contour_mask(image='/home/sai/Downloads/Pisco.cii.455kms.image.fits',
                    ra=ra,
                    dec=dec,
                    sigma=2,
                    plot=False)

flux,err = spectrum(cube=cub_2, ra=ra,dec=dec,radius=0.8,calc_error=True)

flux_mask,err_mask = spectrum(cube=cub_2, ra=ra,dec=dec,radius=0.8, contour_mask= mask,calc_error=True)

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




