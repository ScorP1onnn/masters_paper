import interferopy.tools as iftools
import numpy as np
from interferopy.cube import Cube
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Ellipse
import numpy as np
from interferopy.cube import Cube
from matplotlib import colors
from matplotlib.path import Path
from scipy import integrate, constants
import astropy.units as u
import utils



def closest_contour(contour_set,px,py,search_radius,pixsize):

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

    # Convert search_radius arcsecond to pixel scale
    arcsec_to_pixel = search_radius / pixsize  # Assume self.pixel_scale exists in arcsec/pixel
    if min_distance > arcsec_to_pixel:
        print(f"Warning: No contour detected within {search_radius} arcsecond from the given RA and Dec. Returning None")
        return None

    return nearest_path, contour_paths



def create_contour_mask(image,ra: float = None, dec: float = None, sigma: float = 1.0,
                        search_radius: float = 2., px: int = None, py: int = None, plot=True):

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

    """
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

    # Convert search_radius arcsecond to pixel scale
    arcsec_to_pixel = search_radius / self.pixsize  # Assume self.pixel_scale exists in arcsec/pixel
    if min_distance > arcsec_to_pixel:
        print(f"Warning: No contour detected within {search_radius} arcsecond from the given RA and Dec. Returning None")
        return None
    """""

    nearest_path, contour_paths = closest_contour(contour_set,px,py,search_radius,self.pixsize)

    mask = np.zeros_like(subim, dtype=bool)

    x = np.linspace(extent[0], extent[1], subim.shape[0])
    y = np.linspace(extent[2], extent[3], subim.shape[0])
    X, Y = np.meshgrid(x, y)

    if nearest_path is not None:
        # Transform image indices to the plot coordinates
        pixel_coords = np.column_stack((X.ravel(), Y.ravel()))
        inside = nearest_path.contains_points(pixel_coords).reshape(subim.shape)
        mask[inside] = True

        # Check if any other contour lies within the nearest contour and update the mask
        nearest_path_polygon = Path(nearest_path.vertices)
        for path in contour_paths:
            if path != nearest_path:
                # Check if all vertices of this path lie inside the nearest contour
                vertices = path.vertices
                if np.all(nearest_path_polygon.contains_points(vertices)):
                    path_inside = path.contains_points(pixel_coords).reshape(subim.shape)
                    mask[path_inside] = False

    if plot == True:

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
        n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
        ax1.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)
        ax1.contour(subim.T, extent=extent, colors="red", levels=np.array([sigma]) * rms, zorder=1, linewidths=0.5,linestyles="-")
        # ax1.plot(px, py, 'bo', label='Target Pixel')
        ax1.set_title(fr"Original Image with {sigma}$\sigma$ Contours")

        ax2.imshow(mask, extent=extent, origin='lower', cmap='gray')
        ax2.contour(mask,extent=extent, origin='lower', colors='blue')
        ax2.set_title(fr"Masked Array ({sigma}$\sigma$ contour around source)")
        plt.show()

    return mask







exit()

cont_img = Cube('saimurali/vcb2-id141-c1mm-selfcal.fits')

"""
To account for 15% calibration error in the flux error, the formula used is

corrected/accounted_error = SQRT( (error ** 2) + ((0.15 * flux_value)**2) )

"""

c = 0.15
rms = cont_img.rms
sum_sqr =  (rms**2) + ((c*57e-3) ** 2)
print('rms = ', rms * 1e3)

print('Calibration accounted error = ', np.round(np.sqrt(sum_sqr) * 1e3,1))
print(8.6/(cont_img.beamvol))


print("Next")

cont_img = Cube('saimurali/vdb2-hdf850-c1mm.fits')
rms = cont_img.rms
sum_sqr = (rms**2) + ((c*2e-3) ** 2)
corr_err =  np.round(np.sqrt(sum_sqr) * 1e3,1)

print(corr_err/(cont_img.beamvol))

print("*******************")

nii = Cube('/home/sai/nii-m0/veb2/gn20-nii-m0.fits')
print(nii.rms * 1e3)
print(0.644/2)


