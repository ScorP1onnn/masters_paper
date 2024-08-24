
def map_all(nii_smg,cont_smg,coord_smg,name_smg):

    cutout=9
    aper_rad=1.3
    scale=1e3
    fig= plt.figure(figsize=(8, 8))
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 2), axes_pad=0.5, share_all=False, cbar_location="right", cbar_mode="each", cbar_size="3%", cbar_pad=0)


    nii_gn20=Cube(nii_smg[0])
    cont_gn20=Cube(cont_smg[0])
    ra,dec = coord_smg[0]


    #nii_gn20


    cub=nii_gn20
    ax=grid[0]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,linewidths=0.5, linestyles="-")
    ax.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax.add_patch(ellipse)

    # set limits to exact cutout size
    ax.set_xlim(cutout, -cutout)
    ax.set_ylim(-cutout, cutout)

    ticks = [-int(vmax), -2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres, int(vmax)]
    cb = ax.cax.colorbar(axim, ticks=ticks, format='%0.1f')
    #cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)")
    ax.set_xlabel(r"$\Delta$ RA (arcsec)")
    ax.set_title(name_smg[0],x=1.18)


    #cont_gn20
    cub = cont_gn20
    ax = grid[1]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,
               linewidths=0.5, linestyles="-")
    ax.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,
               linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],
                      angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax.add_patch(ellipse)

    # set limits to exact cutout size
    ax.set_xlim(cutout, -cutout)
    ax.set_ylim(-cutout, cutout)



    ticks = [ -2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres]
    #ticks = [-3*rms,0,3*rms,6*rms,9*rms]
    cb = ax.cax.colorbar(axim, ticks=ticks, format='%0.1f')
    cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)")
    ax.set_xlabel(r"$\Delta$ RA (arcsec)")
    #ax.tick_params(left=False)



    ##########################################################################################################

    nii_gn20 = Cube(nii_smg[1])
    cont_gn20 = Cube(cont_smg[1])
    ra, dec = coord_smg[1]

    # nii_ID141

    cub = nii_gn20
    ax = grid[2]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,
               linewidths=0.5, linestyles="-")
    ax.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,
               linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],
                      angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax.add_patch(ellipse)

    # set limits to exact cutout size
    ax.set_xlim(cutout, -cutout)
    ax.set_ylim(-cutout, cutout)

    ticks = [-int(vmax), -2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres, int(vmax)]
    cb = ax.cax.colorbar(axim, ticks=ticks, format='%0.1f')
    # cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)")
    ax.set_xlabel(r"$\Delta$ RA (arcsec)")

    # cont_ID141
    cub = cont_gn20
    ax = grid[3]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,
               linewidths=0.5, linestyles="-")
    ax.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,
               linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],
                      angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax.add_patch(ellipse)

    # set limits to exact cutout size
    ax.set_xlim(cutout, -cutout)
    ax.set_ylim(-cutout, cutout)



    ticks = [-2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres]
    # ticks = [-3*rms,0,3*rms,6*rms,9*rms]
    cb = ax.cax.colorbar(axim, ticks=ticks, format='%0.1f')
    cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)")
    ax.set_xlabel(r"$\Delta$ RA (arcsec)")
    # ax.tick_params(left=False)

    ##################################################################################################################

    nii_gn20 = Cube(nii_smg[2])
    cont_gn20 = Cube(cont_smg[2])
    ra, dec = coord_smg[2]

    # nii_hdf850

    cub = nii_gn20
    ax = grid[4]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,
               linewidths=0.5, linestyles="-")
    ax.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,
               linewidths=0.5, linestyles="--")


    print(cub.beam)
    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],
                      angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)

    ax.add_patch(ellipse)


    # set limits to exact cutout size
    ax.set_xlim(cutout, -cutout)
    ax.set_ylim(-cutout, cutout)

    ticks = [-int(vmax), -2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres, int(vmax)]
    cb = ax.cax.colorbar(axim, ticks=ticks, format='%0.1f')
    # cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)")
    ax.set_xlabel(r"$\Delta$ RA (arcsec)")

    # cont_hd850
    cub = cont_gn20
    ax = grid[5]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,
               linewidths=0.5, linestyles="-")
    ax.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,
               linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],
                      angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax.add_patch(ellipse)

    # set limits to exact cutout size
    ax.set_xlim(cutout, -cutout)
    ax.set_ylim(-cutout, cutout)

    ticks = [-int(vmax),-2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres,int(vmax)]
    # ticks = [-3*rms,0,3*rms,6*rms,9*rms]
    cb = ax.cax.colorbar(axim, ticks=ticks, format='%0.1f')
    cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    ax.set_ylabel(r"$\Delta$ Dec (arcsec)")
    ax.set_xlabel(r"$\Delta$ RA (arcsec)")
    # ax.tick_params(left=False)

    plt.show()













def map1(nii_smg,cont_smg,coord_smg,name_smg):
    cutout = 9
    aper_rad = 1.3
    scale = 1e3

    fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(10, 6))
    fig.subplots_adjust(hspace=-0.28, wspace=0.4)
    nii_gn20 = Cube(nii_smg[0])
    cont_gn20 = Cube(cont_smg[0])


    cub = Cube(nii_smg[0])
    ra, dec = coord_smg[0]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax1.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax1.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,linewidths=0.5, linestyles="-")
    ax1.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax1.add_patch(ellipse)

    ellipse =  Ellipse(xy=(0, 0), width=4 * aper_rad, height=4* aper_rad, angle=0, edgecolor='darkgreen', fc="none", lw=2,ls=":")
    ax1.add_patch(ellipse)

    # set limits to exact cutout size
    ax1.set_xlim(cutout, -cutout)
    ax1.set_ylim(-cutout, cutout)

    ticks = [-int(vmax), -2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres, int(vmax)]
    #ax1.colorbar(axim, ticks=ticks, format='%0.1f')
    #cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    ax1.set_ylabel(r"$\Delta$ Dec (arcsec)")
    #ax1.set_xlabel(r"$\Delta$ RA (arcsec)")
    ticks = [-int(vmax), -2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres, int(vmax)]
    fig.colorbar(axim,ax=ax1,ticks=ticks, format='%0.1f', fraction=0.0475, pad=0.0)
    ax1.tick_params(bottom=False,labelbottom = False)
    ax1.set_title(name_smg[0],size=15)
    ax1.text(4,7,"[NII] 205",size=15)



    cub = Cube(nii_smg[1])
    ra, dec = coord_smg[1]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax2.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax2.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,linewidths=0.5, linestyles="-")
    ax2.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax2.add_patch(ellipse)

    ellipse = Ellipse(xy=(0, 0), width=4 * aper_rad, height=4 * aper_rad, angle=0, edgecolor='darkgreen', fc="none",
                      lw=2, ls=":")
    ax2.add_patch(ellipse)

    # set limits to exact cutout size
    ax2.set_xlim(cutout, -cutout)
    ax2.set_ylim(-cutout, cutout)


    #ax1.colorbar(axim, ticks=ticks, format='%0.1f')
    #cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    #ax2.set_ylabel(r"$\Delta$ Dec (arcsec)")
    #ax2.set_xlabel(r"$\Delta$ RA (arcsec)")
    ticks = [-int(vmax),-2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres,int(vmax)]

    fig.colorbar(axim,ax=ax2,ticks=ticks, format='%0.1f', fraction=0.0475, pad=0.0)
    ax2.set_title(name_smg[1],size=15)
    ax2.text(4,7,"[NII] 205",size=15)




    cub = Cube(nii_smg[2])
    ra, dec = coord_smg[2]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax3.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax3.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,
                linewidths=0.5, linestyles="-")
    ax3.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,
                linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],
                      angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax3.add_patch(ellipse)

    ellipse = Ellipse(xy=(0, 0), width=4 * aper_rad, height=4 * aper_rad, angle=0, edgecolor='darkgreen', fc="none",
                      lw=2, ls=":")
    ax3.add_patch(ellipse)

    # set limits to exact cutout size
    ax3.set_xlim(cutout, -cutout)
    ax3.set_ylim(-cutout, cutout)

    # ax1.colorbar(axim, ticks=ticks, format='%0.1f')
    # cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    #ax3.set_ylabel(r"$\Delta$ Dec (arcsec)")
    #ax3.set_xlabel(r"$\Delta$ RA (arcsec)")
    ticks = [-int(vmax), -2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres, int(vmax)] #For SMG
    #ticks = [-2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres] #For QSO

    fig.colorbar(axim, ax=ax3, ticks=ticks, format='%0.1f', fraction=0.0475, pad=0.0,label=r"$S_\nu$ (mJy beam$^{-1}$)")
    ax3.set_title(name_smg[2],size=15)
    ax3.text(4,7,"[NII] 205",size=15)
    # ax1.set_title(name_smg[0],x=1.18)



    #Continuum

    cub = Cube(cont_smg[0])
    ra, dec = coord_smg[0]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax4.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax4.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,
                linewidths=0.5, linestyles="-")
    ax4.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,
                linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],
                      angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax4.add_patch(ellipse)

    ellipse = Ellipse(xy=(0, 0), width=4.2 * aper_rad, height=4.2 * aper_rad, angle=0, edgecolor='darkgreen', fc="none", lw=2,
                      ls=":")
    ax4.add_patch(ellipse)

    # set limits to exact cutout size
    ax4.set_xlim(cutout, -cutout)
    ax4.set_ylim(-cutout, cutout)

    # ax1.colorbar(axim, ticks=ticks, format='%0.1f')
    # cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    ax4.set_ylabel(r"$\Delta$ Dec (arcsec)")
    ax4.set_xlabel(r"$\Delta$ RA (arcsec)")
    ticks = [-2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres]

    fig.colorbar(axim, ax=ax4, ticks=ticks, format='%0.1f', fraction=0.048, pad=0.0)
    ax4.text(5, 7, "Continuum", size=15)
    # ax1.set_title(name_smg[0],x=1.18)




    cub = Cube(cont_smg[1])
    ra, dec = coord_smg[1]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax5.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax5.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,
                linewidths=0.5, linestyles="-")
    ax5.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,
                linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],
                      angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax5.add_patch(ellipse)

    ellipse = Ellipse(xy=(0, 0), width=4.5 * aper_rad, height=4.5 * aper_rad, angle=0, edgecolor='darkgreen', fc="none",
                      lw=2, ls=":")
    ax5.add_patch(ellipse)

    # set limits to exact cutout size
    ax5.set_xlim(cutout, -cutout)
    ax5.set_ylim(-cutout, cutout)

    # ax1.colorbar(axim, ticks=ticks, format='%0.1f')
    # cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    #ax5.set_ylabel(r"$\Delta$ Dec (arcsec)")
    ax5.set_xlabel(r"$\Delta$ RA (arcsec)")
    ticks = [-2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres] #for SMG
    #ticks = [-0.6, -2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres, 0.6]
    print(vmax)

    fig.colorbar(axim, ax=ax5, ticks=ticks, format='%0.1f', fraction=0.048, pad=0.0)
    ax5.text(4.8, 7, "Continuum", size=15)
    # ax1.set_title(name_smg[0],x=1.18)





    cub = Cube(cont_smg[2])
    ra, dec = coord_smg[2]

    px, py = cub.radec2pix(ra, dec)
    r = int(np.round(cutout * 1.05 / cub.pixsize))

    edgera, edgedec = cub.pix2radec([px - r, px + r], [py - r, py + r])  # coordinates of the two opposite corners
    extent = [(edgera - ra) * 3600, (edgedec - dec) * 3600]
    extent = extent[0].tolist() + extent[1].tolist()  # concat two lists

    subim = cub.im[px - r:px + r + 1, py - r:py + r + 1, 0] * scale

    vmax = np.nanmax(subim)
    linthres = 5 * np.std(subim)
    n = colors.SymLogNorm(linthresh=linthres, linscale=0.5, vmin=-vmax, vmax=vmax)
    axim = ax6.imshow(subim.T, origin='lower', cmap="PuOr_r", zorder=-1, norm=n, extent=extent)

    rms = cub.rms[0] * scale
    ax6.contour(subim.T, extent=extent, colors="black", levels=np.array([2, 4, 6, 8, 16, 32]) * rms, zorder=1,
                linewidths=0.5, linestyles="-")
    ax6.contour(subim.T, extent=extent, colors="gray", levels=np.array([-32, -16, -8, -4, -2]) * rms, zorder=1,
                linewidths=0.5, linestyles="--")

    ellipse = Ellipse(xy=(cutout * 0.8, -cutout * 0.8), width=cub.beam["bmin"], height=cub.beam["bmaj"],
                      angle=-cub.beam["bpa"], edgecolor='black', fc='w', lw=0.75)
    ax6.add_patch(ellipse)

    # set limits to exact cutout size
    ax6.set_xlim(cutout, -cutout)
    ax6.set_ylim(-cutout, cutout)

    ellipse = Ellipse(xy=(0, 0), width=4 * aper_rad, height=4 * aper_rad, angle=0, edgecolor='darkgreen', fc="none",
                      lw=2, ls=":")
    ax6.add_patch(ellipse)

    # ax1.colorbar(axim, ticks=ticks, format='%0.1f')
    # cb.set_label(r"$S_\nu$ (mJy beam$^{-1}$)")

    # ax.tick_params(direction='in', which="both")
    # ax3.set_ylabel(r"$\Delta$ Dec (arcsec)")
    ax6.set_xlabel(r"$\Delta$ RA (arcsec)")
    ticks = [-int(vmax),-2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres,int(vmax)] #for SMG
    #ticks = [-2 * linthres, -linthres / 2, 0, linthres / 2, 2 * linthres]

    fig.colorbar(axim, ax=ax6, ticks=ticks, format='%0.1f', fraction=0.048, pad=0.0,label=r"$S_\nu$ (mJy beam$^{-1}$)")
    ax6.text(4.8, 7, "Continuum", size=15)
    # ax1.set_title(name_smg[0],x=1.18)

    plt.show()












nii_smg = ["/home/sai/nii-m0/veb2/gn20-nii-m0.fits","/home/sai/nii-m0/vcb2/id141-nii-m0.fits","/home/sai/nii-m0/vdb2/hdf850-nii-m0.fits"]
cont_smg=["/home/sai/saimurali/veb2-gn20-c1mm-selfcal.fits","/home/sai/saimurali/vcb2-id141-c1mm-selfcal.fits","/home/sai/saimurali/vdb2-hdf850-c1mm.fits"]
coord_smg = [(189.2995833,62.3700278),(216.0580417,2.3846667),(189.2167500,62.2072222)]
name_smg=["GN20","ID141","HDF850.1"]


nii_qso = ["/home/sai/nii-m0/vbb2/2322-nii-m0.fits","/home/sai/nii-m0/w16ef002/2054-nii-m0.fits","/home/sai/nii-m0/w16ef001/2310-nii-m0.fits"]
cont_qso = ["/home/sai/saimurali/vbb2-wbb7-2322-c1mm-selfcal.fits","/home/sai/saimurali/w16ef002-2054-c1mm.fits","/home/sai/saimurali/w16ef001-2310-c1mm.fits"]
coord_qso = [(350.5299167,19.7395556),(313.5271250,-0.0873889),(347.6620881,18.9221944)]
name_qso = ["PSSJ2322+1944","J2054+0005","J2310+1855"]


#map_all(nii_smg,cont_smg,coord_smg,name_smg)
map1(nii_smg,cont_smg,coord_smg,name_smg)
#map1(nii_qso,cont_qso,coord_qso,name_qso)

exit()

