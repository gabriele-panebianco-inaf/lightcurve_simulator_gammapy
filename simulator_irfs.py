
############ Not-gammapy imports

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, PowerNorm

import astropy.units as u

from astropy.table import QTable
from astropy.io import fits

# Gammapy imports
from gammapy.irf import (
    EffectiveAreaTable2D,
    Background2D, Background3D,
    EnergyDispersion2D, EDispKernel, EDispKernelMap,
)


FIGURE_FORMAT = ".pdf"






def Compute_Energy_Dispersion_Matrix(Detector_Response_Matrix,
                                    axis_energy_true,
                                    axis_energy_reco,
                                    aeff,
                                    livetimes,
                                    geom,
                                    logger,
                                    configuration,
                                    transient,
                                    output_directory):
    """
    Returns the Energy Dispersion Matrix as requested by Gammapy.
    Set the Exposure Map of the Energy Dispersion Matrix

    Parameters
    ----------
    Detector_Response_Matrix : `astropy.units.Quantity`
        Astropy matrix of Detector Response as a function of true and reconstructed energy.
    axis_energy_true : `gammapy.maps.MapAxis`
        True Energy Axis.
    axis_energy_reco : `gammapy.maps.MapAxis`
        Reco Energy Axis.
    aeff : `gammapy.irf.EffectiveAreaTable2D`
        Effective Area, needed to set the Exposure Map.
    livetimes : `astropy.units.Quantity`
        Astropy Array of the livetimes, needed to set the Exposure Map.
    logger : `logging.Logger`
        Logger from main.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    output_directory : str
        Output directory where to save a figure of the Effective Area.

    Returns
    -------
    edisp : `gammapy.irf.EDispKernelMap `
        Energy Dispersion Matrix as a DL4 reduced IRF with an Exposure Map.
    """

    logger.info("Compute Energy Dispersion Matrix")

    edisp = EDispKernel(axes = [axis_energy_true, axis_energy_reco], data = Detector_Response_Matrix.value)
    edisp = EDispKernelMap.from_edisp_kernel(edisp, geom = geom)

    # Normalize GBM Data:
    if configuration['Name_Instrument'] == 'GBM':
        DRM = np.zeros(np.shape(edisp.edisp_map.data.T[0][0].T))
        for i, r in enumerate(edisp.edisp_map.data.T[0][0].T):
            norm_row = np.sum(r)
            if norm_row != 0.0:
                DRM[i] = r / norm_row

        DRM = np.reshape(DRM, np.shape(edisp.edisp_map.data))
        edisp.edisp_map.data = DRM
        logger.warning(f"Normalization to 1 applied: assuming no lost photons.")

    # Set correct Units for exposure map
    edisp.exposure_map = edisp.exposure_map.to_unit(aeff.unit * livetimes.unit)
    # Initialize the Exposure with effective area values * livetimes. Assume all lifetimes are equal
    edisp.exposure_map.data *= 0.0
    edisp.exposure_map.data += np.reshape( aeff.data.T[0], edisp.exposure_map.data.T.shape ).T
    edisp.exposure_map.data *= livetimes[0].value

    logger.info("Exposure Map of Energy Dispersion set.")

    # Plot the Energy Dispersion Matrix


    # Prepare Grid
    X, Y = np.meshgrid(axis_energy_true.center.value, axis_energy_reco.center.value)

    # Copy Data with Masking
    Z = np.ma.masked_where(edisp.edisp_map.data.T[0][0] <= 0, edisp.edisp_map.data.T[0][0])

    # Plot
    fig, ax = plt.subplots(1, figsize=(9,5))

    # Define Levels
    levs = np.linspace(np.floor(np.power(Z.min(),0.3)),
                       np.ceil( np.power(Z.max(),0.3)),
                       num = 50
                      )
    levs = np.power(levs, 1.0/0.3)

    # Plot Data
    cs = ax.contourf(X, Y, Z, levs, norm = PowerNorm(gamma=0.3), cmap = 'plasma')
    ax.contour(X, Y, Z, levs, norm = PowerNorm(gamma=0.3), colors='white', alpha=0.05)
    cbar = fig.colorbar(cs)

    # Labels
    ax.set_facecolor('k')
    ax.set_xscale('log')
    ax.set_yscale('log')
    cbar.set_label('Redistribution Probability', fontsize = 'large')
    ax.set_xlabel('True Energy ['+axis_energy_true.unit.to_string()+']', fontsize = 'large')
    ax.set_ylabel('Energy ['     +axis_energy_reco.unit.to_string()+']', fontsize = 'large')

    title = f"Energy Dispersion Matrix {configuration['Name_Instrument']}"
    title+= f" {configuration['Name_Detector']}, {transient['name']}."
    ax.set_title(title, fontsize = 'large')

    # Save figure
    figure_name = output_directory+"IRF_energy_dispersion_matrix"+FIGURE_FORMAT
    logger.info(f"Saving Energy Dispersion Matrix plot : {figure_name}\n")
    fig.savefig(figure_name, facecolor = 'white')


    return edisp


def Read_Background_Spectrum(configuration,
                            geometry,
                            logger,
                            hdu_ebounds="EBOUNDS",
                            hdu_spectrum="SPECTRUM"
                            ):
    """
    Return the background Spectral Model.

    Parameters
    ----------
    configuration : dict
        Dictionary with the parameters from the YAML file.
    geometry : `gammapy.maps.region.geom.RegionGeom`
        Geometry to compute solid angle Area.
    logger : `logging.Logger`
        Logger from main.
    hdu_ebounds : str
        Name of the HDU that contains the Reconstructed Energy Axis.
    hdu_spectrum : str
        Name of the HDU that contains the Spectral Model.

    Returns
    -------
    `astropy.units.Quantity`
    """

    logger.info(f"Read the Background Spectral Model from BAK file: {configuration['Input_bak']}")

    # Define Background Tables
    with fits.open(configuration['Input_bak']) as hdulist:
        table_spectrum = QTable.read(hdulist[hdu_spectrum])
        try:
            table_ebounds = QTable.read(hdulist[hdu_ebounds])
        except:
            logger.warning(f"Energy Axis not found in BAK. Try to look in {configuration['Input_rmf']}")
            with fits.open(configuration['Input_rmf']) as hdulist_rmf:
                table_ebounds = QTable.read(hdulist_rmf[hdu_ebounds])
        
        if 'RATE' in table_spectrum.colnames:
            if table_spectrum['RATE'].unit is None:
                logger.warning(f"RATE unit not found. Using 1/{configuration['Time_Unit']}")
                table_spectrum['RATE'] = table_spectrum['RATE'] / u.Unit(configuration['Time_Unit'])
        elif 'COUNTS' in table_spectrum.colnames:
            logger.warning(f"Column RATE not found. Using COUNTS / (EXPOSURE [s])")
            Integration_time = hdulist[hdu_spectrum].header['EXPOSURE']*u.s
            logger.info(f"EXPOSURE (s) = {Integration_time.value}")
        else:
            logger.error(f"We could not find column RATE nor COUNTS")
            exit()

        # Check Units
        if table_ebounds['E_MIN'].unit is None:
            logger.warning(f"Energy unit not found. Using {configuration['Energy_Unit']}")
            table_ebounds['E_MIN'] = table_ebounds['E_MIN'] * u.Unit(configuration['Energy_Unit'])
            table_ebounds['E_MAX'] = table_ebounds['E_MAX'] * u.Unit(configuration['Energy_Unit'])
    
    # Define a new Table with the Columns we need.
    table = QTable()
       
    table['E_MIN'] = table_ebounds['E_MIN']
    table['E_MAX'] = table_ebounds['E_MAX']

    if 'RATE' in table_spectrum.colnames:
        table['RATE'] = table_spectrum['RATE']
    elif 'COUNTS' in table_spectrum.colnames:
        table['RATE'] = table_spectrum['COUNTS'].value / Integration_time

    # Define the column of the Background Spectral Model    
    table['BKG_MOD'] = table['RATE'] / (table['E_MAX']-table['E_MIN'])
    table['BKG_MOD'] = table['BKG_MOD']/ geometry.solid_angle()


    energy_edges = np.append(table['E_MIN'], table['E_MAX'][-1])
    energy_range = configuration['Energy_Range_Reco'] * u.Unit(configuration['Energy_Unit'])
    i_start, i_stop = 0, -1
    
    if configuration['Energy_Slice']:
        range_message = f"Original Reco Energy Range:"
        range_message+= f" [{np.round(energy_edges[0].value,3)},"
        range_message+= f" {np.round(energy_edges[-1].value,3)}]"
        range_message+= f" {energy_edges.unit}. Energy bins: {len(table)}."
        logger.info(range_message)

        dummy_array = energy_edges - energy_range[0]
        i_start = np.argmin(np.abs(dummy_array.value))

        dummy_array = energy_edges - energy_range[1]
        i_stop = np.argmin(np.abs(dummy_array.value))
        
        table = table[i_start: i_stop]

        logger.info(f"Slice Energies between indexes [{i_start},{i_stop}]")

    range_message = f"Background Spectral Model defined at Reco Energy Range:"
    range_message+= f" [{np.round(energy_edges[i_start].value,3)},"
    range_message+= f" {np.round(energy_edges[i_stop].value,3)}]"
    range_message+= f" {energy_edges.unit}. Energy bins: {len(table)}."
    logger.info(range_message)
    
    logger.info(f"Background Spectral Model defined with unit {table['BKG_MOD'].unit}\n")
    
    return table['BKG_MOD']

def Compute_Background_3D(bak_model,
                          axis_energy_reco_bkg,
                          axis_fovlon,
                          axis_fovlat,
                          geometry,
                          logger,
                          configuration,
                          transient,
                          output_directory
                         ):
    """
    Returns the Background IRF as requested by Gammapy.
    We assume that it is constant with offset.

    Parameters
    ----------
    bak_model : `astropy.units.Quantity`
        Astropy array of Background Spectrum as a function of reco energy.
    axis_energy_reco_bkg : `gammapy.maps.MapAxis`
        Reco Energy Axis.
    axis_fovlon : `gammapy.maps.MapAxis`
        Axis FoV Lon.
    axis_fovlat : `gammapy.maps.MapAxis`
        Axis FoV Lat.
    geometry : `gammapy.maps.region.geom.RegionGeom`
        Geometry to compute solid angle Area.
    logger : `logging.Logger`
        Logger from main.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    output_directory : str
        Output directory where to save a figure of the Effective Area.

    Returns
    -------
    bkg : `gammapy.irf.Background3D`
    
    """

    logger.info("Assume background is constant in the Instrument FoV")

    # Replicate the Background array for each bin of the Fov Lon and Lat Axes
    data_bkg = np.transpose(bak_model.value * np.ones((axis_fovlat.nbin,axis_fovlon.nbin,axis_energy_reco_bkg.nbin)))

    bkg = Background3D(axes = [axis_energy_reco_bkg, axis_fovlon, axis_fovlat],
                        data = data_bkg,
                        unit = bak_model.unit,
                      )

    logger.info(bkg)

    # Plot and save
    fig, ax = plt.subplots(1, figsize=(7,5))

    bak_to_plot = bak_model*geometry.solid_angle()
    ax.step(axis_energy_reco_bkg.center.value, bak_to_plot, color = 'C3')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(f"Energy [{axis_energy_reco_bkg.unit.to_string()}]", fontsize = 'large')
    ax.set_ylabel(f"Background rate [{bak_to_plot.unit.to_string()}]", fontsize = 'large')
    title = f"Background Spectral Model {configuration['Name_Instrument']}"
    title+= f" {configuration['Name_Detector']}, {transient['name']}."
    ax.set_title(title, fontsize = 'large')
    ax.grid()

    # Save figure
    figure_name = output_directory+"IRF_background_spectrum"+FIGURE_FORMAT
    logger.info(f"Saving Background Spectral Model plot : {figure_name}\n")
    fig.savefig(figure_name, facecolor = 'white')

    return bkg

