import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
from astropy.table import QTable
from astropy.io import fits
from gammapy.irf import Background3D

FIGURE_FORMAT = ".pdf"





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

