###########################################
#                                         #
#                                         #
#                                         #
###########################################

############ Not-gammapy imports

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm, PowerNorm

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table, QTable, hstack
from astropy.io import fits

import logging
import os
import yaml
from yaml import SafeLoader
from time import time

# Gammapy imports
from gammapy.data import Observation
from gammapy.datasets import SpectrumDataset, Datasets
from gammapy.irf import (
    EffectiveAreaTable2D,
    Background2D, Background3D,
    EnergyDispersion2D, EDispKernel, EDispKernelMap,
)
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom, TimeMapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    SkyModel,
    LightCurveTemplateTemporalModel
)
from gammapy.utils.random import get_random_state

############ Functions

def Welcome(logger):
    """
    Welcome function. This is used to print info about the code on execution.
    """
    logger.info('Welcome! A proper Welcome must be implemented.\n')
    return None

def Initialize(logger, Configuration_file_path):
    """
    Read the YAML Configuration file into a dictionary.
    Load the Transient Catalogue and choose one transient.
    Create the output directory and a log file.
    
    Parameters
    ----------
    logger : `logging.Logger`
        Logger from main.
    Configuration_file_path : str
        Name of the input YAML file.
    Returns
    -------
    configuration : dict
        Dictionary with the parameters from the YAML file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    """

    # Load YAML as a dict
    logger.info("Loading YAML file...")
    with open(Configuration_file_path) as f:
        configuration = yaml.load(f, Loader = SafeLoader)

    # Load Catalogue of transients
    logger.info("Loading Sources Catalogue...")
    with fits.open(configuration['Input_Catalogue']) as hdulist:
        table_catalog = QTable.read(hdulist['CATALOG'])
        table_catalog.add_index('name')
    
    # Choose a burst
    if configuration['Name_Transient'] is not None:
        try:
            transient = table_catalog.iloc[table_catalog['name'] == configuration['Name_Transient']][0]
            # Note: we are assuming all names are different.
            # The transient is an astropy.table.row.Row
            logger.info('Requested transient available.\n')
        except IndexError:
            logger.error(f"Transient {configuration['Name_Transient']} was not found.")
            raise
    else:
        rng = np.random.default_rng(configuration['Random_Seed'])
        random_index = rng.integers(0, high=len(table_catalog))
        transient = table_catalog[random_index]
        logger.info('Transient randomly sampled.\n')
    
    # Create output directory
    observation_livetime = configuration['Observation_Livetime'] * u.Unit(configuration['Time_Unit'])
    energy_unit = u.Unit(configuration['Energy_Unit'])
    energy_range_reco = configuration['Energy_Range_Reco'] * energy_unit

    output_directory = configuration['Output_Directory']
    output_directory+= transient['name'] + '/'
    output_directory+= configuration['Name_Instrument' ] + '_'
    output_directory+= configuration['Name_Detector'   ] + '_'
    output_directory+= str(int(observation_livetime.to('ms').value)) + 'ms_'
    output_directory+= str(int(energy_range_reco.value[0])) + '_'
    output_directory+= str(int(energy_range_reco.value[1])) + '_'
    output_directory+= energy_unit.to_string() + '/'
    if configuration['Output_Run_ID'] is not None:
        output_directory += configuration['Output_Run_ID']+'/'
    
    os.makedirs(os.path.dirname(output_directory), exist_ok=True)

    # Define Logger for file
    f_handler = logging.FileHandler(output_directory+'file.log', mode='w')
    f_handler.setLevel(logging.INFO)
    f_handler.setFormatter(logging.Formatter('%(asctime)s. %(levelname)s: %(message)s'))# -%(name)s
    logger.addHandler(f_handler)

    logger.info(f"Configuration. YAML File : {Configuration_file_path}")
    logger.info(f"Configuration. Source Catalogue: {configuration['Input_Catalogue']}")

    if configuration['Name_Transient'] is not None:
        logger.info(f"Configuration. Requested Transient Name: {transient['name']}")
    else:
        logger.info(f"Configuration. Random Transient Name: {transient['name']}")

    logger.info(f"Configuration. Output Directory: {output_directory} \n")
    
    return configuration, transient

def Define_Reference_Time(transient, time_format, logger):
    """
    Return the Burst Trigger Time to be used as Reference Time for the simulation.

    Parameters
    ----------
    transient: `astropy.table.row.Row`
        Row of the chosen transient.
    time_format: str
        Time format of the TimeMapAxis.
    logger : `logging.Logger`
        Logger from main.

    Returns
    -------
    trigger_time: `astropy.time.Time`
        Trigger Time
    """
    trigger_time = Time(transient['trigger_time'], format='mjd', scale='utc')
    trigger_time.format = time_format

    logger.info(f"Define Trigger Time (format={time_format}): {trigger_time}.\n")
    return trigger_time



def Define_FoV_Axes(logger):
    """
    Return the FoV Axes: Offset, Lon, Lat.

    Parameters
    ----------
    logger : `logging.Logger`
        Logger from main.

    Returns
    -------
    axis_offset, axis_fovlon, axis_fovlat: `gammapy.maps.axes.MapAxis`
        FoV Axes
    """
    logger.info(f"Define Instrument FoV Axes.")

    # Assumption: The FoV Center is (0,0), FoV_Lon and FoV_Lat are symmetric wrt to 0.0
    fov_maximum_offset = 5.0 * u.deg
    fov_n_bin = 5
    
    # Create Instrument Offset axis
    axis_offset = MapAxis.from_bounds(0.0, fov_maximum_offset.value,
                                      unit = fov_maximum_offset.unit, nbin = fov_n_bin, name = "offset"
                                     )
    # Create Instrument FoV_lon axis
    axis_fovlon = MapAxis.from_bounds(-fov_maximum_offset.value/2.0, +fov_maximum_offset.value/2.0,
                                      unit = fov_maximum_offset.unit, nbin = fov_n_bin, name = "fov_lon"
                                     )
    # Create Instrument FoV_lat axis
    axis_fovlat = MapAxis.from_bounds(-fov_maximum_offset.value/2.0, +fov_maximum_offset.value/2.0,
                                      unit = fov_maximum_offset.unit, nbin = fov_n_bin, name = "fov_lat"
                                     )
    
    logger.info(axis_offset)
    logger.info(axis_fovlon)
    logger.info(axis_fovlat)

    return axis_offset, axis_fovlon, axis_fovlat


def Define_Schedule(configuration, trigger_time, logger):
    """
    Return the Schedule for the simulation:
    Number of observations, Start time and duration of each observation.

    Parameters
    ----------
    logger : `logging.Logger`
        Logger from main.
    trigger_time : `astropy.time.Time`
        Trigger time.
    configuration : dict
        Dictionary with the parameters from the YAML file.

    Returns
    -------
    observations_number : int
        Number of Observations.
    observations_start : `astropy.time.core.Time`
        Absolute start time of each observation.
    observations_livetimes : `astropy.units.quantity.Quantity`
        Array of duration of the observations.    
    """

    logger.info("Define the Observations' schedule.")


    Time_Unit = u.Unit(configuration['Time_Unit'])
    Time_Start= configuration['Observation_Time_Start'] * Time_Unit
    Time_Stop = configuration['Observation_Time_Stop' ] * Time_Unit
    Livetime  = configuration['Observation_Livetime'  ] * Time_Unit
    Deadtime  = configuration['Observation_Deadtime'  ] * Time_Unit

    # Estimate Number of observations
    observations_number = (Time_Stop - Time_Start) / Livetime
    observations_number = int(np.floor(observations_number.value))

    # Define starting time of each observation linearly spaced during the night (wrt trigger time)
    observations_start = np.linspace(Time_Start.value,
                                     Time_Stop.value,
                                     num = observations_number
                                    )
    observations_start = observations_start.tolist() * Time_Unit

    # Define the duration of each observation as the difference between
    # two following starting times minus the rest time between them.
    # These are still quantities relative to the trigger time
    observations_livetimes = observations_start[1:] - observations_start[:-1] - Deadtime

    # Remove last edge to have the same array dimesion for starting times and livetimes.
    observations_start = observations_start[:-1]

    # Turn them from astropy.units.quantity.Quantity to astropy.time.core.Time:
    # From relative to absolute times.
    observations_start = Time(trigger_time + observations_start)

    # Adjust the number of observations after resizing observations_start
    observations_number = observations_start.size


    # Log
    logger.info(f"Number of Observations: {observations_number}.")
    logger.info(f"Observation start: {observations_start[0]}.")
    logger.info(f"Observation livetime: {np.round(observations_livetimes[0],4)}.\n")

    return observations_number, observations_start , observations_livetimes




def Define_Energy_Axis(name_file_fits,
                       name_hdu,
                       configuration,
                       logger,
                       energy_is_true = False
                      ):
    """
    Returns an Energy Axis.

    Parameters
    ----------
    name_file_fits: str
        Name of a FITS file that must be opened to get the Energy Axis information.
    name_hdu: str
        Name of the HDU that contains the Energy Axis information.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    logger : `logging.Logger`
        Logger from main.
    energy_is_true: bool
        True if we want to create a True Energy Axis, False for a Reconstructed Energy Axis.

    Returns
    -------
    energy_axis: `gammapy.maps.MapAxis`
        True or Reconstructed Energy Axis
    """

    # Load Energy Table
    with fits.open(name_file_fits) as hdulist:
        energy_table = QTable.read(hdulist[name_hdu])

    # True or Reconstructed Energy?        
    if energy_is_true:
        energy_col_names = ("ENERG_LO", "ENERG_HI")
        energy_axis_name = "energy_true"
        logger.info("Define True Energy Axis")
    else:
        energy_col_names = ("E_MIN", "E_MAX")
        energy_axis_name = "energy"
        logger.info("Define Reconstructed Energy Axis")
    
    logger.info(f"Read Axis from: {name_file_fits}")
    
    # Define Edges Columns
    energy_col_min = energy_table[energy_col_names[0]]
    energy_col_max = energy_table[energy_col_names[1]]

    # Define Energy Unit
    Energy_Unit = u.Unit(configuration['Energy_Unit'])
    
    if energy_col_min.unit is u.Unit(""):
        logger.warning(f"Energy Axis file has adimensional values. Correct into {Energy_Unit}")
        energy_col_min = energy_col_min * Energy_Unit
        energy_col_max = energy_col_max * Energy_Unit
    
    # To avoid that min edge is 0
    energy_col_min[0] += 1e-2 * (energy_col_max[0] - energy_col_min[0])

    # Define Edges
    energy_edges = np.append(energy_col_min.value, energy_col_max.value[-1]) * energy_col_min.unit

    # Define Axis
    energy_axis = MapAxis.from_edges(energy_edges,
                                      name = energy_axis_name,
                                      interp = configuration['Energy_Interpolation']
                                     )
    
    # Slice Energy
    if configuration['Energy_Slice']:
        range_message = f"Original Energy Range: [{np.round(energy_axis.bounds[0].value,3)}, {np.round(energy_axis.bounds[1].value,3)}] "
        range_message+= str(energy_axis.unit) + f". Energy bins: {energy_axis.nbin}."
        logger.info(range_message)

        if energy_is_true:
            energy_custom_range = configuration['Energy_Range_True'] * Energy_Unit
        else:
            energy_custom_range = configuration['Energy_Range_Reco'] * Energy_Unit
        
        dummy_array = energy_axis.edges - energy_custom_range[0]
        i_start_energy = np.argmin(np.abs(dummy_array.value))

        dummy_array = energy_axis.edges - energy_custom_range[1]
        i_stop_energy  = np.argmin(np.abs(dummy_array.value))

        energy_axis = energy_axis.slice(slice(i_start_energy, i_stop_energy))

        logger.info(f"Slice between indexes [{i_start_energy},{i_stop_energy}].")

    range_message = f"Final Energy range: [{np.round(energy_axis.bounds[0].value,3)}, {np.round(energy_axis.bounds[1].value,3)}] "
    range_message+= str(energy_axis.unit)+f". Energy bins: {energy_axis.nbin}."
    logger.info(range_message)
    logger.info(energy_axis)

    return energy_axis


def Define_Geometry(pointing, axis_energy_reco, logger,radius=1.0):
    """
    Define the Source Sky Geometry: a circular region in the sky where we place the source
    and we assume that we can receive photons with reconstructed energy in the given axis.

    Parameters
    ----------
    pointing: `astropy.coordinates.SkyCoord`
        Center of the Region.
    axis_energy_reco: `gammapy.maps.MapAxis`
        Reconstructed Energy Axis, non-spatial dimension of the source geometry.
    logger : `logging.Logger`
        Logger from main.
    radius: float
        radius of the Circular Geometry in degree. Must be smaller than the FoV.

    Returns
    -------
    geometry : `gammapy.maps.region.geom.RegionGeom`
        Circular Source Geometry.
    """

    logger.info("Define Source Geometry.")

    geometry_radius = radius * u.deg

    source_geometry_str = pointing.frame.name + ';circle('
    source_geometry_str+= pointing.to_string().split()[0] + ', '
    source_geometry_str+= pointing.to_string().split()[1] + ', '
    source_geometry_str+= str(geometry_radius.value) + ')'

    geometry = RegionGeom.create(source_geometry_str, axes = [axis_energy_reco])

    logger.info(geometry)

    return geometry

def Define_Effective_Area_from_ARF(name_file_fits,
                                   name_hdu,
                                   configuration,
                                   logger,
                                   name_column_aeff = 'SPECRESP'
                                  ):
    """
    Return the Effective Area as a function of True Energy with their unit.

    Parameters
    ----------
    name_file_fits: str
        Name of a FITS file that must be opened to get the Energy Axis information.
    name_hdu: str
        Name of the HDU that contains the Energy Axis information.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    logger : `logging.Logger`
        Logger from main.
    name_column_aeff : str
        Name of the Column with the Effective Area values in the ARF file.

    Returns
    -------
    `astropy.units.Quantity`
    """

    logger.info(f"Read Effective Area from ARF file: {name_file_fits}")

    # Load Energy Table
    with fits.open(name_file_fits) as hdulist:
        table = QTable.read(hdulist[name_hdu])
    
    # This assumes SPECRESP is the name of the column with the effective area.
    table['SPECRESP'] = table['SPECRESP'].to(u.cm**2)

    edges = np.append(table['ENERG_LO'].value, table['ENERG_HI'].value[-1]) * table['ENERG_LO'].unit
    energy_range = configuration['Energy_Range_True'] * u.Unit(configuration['Energy_Unit'])
    i_start, i_stop = 0, -1
    
    if configuration['Energy_Slice']:

        range_message = f"Original Energy Range: [{np.round(table['ENERG_LO'][0].value,3)}, {np.round(table['ENERG_HI'][-1].value,3)}] "
        range_message+= str(table['ENERG_LO'].unit) + f". Energy bins: {len(table['ENERG_LO'])}."
        logger.info(range_message)

        dummy_array = edges - energy_range[0]
        i_start = np.argmin(np.abs(dummy_array.value))

        dummy_array = edges - energy_range[1]
        i_stop  = np.argmin(np.abs(dummy_array.value))
        
        table = table[i_start: i_stop]

        logger.info(f"Slice between indexes [{i_start},{i_stop}].")

    range_message = f"Final Energy range: [{np.round(edges[i_start].value,3)}, {np.round(edges[i_stop].value,3)}] "
    range_message+= f"{edges.unit}. Energy bins: {len(table['SPECRESP'])}.\n"
    logger.info(range_message)

    return table['SPECRESP']



def Define_Response_Matrix_rfom_RSP(name_file_fits,
                                    name_hdu,
                                    name_hdu_ebounds,
                                    configuration,
                                    logger
                                   ):
    """
    Return the Detector Response Matrix as a function of true and reconstructed energy with their unit.

    Parameters
    ----------
    name_file_fits: str
        Name of a FITS file that must be opened to get the Response as a function of True Energy.
    name_hdu: str
        Name of the HDU that contains the Response information as a function of True Energy.
    name_hdu: str
        Name of the HDU that contains information on the Reconstructed Energy.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    logger : `logging.Logger`
        Logger from main.    

    Returns
    -------
    `astropy.units.Quantity`
    """

    logger.info(f"Read Detector Response Matrix from: {name_file_fits}")

    # Load the Response
    with fits.open(name_file_fits) as hdulist:
        DETCHANS = hdulist[name_hdu].header["DETCHANS"]
        DRM_unit = u.Unit(hdulist[name_hdu].header['TUNIT6'])
        DRM_specresp = QTable.read(hdulist[name_hdu])
        DRM_ebounds = QTable.read(hdulist['EBOUNDS'])

    # Define the Matrix
    DRM = np.zeros([len(DRM_specresp), DETCHANS], dtype = np.float64)
    
    for i, l in enumerate(DRM_specresp):
        if l["N_GRP"]:
            m_start = 0
            for k in range(l["N_GRP"]):
            
                if np.isscalar(l["N_CHAN"]):
                    f_chan = l["F_CHAN"]    -1 # Should this be here?
                    n_chan = l["N_CHAN"]
                else:
                    f_chan = l["F_CHAN"][k] -1 # Should this be here?
                    n_chan = l["N_CHAN"][k]

                DRM[i, f_chan : f_chan+n_chan] = l["MATRIX"][m_start : m_start+n_chan]
                m_start += n_chan

    energy_true_edges = np.append(DRM_specresp['ENERG_LO'], DRM_specresp['ENERG_HI'][-1])
    energy_true_range = configuration['Energy_Range_True'] * u.Unit(configuration['Energy_Unit'])
    energy_reco_edges = np.append(DRM_ebounds['E_MIN'], DRM_ebounds['E_MAX'][-1])
    energy_reco_range = configuration['Energy_Range_Reco'] * u.Unit(configuration['Energy_Unit'])

    i_start_energy_true, i_start_energy_reco = 0, 0
    i_stop_energy_true, i_stop_energy_reco = -1, -1
    
    
    if configuration['Energy_Slice']:
        range_message = f"Original True Energy Range:"
        range_message+= f" [{np.round(energy_true_edges[0].value,3)},"
        range_message+= f" {np.round(energy_true_edges[-1].value,3)}]"
        range_message+= f" {energy_true_edges.unit}. Energy bins: {len(DRM_specresp)}."
        logger.info(range_message)

        range_message = f"Original Reco Energy Range:"
        range_message+= f" [{np.round(energy_reco_edges[0].value,3)},"
        range_message+= f" {np.round(energy_reco_edges[-1].value,3)}]"
        range_message+= f" {energy_reco_edges.unit}. Energy bins: {len(DRM_ebounds)}."
        logger.info(range_message)
        
        # True
        dummy_array = energy_true_edges - energy_true_range[0]
        i_start_energy_true = np.argmin(np.abs(dummy_array.value))

        dummy_array = energy_true_edges - energy_true_range[1]
        i_stop_energy_true  = np.argmin(np.abs(dummy_array.value))
        
        # Reco
        dummy_array = energy_reco_edges - energy_reco_range[0]
        i_start_energy_reco = np.argmin(np.abs(dummy_array.value))

        dummy_array = energy_reco_edges - energy_reco_range[1]
        i_stop_energy_reco  = np.argmin(np.abs(dummy_array.value))
        
        # Slice        
        DRM = DRM[i_start_energy_true:i_stop_energy_true, i_start_energy_reco:i_stop_energy_reco]

        logger.info(f"Slice True Energies between indexes [{i_start_energy_true},{i_stop_energy_true}]")
        logger.info(f"Slice Reco Energies between indexes [{i_start_energy_reco},{i_stop_energy_reco}]")

    range_message = f"Response Matrix defined at True Energy Range:"
    range_message+= f" [{np.round(energy_true_edges[i_start_energy_true].value,3)},"
    range_message+= f" {np.round(energy_true_edges[  i_stop_energy_true].value,3)}]"
    range_message+= f" {energy_true_edges.unit}. Energy bins: {DRM.shape[0]}."
    logger.info(range_message)

    range_message = f"Response Matrix defined at Reco Energy Range:"
    range_message+= f" [{np.round(energy_reco_edges[i_start_energy_reco].value,3)},"
    range_message+= f" {np.round(energy_reco_edges[  i_stop_energy_reco].value,3)}]"
    range_message+= f" {energy_reco_edges.unit}. Energy bins: {DRM.shape[1]}."
    logger.info(range_message)

    logger.info(f"Response Matrix unit: {DRM_unit}.\n")
    
    return DRM * DRM_unit



def Goodbye(logger, execution_time_start):
    """
    Goodbye function. Mostly counts time.

    Parameters
    ----------
    logger : `logging.Logger`
        Logger from main.
    execution_time_start: float
        Time in seconds at program beginning since epoch.
    """
    logger.info(f"Total Runtime =  {np.round(time()-execution_time_start,3)} s. Goodbye!")
    return 0



