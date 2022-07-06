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
    logger.info('Welcome! A proper Welcome must be implemented.')
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
            logger.info('Requested transient available.')
        except IndexError:
            logger.error(f"Transient {configuration['Name_Transient']} was not found.")
            raise
    else:
        rng = np.random.default_rng(configuration['Random_Seed'])
        random_index = rng.integers(0, high=len(table_catalog))
        transient = table_catalog[random_index]
        logger.info('Transient randomly sampled.')
    
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

    logger.info(f"YAML File : {Configuration_file_path}")
    logger.info(f"Source Catalogue: {configuration['Input_Catalogue']}")

    if configuration['Name_Transient'] is not None:
        logger.info(f"Requested Transient Name: {transient['name']}")
    else:
        logger.info(f"Random Transient Name: {transient['name']}")

    logger.info(f"Output Directory: {output_directory}")
    
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

    logger.info(f"Trigger Time (format={time_format}): {trigger_time}")
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
    logger.info(f"Number of Observations: {observations_number}")
    logger.info(f"Observation start: {observations_start[0]}")
    logger.info(f"Observation livetime: {observations_livetimes[0]}")

    return observations_number, observations_start , observations_livetimes


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



