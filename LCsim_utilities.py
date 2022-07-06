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
    observation_livetime = configuration['Observation_Livetimes'] * u.Unit(configuration['Time_Unit'])
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

def Goodbye(logger, execution_time_start):
    """
    Goodbye function. Mostly counts time.
    """
    logger.info(f"Total Runtime =  {np.round(time()-execution_time_start,3)} (s). Goodbye!")
    return 0



