from time import time
EXECUTION_TIME_START = time()

import argparse
from LCsim_utilities import *


def Simulator(configuration, transient):
    """
    Main body of the script. It executes the simulator.
    Parameters
    -------
    configuration : dict
        Dictionary with the parameters from the YAML file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    Returns
    -------
        None
    """
    
    # Section 1: Define the general configuration for the simulation.

    # Define Gammapy Time Axis format
    TimeMapAxis.time_format = "iso" 
    # Define the Reference Time as the Burst Trigger Time
    trigger_time_t0 = Define_Reference_Time(transient, TimeMapAxis.time_format, logger)

    # Define Pointing Direction (FoV Centre) as SkyCoord
    pointing = SkyCoord(transient['ra'].value, transient['dec'].value,
                        unit = transient['ra'].unit, frame = 'fk5', equinox='J2000')
    logger.info(f"Define Pointing Direction: {pointing}.\n")

    # Define Instrument FoV Axes: Offset, FovLon, FoVLat
    axis_offset, axis_fovlon, axis_fovlat = Define_FoV_Axes(logger)

    # Define Number of Observations, Starting Times, Livetimes.
    observations_number, observations_start , observations_livetimes = Define_Schedule(configuration,
                                                                                       trigger_time_t0,
                                                                                       logger
                                                                                      )

    # Load the Empirical Light Curve for Comparison
    # -----------------TO DO?

    # Section 2: Define the Detector Response
    logger.info(f"Responses for instrument: {configuration['Name_Instrument']}, detector {configuration['Name_Detector']}.\n")
    
    # GBM and COSI have different types of response file
    if configuration['Name_Instrument']=="COSI":
        File_rmf = configuration['Input_rmf']
        Hdu_edisp= "MATRIX"

        File_arf = configuration['Input_arf']
        Hdu_aeff = "SPECRESP"

    elif configuration['Name_Instrument']=="GBM":
        File_rmf = configuration['Input_rsp']
        Hdu_edisp= "SPECRESP MATRIX"
        Detector_Response_Matrix = Define_Response_Matrix_rfom_RSP(File_rmf,
                                                                   Hdu_edisp,
                                                                   "EBOUNDS",
                                                                   configuration,
                                                                   logger
                                                                  )
    else:
        logger.error(f"Functions for instrument {configuration['Name_Instrument']} not implemented. Accepted: GBM, COSI.")
        exit() # Sistemare sta cosa

    # Define the Reconstructed Energy Axis from EBOUNDS HDU of RMF/RSP
    axis_energy_reco = Define_Energy_Axis(File_rmf,
                                          "EBOUNDS",
                                          configuration,
                                          logger,
                                          energy_is_true = False
                                         )
    # Define the True Energy Axis from MATRIX/SPECRESP MATRIX HDU of RMF/RSP
    axis_energy_true = Define_Energy_Axis(File_rmf,
                                          Hdu_edisp,
                                          configuration,
                                          logger,
                                          energy_is_true = True
                                         )

    # Define the Source Geometry
    geom = Define_Geometry(pointing, axis_energy_reco, logger)  


    # Define the Effective Area
    if configuration['Name_Instrument']=="COSI":
        aeff_array = Define_Effective_Area_from_ARF(File_arf,
                                                    Hdu_aeff,
                                                    configuration,
                                                    logger
                                                    )
    elif configuration['Name_Instrument']=="GBM":
        logger.info("Compute Effective Area.")
        aeff_array = np.sum(Detector_Response_Matrix, axis=1)
     
      

    #
    logger.info(f"Currently here!")

    return None



if __name__ == '__main__':

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format ='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Print message with initial info
    Welcome(logger)

    # Set Script Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--configurationfile", help="name of the configuration YAML file")
    args = parser.parse_args()

    # 1 - Load Configuration YAML file and Choose a Transient
    configuration, transient = Initialize(logger, args.configurationfile)
    
    # 2 - Execute the simulator
    Simulator(configuration, transient)

    # Goodbye
    Goodbye(logger, EXECUTION_TIME_START)

