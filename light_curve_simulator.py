from time import time
EXECUTION_TIME_START = time()

from simulator_configurator import *
from simulator_irfs import *
from simulator_models import *

from simulator_initializer import Initializer
from simulator_dataset_creator_XSPEC import Dataset_Creator_GBM, Dataset_Creator_COSI
from simulator_dataset_creator import Dataset_Creator_CTA

import argparse
import logging

def Simulator(configuration, transient, output_directory):
    """
    Main body of the script. It executes the simulator.
    
    Parameters
    -------
    configuration : dict
        Dictionary with the parameters from the YAML file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    output_directory : str
        Directory used to write and save output.
        
    Returns
    -------
        None
    """
    
    
    # Section 1: Define the general configuration for the simulation.
    logger.info(f"{30*'='}SECTION 1: SIMULATION CONFIGURATION{36*'='}\n")
    
    # Define the object that holds the general configuration.
    if configuration['Name_Instrument']=="GBM":
        configurator = Dataset_Creator_GBM(logger)
        File_rmf = configuration['Input_rsp']
        Hdu_edisp= "SPECRESP MATRIX"
        File_arf = None
        Hdu_aeff = None
        
    elif configuration['Name_Instrument']=="COSI":
        configurator = Dataset_Creator_COSI(logger)
        File_rmf = configuration['Input_rmf']
        Hdu_edisp= "MATRIX"
        File_arf = configuration['Input_arf']
        Hdu_aeff = "SPECRESP"
        
    elif configuration['Name_Instrument']=="CTA":
        configurator = Dataset_Creator_CTA(logger)
        raise NotImplementedError(f"Functions for instrument {configuration['Name_Instrument']} not implemented. Accepted: GBM, COSI.")
    else:
        raise NotImplementedError(f"Functions for instrument {configuration['Name_Instrument']} not implemented. Accepted: GBM, COSI.")

    
    # Define the Reference Time as the Burst Trigger Time
    TimeMapAxis.time_format = "iso"
    configurator.set_reference_time(transient, TimeMapAxis.time_format)
 
    # Define Number of Observations, Starting Times, Livetimes.
    configurator.define_schedule(configuration)

    observations_number    = configurator.observations_number
    observations_start     = configurator.observations_start
    observations_livetimes = configurator.observations_livetimes

    logger.info(f"{100*'='}\n")


    # Section 2: Define the Detector Response
    logger.info(f"{30*'='}SECTION 2: DEFINE INSTRUMENT RESPONSE{34*'='}\n")
    logger.info(f"Responses for instrument: {configuration['Name_Instrument']}, detector {configuration['Name_Detector']}.\n")
    
    # Read and define the Reconstructed Energy Axis from EBOUNDS HDU of RMF/RSP
    configurator.set_axis_energy_reco(File_rmf, "EBOUNDS", configuration)
    
    # Read and define the True Energy Axis from MATRIX/SPECRESP MATRIX HDU of RMF/RSP
    configurator.set_axis_energy_true(File_rmf, Hdu_edisp, configuration)
    
    # Define the Source Geometry around the Transient Coordinates
    configurator.set_geometry(transient) # radius=1.0

    # Read the Detector Response Matrix into a 2D Astropy Quantity (vs Energy True, Reco)
    configurator.read_response_matrix_from_RSP(File_rmf, Hdu_edisp, configuration)

    # Read the Effective Area into a 1D Astropy Quantity (vs Energy True)
    configurator.compute_effective_area_array(File_arf, Hdu_aeff, configuration, transient, output_directory)
    
    # Compute the Exposure Map
    map_exposure, map_exposure_edisp = configurator.compute_exposure_map(configurator.aeff_array,
                                                                         observations_livetimes
                                                                         )

    
    # Define the Energy Dispersion Matrix as a Gammapy object (with an Exposure Map)
    energy_dispersion_matrix_map = configurator.compute_energy_dispersion_map(map_exposure_edisp,
                                                                              configuration,
                                                                              transient,
                                                                              output_directory
                                                                              )


    # Load the Background Spectrum into a 1D Astropy Quantity (vs Energy Reco)
    bak_model = configurator.read_background_spectrum(configuration)
    
    # Compute the Background as a Map for the SpectrumDataset
    background_map = configurator.compute_background_map(bak_model, configuration, transient, output_directory)
    
    # Define the Background as a Gammapy Object
    if configuration['Name_Instrument']=="COSI":
        axis_energy_reco_bkg = axis_energy_reco
    elif configuration['Name_Instrument']=="GBM":
        logger.warning("GBM wants another Reco Energy Axis for the Background, taken from BAK, not RSP.")
        axis_energy_reco_bkg = Define_Energy_Axis(configuration['Input_bak'],
                                                  "EBOUNDS",
                                                  configuration,
                                                  logger,
                                                  energy_is_true = False
                                                 )

    Background = Compute_Background_3D(bak_model,
                                        axis_energy_reco_bkg,
                                        axis_fovlon,
                                        axis_fovlat,
                                        geom,
                                        logger,
                                        configuration,
                                        transient,
                                        output_directory
                                    )
    
    # Define the IRFs Dictionary
    IRFs = {'aeff' : Effective_Area, 'bkg'  : Background}
    
    logger.info(f"{100*'='}\n")



    logger.info(f"{30*'='}SECTION 3: DEFINE SPECTRAL AND TEMPORAL MODELS{25*'='}\n")

    # Section 3: Define the Models
    logger.info("Define the Temporal and Spectral Models.\n")

    # Define the Temporal Model
    if configuration['Light_Curve_Template']:
        Temporal_Model, correction_factor = Define_Template_Temporal_Model(trigger_time_t0,
                                                                            configuration,
                                                                            logger
                                                                          )
        tem_mod_name = "Empirical"
    else:
        logger.warning("Light Curve temporal model shape not provided. Assuming Gaussian Pulse.")
        Temporal_Model, correction_factor = Define_Gaussian_Pulse(trigger_time_t0,
                                                                    configuration,
                                                                    transient,
                                                                    logger    
                                                                )
        tem_mod_name = "Gaussian"

    # Implement Spectral Model
    Spectral_Model = Define_Spectral_Model(configuration,
                                            transient,
                                            logger,
                                            correction_factor,
                                            axis_energy_true
                                        )

    # Implement Sky Model
    logger.info("Define the Sky Model: Spectral Model * Temporal Model")
    model_simulations = SkyModel(spectral_model = Spectral_Model,
                                 temporal_model = Temporal_Model,
                                 name = configuration['Spectral_Model_Name']+'-'+tem_mod_name
                                )
    Plot_Sky_Model(model_simulations, configuration, transient, logger, axis_energy_true, trigger_time_t0,
                    correction_factor, output_directory)

    logger.info(f"{100*'='}\n")


    logger.info(f"{30*'='}SECTION 4: PERFORM THE SIMULATIONS{37*'='}\n")

    # Section 4 - Perform the simulations
    N_Light_Curves = configuration['N_Light_Curves']

    logger.info(f"Perform the simulation of {N_Light_Curves} Light Curves.")
    logger.info(f"Each Light Curve will contain {observations_number} points.")

    # Create a List of `Datasets`, each will be a Light curve
    List_of_Datasets = []
    # Fill it with empty `Datasets`
    for i_LC in range(N_Light_Curves):
        List_of_Datasets.append(Datasets())

    # Create an empty `SpectrumDataset` object to host geometry and energy info.
    empty = SpectrumDataset.create(geom = geom,
                                   energy_axis_true = axis_energy_true,
                                   name = "empty",
                                   edisp = Energy_Dispersion_Matrix_Map
                                  )

    # Create Maker object.
    # containment_correction must be set True if I have a PSF, to adapt the
    # exposure map and account for PSF broadening and the PSF containment.
    maker = SpectrumDatasetMaker(selection = ["exposure", "background"],
                                 containment_correction = False
                                )

    # This mock light curve will store info on the predicted background and excess rates.
    # It won't contain any simulation.
    datasets_generic = Datasets()

    LOOP_START = time()

    for idx in tqdm(range(observations_number), desc='Loop observations'):

        # Set the current observation. Observations differ only in their starting time.
        obs = Observation.create(pointing      = pointing,
                                 livetime      = observations_livetimes[idx],
                                 tstart        = observations_start[idx],
                                 irfs          = IRFs,
                                 reference_time= trigger_time_t0,
                                 obs_id        = idx
                                )

        # Set name according to observation number. Set geometry and energy information.
        dataset_generic = empty.copy(name = f"dataset-{idx}")

        # Run: creates the SpectrumDataset. It also sets the Background counts.
        dataset_generic = maker.run(dataset_generic, obs)

        # Set the Energy Disperison    
        dataset_generic.edisp = Energy_Dispersion_Matrix_Map

        # Set the source model and compute the predicted excess counts from it.
        dataset_generic.models = model_simulations

        # For this observation simulate background and counts for each different light curve
        for i_LC in range(N_Light_Curves):

            dataset = dataset_generic.copy(name = f"LC-{i_LC}"+f"--Dataset-{idx}")

            # Set models, that CANNOT BE COPIED
            dataset.models = model_simulations

            # Simulate counts from Poisson(avg=Predicted Backgrounds+Predicted count excess)
            dataset.fake()

            # Add this dataset to the right collection.
            datasets = List_of_Datasets[i_LC]
            datasets.append(dataset)

        datasets_generic.append(dataset_generic)


        # Repeat for a new observation.

    logger.info(f"Loop Runtime = {np.round(time()-LOOP_START,3)} s.\n")


    logger.info(f"{100*'='}\n")


    logger.info(f"{30*'='}SECTION 5: EXPORT RESULTS{46*'='}\n")

    # Section 5 - Export Results
    Time_Centroids = observations_start - trigger_time_t0
    Time_Centroids = Time_Centroids.to("s") + observations_livetimes/2.0

    Write_Light_Curves(Time_Centroids, List_of_Datasets, logger, output_directory)
    
    # 5.1 - Print the simulated light curves
    Print_Light_Curves(Time_Centroids, List_of_Datasets, output_directory, configuration, transient)
    
    logger.info(f"{100*'='}\n")


    return None


def Welcome(logger):
    """
    Welcome function. This is used to print info about the code on execution.
    
    Parameters
    ----------
    logger : `logging.Logger`
        Logger from main.
    """
    logger.info('Welcome! A proper Welcome must be implemented.\n')
    return None

def Goodbye(logger):
    """
    Goodbye function. Mostly counts time.

    Parameters
    ----------
    logger : `logging.Logger`
        Logger from main.
    """
    logger.info(f"Total Runtime =  {np.round(time()-EXECUTION_TIME_START,3)} s. Goodbye!\n")
    return 0


if __name__ == '__main__':

    # Configure logger
    logging.basicConfig(
        level=logging.INFO,
        format ='%(levelname)s: %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # 1 - Print message with initial info
    Welcome(logger)

    # 2 - Set Script Arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--configurationfile", help="name of the configuration YAML file")
    args = parser.parse_args()

    # 3 - Load Configuration YAML file, Choose a Transient and an output directory.
    initializer = Initializer()
    initializer.set_logger(logger)
    initializer.run_initialization(args.configurationfile)
    
    # 4 - Execute the simulator (Main function)
    Simulator(initializer.configuration, initializer.transient, initializer.output_directory)

    # 5 - Goodbye
    Goodbye(logger)

