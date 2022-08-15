from time import time
EXECUTION_TIME_START = time()

from LCsim_utilities import *
from simulator_irfs import *
from simulator_models import *

import argparse

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

    # Define Gammapy Time Axis format
    TimeMapAxis.time_format = "iso" 
    # Define the Reference Time as the Burst Trigger Time
    trigger_time_t0 = Define_Reference_Time(transient, TimeMapAxis.time_format, logger)

    # Define Pointing Direction (FoV Centre) as SkyCoord
    pointing = SkyCoord(
                transient['ra'].value,
                transient['dec'].value,
                unit = transient['ra'].unit,
                frame = 'fk5',
                equinox='J2000'
                )
    logger.info(f"Define Pointing Direction: {pointing}.\n")

    # Define Instrument FoV Axes: Offset, FovLon, FoVLat
    axis_offset, axis_fovlon, axis_fovlat = Define_FoV_Axes(logger)

    # Define Number of Observations, Starting Times, Livetimes.
    observations_number, observations_start , observations_livetimes = Define_Schedule(configuration,
                                                                                       trigger_time_t0,
                                                                                       logger
                                                                                      )





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
    else:
        raise NotImplementedError(f"Functions for instrument {configuration['Name_Instrument']} not implemented. Accepted: GBM, COSI.")


    # Read and define the Reconstructed Energy Axis from EBOUNDS HDU of RMF/RSP
    axis_energy_reco = Define_Energy_Axis(File_rmf,
                                          "EBOUNDS",
                                          configuration,
                                          logger,
                                          energy_is_true = False
                                         )
    # Read and define the True Energy Axis from MATRIX/SPECRESP MATRIX HDU of RMF/RSP
    axis_energy_true = Define_Energy_Axis(File_rmf,
                                          Hdu_edisp,
                                          configuration,
                                          logger,
                                          energy_is_true = True
                                         )

    # Define the Source Geometry
    geom = Define_Geometry(pointing, axis_energy_reco, logger)  


    # Read the Detector Response Matrix into a 2D Astropy Quantity (vs Energy True, Reco)
    Detector_Response_Matrix = Read_Response_Matrix_from_RSP(File_rmf,
                                                                Hdu_edisp,
                                                                "EBOUNDS",
                                                                configuration,
                                                                logger
                                                                )

    # Read the Effective Area into a 1D Astropy Quantity (vs Energy True)
    if configuration['Name_Instrument']=="COSI":
        aeff_array = Read_Effective_Area_from_ARF(File_arf,
                                                    Hdu_aeff,
                                                    configuration,
                                                    logger
                                                    )
    elif configuration['Name_Instrument']=="GBM":
        logger.info("Compute Effective Area.")
        aeff_array = np.sum(Detector_Response_Matrix, axis=1)

    # Define the Effective Area as a Gammapy object
    Effective_Area = Compute_Effective_Area_2D(aeff_array,
                                               axis_offset,
                                               axis_energy_true,
                                               logger,
                                               configuration,
                                               transient,
                                               output_directory
                                              )

    
    # Define the Energy Dispersion Matrix as a Gammapy object (with an Exposure Map)
    Energy_Dispersion_Matrix_Map = Compute_Energy_Dispersion_Matrix(Detector_Response_Matrix,
                                                                    axis_energy_true,
                                                                    axis_energy_reco,
                                                                    Effective_Area,
                                                                    observations_livetimes,
                                                                    geom,
                                                                    logger,
                                                                    configuration,
                                                                    transient,
                                                                    output_directory)



    # Load the Background Spectrum into a 1D Astropy Quantity (vs Energy Reco)
    bak_model = Read_Background_Spectrum(configuration,
                                       geom,
                                       logger
                                      )
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
                                            correction_factor
                                        )

    # Implement Sky Model
    logger.info("Define the Sky Model: Spectral Model * Temporal Model")
    model_simulations = SkyModel(spectral_model = Spectral_Model,
                                 temporal_model = Temporal_Model,
                                 name = configuration['Spectral_Model_Name']+'-'+tem_mod_name
                                )
    Plot_Sky_Model(model_simulations, configuration, transient, logger, axis_energy_true, trigger_time_t0,
                    correction_factor, output_directory)





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
        #logger.info(f'Creating Dataset {idx+1}/{observations_number}...             \r', end = '')

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


    # Section 5 - Export
    hdu_list = []
    hdu_list.append(fits.PrimaryHDU())

    qtable = QTable(datasets_generic.info_table())
    hdu_list.append(fits.table_to_hdu(qtable))

    for i_LC in tqdm(range(N_Light_Curves),desc='Writing Light Curves'):
        qtable = QTable(List_of_Datasets[i_LC].info_table())
        hdu_list.append(fits.table_to_hdu(qtable))

    hdu_list = fits.HDUList(hdu_list)

    logger.info(f"Write Lightcurves: {output_directory}lightcurves.fits")
    hdu_list.writeto(output_directory+"lightcurves.fits", overwrite=True)
    


    return None



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

    # 3 - Load Configuration YAML file and Choose a Transient
    configuration, transient, output_directory = Initialize(logger, args.configurationfile)
    
    # 4 - Execute the simulator (Main function)
    Simulator(configuration, transient, output_directory)

    # 5 - Goodbye
    Goodbye(logger, EXECUTION_TIME_START)

