from time import time
EXECUTION_TIME_START = time()

import argparse
from LCsim_utilities import *


def Simulator(configuration):
    """
    Main body of the script. It executes the simulator.
    """
    # Section 1: define variables for the whole pipeline
    # # Transient Information
    # Name_Transient = configuration['Name_Transient']
    # Name_Instrument= configuration['Name_Instrument']
    # Name_Detector  = configuration['Name_Detector']

    # # # IRFs files
    # File_rsp = configuration['Input_rsp']
    # File_arf = configuration['Input_arf']
    # File_rmf = configuration['Input_rmf']

    # # Insert exception here


    # File_bak = configuration['Input_bak']
    # File_Light_Curve = configuration['Input_Light_Curve']


    # # Energy Axes Parameters
    # Energy_interpolation = data['Energy_interpolation']
    # Energy_slice = data['Energy_slice']
    # Energy_unit = u.Unit(data["Energy_unit"])
    
    # custom_range_energy_reco = tuple(data['Energy_range_reco']) * Energy_unit
    # custom_range_energy_true = tuple(data['Energy_range_true']) * Energy_unit
    
    
    # # Observation Parameters
    # Number_of_LightCurves = data['N_light_curves']
    # Time_unit = u.Unit(data['Time_unit'])
    
    # t_start_obs= data['Observation_time_start'] * Time_unit
    # t_stop_obs = data['Observation_time_stop' ] * Time_unit
    # live_t_obs = data['Observation_livetimes' ] * Time_unit
    # dead_times = data['Observation_deadtimes' ] * Time_unit
    
    # # Create Output Directory
    # Output_Directory = configuration['Output_Directory']
    # Output_Directory+= Name_Transient + '/'
    # Output_Directory+= Name_Instrument + '_' + Name_Detector + '_'
    # Output_Directory+= str(int(live_t_obs.to('ms').value)) + 'ms_'
    # Output_Directory+= str(int(custom_range_energy_reco.value[0])) + '_'
    # Output_Directory+= str(int(custom_range_energy_reco.value[1])) + '_'
    # Output_Directory+= Energy_unit.to_string()+'/'
    
    # if configuration['Output_run_id'] is not None:
    #     Output_Directory += configuration['Output_run_id']+'/'
    
    # os.makedirs(os.path.dirname(Output_Directory), exist_ok=True)


    logger.info('Main body')



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

    # Load Configuration YAML file
    configuration, transient = Initialize(logger, args.configurationfile)
    
    # Execute the simulator
    Simulator(configuration)

    # Goodbye
    Goodbye(logger, EXECUTION_TIME_START)

