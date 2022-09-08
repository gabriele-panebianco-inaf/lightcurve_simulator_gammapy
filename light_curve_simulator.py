from time import time
EXECUTION_TIME_START = time()

import argparse
import logging

from simulator_initializer import Initializer
from simulator_simulator import Simulator

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
    duration = time()-EXECUTION_TIME_START
    logger.info(f"Total Runtime =  {duration:.3f} s. Goodbye!\n")
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
    simulator = Simulator()
    simulator.set_logger(logger).set_configuration(initializer.configuration)
    simulator.set_transient(initializer.transient).set_output_directory(initializer.output_directory)

    simulator.run_simulation()

    # 5 - Goodbye
    Goodbye(logger)

