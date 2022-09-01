###########################################
#                                         #
#        INITIALIZER                      #
#                                         #
###########################################

import astropy.units as u
import logging
import numpy as np
import os
import shutil
import yaml

from astropy.io import fits
from astropy.table import QTable
from yaml import SafeLoader

class Initializer():
    """
    Class to choose (randomly or by name) a transient from a catalogue and create an output directory.
    
    Parameters
    ----------
    log : `logging.Logger`
        Logger for terminal and .log file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    output_directory : str
        Directory used to write and save output.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    """
    def __init__(self) -> None:
        pass
    
    def set_logger(self, logger):
        """Setter for logger"""
        self.log = logger
        return self
    
    def set_transient(self, transient):
        """Setter for transient"""
        self.transient = transient
        return self
    
    def set_configuration(self, configuration):
        """Setter for configuration"""
        self.configuration = configuration
        return self
    
    def set_output_directory(self, output_directory):
        """Setter for output_directory"""
        self.output_directory = output_directory
        return self   
    
    def _choose_transient(self):
        """
        Choose a transient either randomly or a by requested name.
        """
        
        # Load Catalogue of transients
        self.log.info("Loading Sources Catalogue...")
        with fits.open(self.configuration['Input_Catalogue']) as hdulist:
            table_catalog = QTable.read(hdulist['CATALOG'])
            table_catalog.add_index('name')
            
        # Choose a Burst # Note: we are assuming all names are different.
        if self.configuration['Name_Transient'] is not None:
            try:
                transient = table_catalog.iloc[table_catalog['name'] == self.configuration['Name_Transient']][0]
                self.log.info('Requested transient available.\n')
            except IndexError:
                self.log.error(f"Transient {self.configuration['Name_Transient']} was not found.")
                raise
        else:
            rng = np.random.default_rng(self.configuration['Random_Seed'])
            random_index = rng.integers(0, high=len(table_catalog))
            transient = table_catalog[random_index]
            self.log.info('Transient randomly sampled.\n')
        
        self.set_transient(transient)
        return None
        
        
    def _create_output_directory(self):
        """
        Create the output directory according to parameter names.       
        """
        
        livetime = self.configuration['Observation_Livetime'] * u.Unit(self.configuration['Time_Unit'])
        energy_unit = u.Unit(self.configuration['Energy_Unit'])
        energy_range_reco = self.configuration['Energy_Range_Reco'] * energy_unit

        output_directory = self.configuration['Output_Directory']
        output_directory+= self.transient['name'] + '/'
        output_directory+= self.configuration['Name_Instrument' ] + '_'
        output_directory+= self.configuration['Name_Detector'   ] + '_'
        output_directory+= f"{int(livetime.to('ms').value)}ms_"
        output_directory+= f"{int(energy_range_reco.value[0])}_"
        output_directory+= f"{int(energy_range_reco.value[1])}_"
        output_directory+= energy_unit.to_string() + '/'
        if self.configuration['Output_Run_ID'] is not None:
            output_directory += self.configuration['Output_Run_ID']+'/'

        os.makedirs(os.path.dirname(output_directory), exist_ok=True)
        
        self.set_output_directory(output_directory)
        
        return None
    
        
    def _load_YAML(self, Configuration_file_path):
        """
        Parameters
        ----------
        Configuration_file_path : str
            Name of the input YAML file.
        """
        
        # Load YAML as a dict
        self.log.info("Loading YAML file...")
        with open(Configuration_file_path) as f:
            configuration = yaml.load(f, Loader = SafeLoader)
        
        self.set_configuration(configuration)
        return None
    
    
    
    
    def run_initialization(self, Configuration_file_path):
        """
        Read the YAML Configuration file into a dictionary.
        Load the Transient Catalogue and choose one transient.
        Create the output directory and a log file.

        Parameters
        ----------
        Configuration_file_path : str
            Name of the input YAML file.
        """

        self.log.info(f"{30*'='}INITIALIZATION{56*'='}\n")
        
        # Load and set YAML file as dict configuration
        self._load_YAML(Configuration_file_path)

        # Load Catalogue of transients and choose one. Set transient.
        self._choose_transient()

        # Create and set output_directory. Copy initialization file into it.
        self._create_output_directory()
        shutil.copy(Configuration_file_path, os.path.join(self.output_directory,"initializer.yml"))
        
        # Define Logger for file
        f_handler = logging.FileHandler(self.output_directory+'file.log', mode='w')
        f_handler.setLevel(logging.INFO)
        f_handler.setFormatter(logging.Formatter('%(asctime)s. %(levelname)s: %(message)s'))# -%(name)s
        self.log.addHandler(f_handler)

        # Log some info
        self.log.info(f"Configuration. YAML File : {Configuration_file_path}")
        self.log.info(f"Configuration. Source Catalogue: {self.configuration['Input_Catalogue']}")
        if self.configuration['Name_Transient'] is not None:
            self.log.info(f"Configuration. Requested Transient Name: {self.transient['name']}")
        else:
            self.log.info(f"Configuration. Random Transient Name: {self.transient['name']}")
        self.log.info(f"Configuration. Output Directory: {self.output_directory} \n")
        self.log.info(f"{100*'='}\n")

        return None
    
    