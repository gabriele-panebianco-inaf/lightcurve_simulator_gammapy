###########################################
#                                         #
#     CLASSES TO MAKE DL4 FROM IRFS       #
#     SAVED AS CTA IRFS                   #
###########################################

import astropy.units as u
import numpy as np


from astropy.time import Time

from gammapy.datasets import SpectrumDataset
from gammapy.irf import load_cta_irfs



class Dataset_Creator():
    """
    Abstract class to create the DL4 that are going to be filled.
    
    Parameters
    ----------
    log : `logging.Logger`
            Logger.
    spectrum_dataset : `gammapy.datasets.SpectrumDataset`
            Spectrum Dataset with the reduced DL4 IRFs.
    """
    def __init__(self, log):
        """
        Constructor.
    
        Parameters
        ----------
        log : `logging.Logger`
            Logger.
        """
        self.log = log
        return None
    
    
    def set_reference_time(self):
        raise NotImplementedError("Call funciton from a derived class.")
        pass
        
    
    def define_schedule(self, configuration):
        """
        Return the Schedule for the simulation:
        Number of observations, Start time and duration of each observation.

        Parameters
        ----------
        configuration : dict
            Dictionary with the parameters from the YAML file. 
        """
        self.log.info("Define the Observations' schedule.")

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
        observations_start = Time(self.reference_time + observations_start)

        # Adjust the number of observations after resizing observations_start
        observations_number = observations_start.size

        # Log
        self.log.info(f"Number of Observations: {observations_number}.")
        self.log.info(f"Observation start wrt Trigger Time: {Time_Start}. Absolute value: {observations_start[0]}.")
        self.log.info(f"Observation stop  wrt Trigger Time: {Time_Stop}.")
        self.log.info(f"Observation livetime: {np.round(observations_livetimes[0],4)}.\n")

        # Set attributes
        self.observations_number    = observations_number
        self.observations_start     = observations_start
        self.observations_livetimes = observations_livetimes

        return None
    
    def set_spectrum_dataset(self,
                             geometry,
                             energy_axis_true,
                             reference_time,
                             energy_dispersion_map,
                             exposure_map,
                             background_map
                             ):
        """
        Setter for the spectrum_dataset.
        
        Parameters
        ----------
        geometry : `gammapy.maps.region.geom.RegionGeom`
            Circular Source Geometry.
        energy_axis_true : `gammapy.maps.MapAxis`
            True and Reconstructed Energy Axis.
        reference_time : `astropy.time.Time`
            Reference Time. It is defined as the Trigger time of the Transient.
        energy_dispersion_map : `gammapy.irf.EDispKernelMap`
            Energy Dispersion Map created from kernel.
        exposure_map : `gammapy.maps.RegionNDMap`
            Exposure NDMap (vs True Energy).
        background_map : `gammapy.maps.RegionNDMap`
            Background NDMap (vs Energy).
        """
        self.spectrum_dataset = SpectrumDataset.create(geom = geometry,
                                                       energy_axis_true = energy_axis_true,
                                                       reference_time = reference_time,
                                                       name = "Empty_dataset",
                                                       edisp = energy_dispersion_map,
                                                       exposure = exposure_map,
                                                       background = background_map
                                                       )
        return self


    
    
class Dataset_Creator_CTA(Dataset_Creator):
    """Class to create the DL4 that are going to be filled from CTA IRFs."""
    def __init__(self, log):
        super().__init__(log)
        return None





