###########################################
#                                         #
#     CLASSES TO MAKE DL4 FROM IRFS       #
#     SAVED AS CTA IRFS                   #
###########################################


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





