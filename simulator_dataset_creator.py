###########################################
#                                         #
#     CLASSES TO MAKE DL4 FROM IRFS       #
#                                         #
###########################################

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time

from gammapy.irf import EffectiveAreaTable2D, load_cta_irfs
from gammapy.maps import MapAxis, RegionGeom


FIGURE_FORMAT = ".pdf"

class Dataset_Creator():
    """Abstract class to create the DL4 that are going to be filled."""
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


class Dataset_Creator_From_XSPEC(Dataset_Creator):
    """
    Abstract class to create the DL4 that are going to be filled
    with IRF data stored in XSPEC IRF files (.RSP, .RMF, .ARF).
    
    Parameters
    ----------
    reference_time: `astropy.time.Time`
        Reference Time. It is defined as the Trigger time of the Transient.
    pointing : `astropy.coordinates.SkyCoord`
        Pointing Direction (FoV centre).
    observations_number : int
        Number of Observations.
    observations_start : `astropy.time.core.Time`
        Absolute start time of each observation.
    observations_livetimes : `astropy.units.quantity.Quantity`
        Array of duration of the observations.
    axis_energy_true, axis_energy_reco : `gammapy.maps.MapAxis`
        True and Reconstructed Energy Axis.
    _axis_offset, _axis_fovlon, _axis_fovlat : `gammapy.maps.axes.MapAxis`
        FoV Axes.
    geometry : `gammapy.maps.region.geom.RegionGeom`
        Circular Source Geometry.
    detector_response_matrix : `astropy.units.Quantity`
        Detector Response Matrix read from IRF file.
    log : `logging.Logger`
        Logger.
    """
    def __init__(self, log):
        super().__init__(log)
        return None
        

    def set_reference_time(self, transient, time_format):
        """
        Set the Reference Time of the simulation as the Transient Trigger Time.

        Parameters
        ----------
        transient: `astropy.table.row.Row`
            Row of the chosen transient.
        time_format: str
            Time format of the TimeMapAxis.
        """
        trigger_time = Time(transient['trigger_time'], format='mjd', scale='utc')
        
        trigger_time.format = time_format
        trigger_time = trigger_time.tt

        self.log.info(f"Define Trigger Time (format={time_format}, scale={trigger_time.scale}): {trigger_time}.\n")
        
        # Set attribute
        self.reference_time = trigger_time
        return None
    
    
    def _set_FoV_axes(self):
        """
        Set the FoV Axes: Offset, Lon, Lat.
        """
        
        self.log.info(f"Set Instrument FoV Axes.")

        # Assumption: The FoV Center is (0,0), FoV_Lon and FoV_Lat are symmetric wrt to 0.0
        fov_maximum_offset = 5.0 * u.deg
        fov_n_bin = 5
    
        # Create Instrument Offset axis
        axis_offset = MapAxis.from_bounds(0.0,
                                          fov_maximum_offset.value,
                                          unit = fov_maximum_offset.unit,
                                          nbin = fov_n_bin,
                                          name = "offset"
                                          )
        # Create Instrument FoV_lon axis
        axis_fovlon = MapAxis.from_bounds(-fov_maximum_offset.value/2.0,
                                          +fov_maximum_offset.value/2.0,
                                          unit = fov_maximum_offset.unit,
                                          nbin = fov_n_bin,
                                          name = "fov_lon"
                                          )
        # Create Instrument FoV_lat axis
        axis_fovlat = MapAxis.from_bounds(-fov_maximum_offset.value/2.0,
                                          +fov_maximum_offset.value/2.0,
                                          unit = fov_maximum_offset.unit,
                                          nbin = fov_n_bin,
                                          name = "fov_lat"
                                          )
        #self.log.info(axis_offset)
        #self.log.info(axis_fovlon)
        #self.log.info(axis_fovlat)
        
        # Set attributes
        self._axis_offset = axis_offset
        self._axis_fovlon = axis_fovlon
        self._axis_fovlat = axis_fovlat
        return None


    def set_pointing(self, transient, frame='fk5', equinox='J2000'):
        """
        Set the pointing direction (FoV centre).
        
        Parameters
        ----------
        transient: `astropy.table.row.Row`
            Row of the chosen transient.
        frame : str
            Coordinate frame.
        equinox : str
            Coordinate equinox
        """
        self.pointing = SkyCoord(transient['ra'].value,
                                 transient['dec'].value,
                                 unit = transient['ra'].unit,
                                 frame = frame,
                                 equinox=equinox
                                 )              
        self.log.info(f"Define Pointing Direction: {self.pointing}.\n")
        
        self._set_FoV_axes()
        return None
    
    
    

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
    
    
    def _define_energy_axis(self,
                            name_file_fits,
                            name_hdu,
                            configuration,
                            energy_is_true = False
                            ):
        """
        Returns an Energy Axis.

        Parameters
        ----------
        name_file_fits : str
            Name of a FITS file that must be opened to get the Energy Axis information.
        name_hdu : str
            Name of the HDU that contains the Energy Axis information.
        configuration : dict
            Dictionary with the parameters from the YAML file.
        energy_is_true : bool
            True if we want to create a True Energy Axis, False for a Reconstructed Energy Axis.

        Returns
        -------
        energy_axis : `gammapy.maps.MapAxis`
            True or Reconstructed Energy Axis
        """

        # Load Energy Table
        with fits.open(name_file_fits) as hdulist:
            energy_table = QTable.read(hdulist[name_hdu])

        # True or Reconstructed Energy?        
        if energy_is_true:
            energy_col_names = ("ENERG_LO", "ENERG_HI")
            energy_axis_name = "energy_true"
            msg = "True"
        else:
            energy_col_names = ("E_MIN", "E_MAX")
            energy_axis_name = "energy"
            msg = "Reconstructed"

        self.log.info(f"Read {msg} Energy Axis from: {name_file_fits}")

        # Define Edges Columns
        energy_col_min = energy_table[energy_col_names[0]]
        energy_col_max = energy_table[energy_col_names[1]]

        # Define Energy Unit
        Energy_Unit = u.Unit(configuration['Energy_Unit'])

        # Correct for Adimensional Columns
        if energy_col_min.unit in [None, u.Unit("")]:
            self.log.warning(f"Energy Axis file has adimensional values. Correct into {Energy_Unit}")
            energy_col_min = energy_col_min * Energy_Unit
            energy_col_max = energy_col_max * Energy_Unit

        # To avoid that min edge is 0
        energy_col_min[0] += 1e-2 * (energy_col_max[0] - energy_col_min[0])

        # Define Edges
        energy_edges = np.append(energy_col_min.value, energy_col_max.value[-1]) * energy_col_min.unit

        # Define Axis
        energy_axis = MapAxis.from_edges(energy_edges,
                                          name = energy_axis_name,
                                          interp = configuration['Energy_Interpolation']
                                         )

        # Slice Energy
        if configuration['Energy_Slice']:
            range_message = f"Original Energy Range: [{np.round(energy_axis.bounds[0].value,3)}, "
            range_message+= f"{np.round(energy_axis.bounds[1].value,3)}] "
            range_message+= str(energy_axis.unit) + f". Energy bins: {energy_axis.nbin}."
            self.log.info(range_message)

            if energy_is_true:
                energy_custom_range = configuration['Energy_Range_True'] * Energy_Unit
            else:
                energy_custom_range = configuration['Energy_Range_Reco'] * Energy_Unit

            dummy_array = energy_axis.edges - energy_custom_range[0]
            i_start_energy = np.argmin(np.abs(dummy_array.value))

            dummy_array = energy_axis.edges - energy_custom_range[1]
            i_stop_energy  = np.argmin(np.abs(dummy_array.value))

            energy_axis = energy_axis.slice(slice(i_start_energy, i_stop_energy))

            self.log.info(f"Slice between indexes [{i_start_energy},{i_stop_energy}].")

        range_message = f"Final Energy range: [{np.round(energy_axis.bounds[0].value,3)}, "
        range_message+= f"{np.round(energy_axis.bounds[1].value,3)}] "
        range_message+= str(energy_axis.unit)+f". Energy bins: {energy_axis.nbin}."
        self.log.info(range_message)
        self.log.info(energy_axis)

        return energy_axis
    
    
    def set_axis_energy_true(self, name_file_fits, name_hdu, configuration):
        """
        Set the True Energy Axis.
        
        Parameters
        ----------
        name_file_fits : str
            Name of a FITS file that must be opened to get the Energy Axis information.
        name_hdu : str
            Name of the HDU that contains the Energy Axis information.
        configuration : dict
            Dictionary with the parameters from the YAML file.
        """
        self.axis_energy_true = self._define_energy_axis(name_file_fits,
                                                         name_hdu,
                                                         configuration,
                                                         energy_is_true = True
                                                         ) 
        return self
    
    def set_axis_energy_reco(self, name_file_fits, name_hdu, configuration):
        """
        Set the Reconstructed Energy Axis.
        
        Parameters
        ----------
        name_file_fits : str
            Name of a FITS file that must be opened to get the Energy Axis information.
        name_hdu : str
            Name of the HDU that contains the Energy Axis information.
        configuration : dict
            Dictionary with the parameters from the YAML file.
        """
        self.axis_energy_reco = self._define_energy_axis(name_file_fits,
                                                         name_hdu,
                                                         configuration,
                                                         energy_is_true = False
                                                         ) 
        return self
    
    
    
    def set_geometry(self, radius=1.0):
        """
        Define the Source Sky Geometry: a circular region in the sky where we place the source
        and we assume that we can receive photons with reconstructed energy in the given axis.

        Parameters
        ----------
        radius: float
            radius of the Circular Geometry in degree. Must be smaller than the FoV.
        """
        self.log.info("Define Source Geometry.")
        geometry_radius = radius * u.deg
        source_geometry_str = self.pointing.frame.name + ';circle('
        source_geometry_str+= self.pointing.to_string().split()[0] + ', '
        source_geometry_str+= self.pointing.to_string().split()[1] + ', '
        source_geometry_str+= f"{geometry_radius.value})"

        # Set and Log
        self.geometry = RegionGeom.create(source_geometry_str, axes = [self.axis_energy_reco])
        self.log.info(self.geometry)

        return self
    
    
    
    def read_response_matrix_from_RSP(self,
                                      name_file_fits,
                                      name_hdu,
                                      configuration,
                                      ):
        """
        Return the Detector Response Matrix as a function of true and reconstructed energy with their unit.

        Parameters
        ----------
        name_file_fits: str
            Name of a FITS file that must be opened to get the Response as a function of True Energy.
        name_hdu: str
            Name of the HDU that contains the Response information as a function of True Energy.
        configuration : dict
            Dictionary with the parameters from the YAML file.        
        """

        self.log.info(f"Read Detector Response Matrix from: {name_file_fits}")

        # Load the Response
        with fits.open(name_file_fits) as hdulist:
            DETCHANS = hdulist[name_hdu].header["DETCHANS"]
            try:
                DRM_Unit = u.Unit(hdulist[name_hdu].header['TUNIT6'])
            except KeyError:
                self.log.warning(f"Key TUNIT6 not found. Assuming adimensional Response Matrix.")
                DRM_Unit = u.Unit("")
            DRM_specresp = QTable.read(hdulist[name_hdu])
            DRM_ebounds = QTable.read(hdulist['EBOUNDS'])

        # Define the Matrix
        DRM = np.zeros([len(DRM_specresp), DETCHANS], dtype = np.float64)

        for i, l in enumerate(DRM_specresp):
            if l["N_GRP"]:
                m_start = 0
                for k in range(l["N_GRP"]):
                
                    if np.isscalar(l["N_CHAN"]):
                        f_chan = l["F_CHAN"]    -1 # Necessary only for GBM (?)
                        n_chan = l["N_CHAN"]
                    else:
                        f_chan = l["F_CHAN"][k] -1 # Necessary only for GBM (?)
                        n_chan = l["N_CHAN"][k]

                    DRM[i, f_chan : f_chan+n_chan] = l["MATRIX"][m_start : m_start+n_chan]
                    m_start += n_chan

        energy_true_edges = np.append(DRM_specresp['ENERG_LO'], DRM_specresp['ENERG_HI'][-1])
        energy_true_range = configuration['Energy_Range_True'] * u.Unit(configuration['Energy_Unit'])
        energy_reco_edges = np.append(DRM_ebounds['E_MIN'], DRM_ebounds['E_MAX'][-1])
        energy_reco_range = configuration['Energy_Range_Reco'] * u.Unit(configuration['Energy_Unit'])

        i_start_energy_true, i_start_energy_reco = 0, 0
        i_stop_energy_true, i_stop_energy_reco = -1, -1


        if configuration['Energy_Slice']:
            
            # True
            range_message = f"Original True Energy Range:"
            range_message+= f" [{np.round(energy_true_edges[0].value,3)},"
            range_message+= f" {np.round(energy_true_edges[-1].value,3)}]"
            range_message+= f" {energy_true_edges.unit}. Energy bins: {len(DRM_specresp)}."
            self.log.info(range_message)
            
            dummy_array = energy_true_edges - energy_true_range[0]
            i_start_energy_true = np.argmin(np.abs(dummy_array.value))
            dummy_array = energy_true_edges - energy_true_range[1]
            i_stop_energy_true  = np.argmin(np.abs(dummy_array.value))
            
            self.log.info(f"Slice True Energies between indexes [{i_start_energy_true},{i_stop_energy_true}]")


            # Reco
            range_message = f"Original Reco Energy Range:"
            range_message+= f" [{np.round(energy_reco_edges[0].value,3)},"
            range_message+= f" {np.round(energy_reco_edges[-1].value,3)}]"
            range_message+= f" {energy_reco_edges.unit}. Energy bins: {len(DRM_ebounds)}."
            self.log.info(range_message)

            dummy_array = energy_reco_edges - energy_reco_range[0]
            i_start_energy_reco = np.argmin(np.abs(dummy_array.value))
            dummy_array = energy_reco_edges - energy_reco_range[1]
            i_stop_energy_reco  = np.argmin(np.abs(dummy_array.value))
            self.log.info(f"Slice Reco Energies between indexes [{i_start_energy_reco},{i_stop_energy_reco}]")

            # Slice        
            DRM = DRM[i_start_energy_true:i_stop_energy_true, i_start_energy_reco:i_stop_energy_reco]

        range_message = f"Response Matrix defined at True Energy Range:"
        range_message+= f" [{np.round(energy_true_edges[i_start_energy_true].value,3)},"
        range_message+= f" {np.round(energy_true_edges[  i_stop_energy_true].value,3)}]"
        range_message+= f" {energy_true_edges.unit}. Energy bins: {DRM.shape[0]}."
        self.log.info(range_message)

        range_message = f"Response Matrix defined at Reco Energy Range:"
        range_message+= f" [{np.round(energy_reco_edges[i_start_energy_reco].value,3)},"
        range_message+= f" {np.round(energy_reco_edges[  i_stop_energy_reco].value,3)}]"
        range_message+= f" {energy_reco_edges.unit}. Energy bins: {DRM.shape[1]}."
        self.log.info(range_message)

        self.log.info(f"Response Matrix unit: {DRM_Unit}.\n")
        
        self.detector_response_matrix = DRM * DRM_Unit
        return None
    
    
    
    def compute_effective_area_2D(self,
                                  aeff_array,
                                  configuration,
                                  transient,
                                  output_directory
                                  ):
        """
        Returns the DL3 Effective Area, assumed to be constant with offset.

        Parameters
        ----------
        aeff_array : `astropy.units.Quantity`
            Astropy array of Effective Area values as a function of energy.
        configuration : dict
            Dictionary with the parameters from the YAML file.
        transient : `astropy.table.row.Row`
            Row that contains the selected transient from the catalogue.
        output_directory : str
            Output directory where to save a figure of the Effective Area.

        Returns
        -------
        aeff : `gammapy.irf.EffectiveAreaTable2D`
        """

        self.log.info("Assume Effective Area is constant in the Instrument FoV")

        # Replicate the Effective Area array for each bin of the Offset Axis
        aeff_matrix = np.transpose(aeff_array.value * np.ones((self._axis_offset.nbin, self.axis_energy_true.nbin)))

        aeff = EffectiveAreaTable2D(axes = [self.axis_energy_true, self._axis_offset],
                                    data = aeff_matrix,
                                    unit = aeff_array.unit
                                   )

        self.log.info(aeff)

        # Plot and save
        title = f"Effective Area {configuration['Name_Instrument']}"
        title+= f" {configuration['Name_Detector']}, {transient['name']}."
        fig, axs = plt.subplots(1,2, figsize=(15,5))

        axs[0].step(self.axis_energy_true.center.value, aeff_array, c='C3')
        axs[0].set_xscale('log')
        axs[0].set_yscale('log')
        axs[0].set_xlabel(f"True Energy [{self.axis_energy_true.unit}]", fontsize = 'large')
        axs[0].set_ylabel(f"Effective Area [{aeff_array.unit}]", fontsize = 'large')
        axs[0].set_title(title, fontsize = 'large')
        axs[0].grid()

        aeff.plot(ax = axs[1], add_cbar = True)
        axs[1].set_title(title, fontsize = 'large')
        axs[1].set_xscale('log')

        figure_name = output_directory + "IRF_effective_area"+FIGURE_FORMAT
        self.log.info(f"Saving Effective area plot: {figure_name}\n")
        fig.savefig(figure_name, facecolor = 'white')

        return aeff









class Dataset_Creator_GBM(Dataset_Creator_From_XSPEC):
    """
    Class to create the DL4 that are going to be filled from GBM IRFs.
    
    Parameters
    ----------
    aeff_array : `astropy.units.Quantity`
        Astropy array of Effective Area values as a function of energy.
    """
    def __init__(self, log):
        super().__init__(log)
        return None
    
    def compute_effective_area_array(self, File_arf, Hdu_aeff, configuration):
        """
        Compute the effective area as a 1D array from the Detector Response Matrix.
        """
        self.log.info("Compute Effective Area.")
        self.aeff_array = np.sum(self.detector_response_matrix, axis=1)
        self.log.info(f"Total effective area: {np.sum(self.aeff_array)}.")
        return None
    
    
    
    
class Dataset_Creator_COSI(Dataset_Creator_From_XSPEC):
    """Class to create the DL4 that are going to be filled from COSI IRFs.
    
    Parameters
    ----------
    aeff_array : `astropy.units.Quantity`
        Astropy array of Effective Area values as a function of energy.
    """
    def __init__(self, log):
        super().__init__(log)
        return None
    
    def _read_effective_area_from_ARF(self, name_file_fits, name_hdu, configuration):
        """
        Return the Effective Area as a function of True Energy with their unit.

        Parameters
        ----------
        name_file_fits: str
            Name of a FITS file that must be opened to get the Energy Axis information.
        name_hdu: str
            Name of the HDU that contains the Energy Axis information.
        configuration : dict
            Dictionary with the parameters from the YAML file.

        Returns
        -------
        `astropy.units.Quantity`
        """

        self.log.info(f"Read Effective Area from ARF file: {name_file_fits}")

        # Load Energy Table
        with fits.open(name_file_fits) as hdulist:
            table = QTable.read(hdulist[name_hdu])

        # This assumes SPECRESP is the name of the column with the effective area.
        table['SPECRESP'] = table['SPECRESP'].to(u.cm**2)

        edges = np.append(table['ENERG_LO'].value, table['ENERG_HI'].value[-1]) * table['ENERG_LO'].unit
        energy_range = configuration['Energy_Range_True'] * u.Unit(configuration['Energy_Unit'])
        i_start, i_stop = 0, -1

        if configuration['Energy_Slice']:

            range_message = f"Original Energy Range: [{np.round(table['ENERG_LO'][0].value,3)}, {np.round(table['ENERG_HI'][-1].value,3)}] "
            range_message+= str(table['ENERG_LO'].unit) + f". Energy bins: {len(table['ENERG_LO'])}."
            self.log.info(range_message)

            dummy_array = edges - energy_range[0]
            i_start = np.argmin(np.abs(dummy_array.value))

            dummy_array = edges - energy_range[1]
            i_stop  = np.argmin(np.abs(dummy_array.value))

            table = table[i_start: i_stop]

            self.log.info(f"Slice between indexes [{i_start},{i_stop}].")

        range_message = f"Final Energy range: [{np.round(edges[i_start].value,3)}, {np.round(edges[i_stop].value,3)}] "
        range_message+= f"{edges.unit}. Energy bins: {len(table['SPECRESP'])}.\n"
        self.log.info(range_message)

        return table['SPECRESP']
    
    def compute_effective_area_array(self, File_arf, Hdu_aeff, configuration):
        """
        Compute the effective area as a 1D array from the Detector Response Matrix.
        """
        self.aeff_array = self._read_effective_area_from_ARF(File_arf, Hdu_aeff,configuration)
        self.log.info(f"Total effective area: {np.sum(self.aeff_array)}.")
        return None
    
    
    
    
    
    
    
    
class Dataset_Creator_CTA(Dataset_Creator):
    """Class to create the DL4 that are going to be filled from CTA IRFs."""
    def __init__(self, log):
        super().__init__(log)
        return None

