###########################################
#                                         #
#     CLASSES TO MAKE DL4 FROM IRFS       #
#     SAVED AS FITS FILES FOR XSPEC       #
###########################################

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import QTable
from astropy.time import Time

from gammapy.irf import EDispKernel, EDispKernelMap
from gammapy.maps import MapAxis, RegionGeom, RegionNDMap

from matplotlib.colors import LogNorm, PowerNorm

from simulator_dataset_creator import Dataset_Creator
from scipy.interpolate import interp1d

FIGURE_FORMAT = ".pdf"

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
    
    
    
    
        # def _set_FoV_axes(self):
    #     """
    #     Set the FoV Axes: Offset, Lon, Lat.
    #     """
        
    #     self.log.info(f"Set Instrument FoV Axes.")

    #     # Assumption: The FoV Center is (0,0), FoV_Lon and FoV_Lat are symmetric wrt to 0.0
    #     fov_maximum_offset = 5.0 * u.deg
    #     fov_n_bin = 5
    
    #     # Create Instrument Offset axis
    #     axis_offset = MapAxis.from_bounds(0.0,
    #                                       fov_maximum_offset.value,
    #                                       unit = fov_maximum_offset.unit,
    #                                       nbin = fov_n_bin,
    #                                       name = "offset"
    #                                       )
    #     # Create Instrument FoV_lon axis
    #     axis_fovlon = MapAxis.from_bounds(-fov_maximum_offset.value/2.0,
    #                                       +fov_maximum_offset.value/2.0,
    #                                       unit = fov_maximum_offset.unit,
    #                                       nbin = fov_n_bin,
    #                                       name = "fov_lon"
    #                                       )
    #     # Create Instrument FoV_lat axis
    #     axis_fovlat = MapAxis.from_bounds(-fov_maximum_offset.value/2.0,
    #                                       +fov_maximum_offset.value/2.0,
    #                                       unit = fov_maximum_offset.unit,
    #                                       nbin = fov_n_bin,
    #                                       name = "fov_lat"
    #                                       )
    #     #self.log.info(axis_offset)
    #     #self.log.info(axis_fovlon)
    #     #self.log.info(axis_fovlat)
        
    #     # Set attributes
    #     self._axis_offset = axis_offset
    #     self._axis_fovlon = axis_fovlon
    #     self._axis_fovlat = axis_fovlat
    #     return None


    def _define_pointing(self, transient, frame='fk5', equinox='J2000'):
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
        pointing = SkyCoord(transient['ra'].value,
                            transient['dec'].value,
                            unit = transient['ra'].unit,
                            frame = frame,
                            equinox = equinox
                            )              
        self.log.info(f"Define Pointing Direction: {pointing}.\n")
        
        return pointing
    
    def set_geometry(self, transient, radius=1.0):
        """
        Define the Source Sky Geometry: a circular region in the sky where we place the source
        and we assume that we can receive photons with reconstructed energy in the given axis.

        Parameters
        ----------
        transient: `astropy.table.row.Row`
            Row of the chosen transient.
        radius: float
            Radius of the Circular Geometry in degree.
        """
        self.log.info("Define Source Geometry.")
        
        # Define Pointing Direction
        pointing = self._define_pointing(transient)  # default: frame='fk5', equinox='J2000'
        
        geometry_radius = radius * u.deg
        source_geometry_str = pointing.frame.name + ';circle('
        source_geometry_str+= pointing.to_string().split()[0] + ', '
        source_geometry_str+= pointing.to_string().split()[1] + ', '
        source_geometry_str+= f"{geometry_radius.value})"

        # Set and Log
        self.geometry = RegionGeom.create(source_geometry_str, axes = [self.axis_energy_reco])
        self.log.info(self.geometry)

        return self
    
    
    
    def read_response_matrix_from_RSP(self, name_file_fits, name_hdu, configuration):
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
    
    
    
    # def compute_effective_area_2D(self, aeff_array):
    #     """
    #     Returns the DL3 Effective Area, assumed to be constant with offset.

    #     Parameters
    #     ----------
    #     aeff_array : `astropy.units.Quantity`
    #         Astropy array of Effective Area values as a function of energy.
        
    #     Returns
    #     -------
    #     aeff : `gammapy.irf.EffectiveAreaTable2D`
    #     """

    #     self.log.info("Assume Effective Area is constant in the Instrument FoV")

    #     # Replicate the Effective Area array for each bin of the Offset Axis
    #     aeff_matrix = np.transpose(aeff_array.value * np.ones((self._axis_offset.nbin, self.axis_energy_true.nbin)))

    #     aeff = EffectiveAreaTable2D(axes = [self.axis_energy_true, self._axis_offset],
    #                                 data = aeff_matrix,
    #                                 unit = aeff_array.unit
    #                                )

    #     self.log.info(aeff)

    #     return aeff
    
    
    def compute_effective_area_array(self):
        raise NotImplementedError("Call funciton from a derived class.")
        pass    
    
    def _plot_aeff_array(self, aeff_array, configuration, transient, output_directory):
        """
        Plot the Effective Area as a function of True Energy.
        
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
        """
        title = f"Effective Area {configuration['Name_Instrument']}"
        title+= f" {configuration['Name_Detector']}, {transient['name']}."
        fig, ax = plt.subplots(1, figsize=(15,10))

        ax.step(self.axis_energy_true.center.value, aeff_array, c='C3')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f"True Energy [{self.axis_energy_true.unit}]", fontsize = 'large')
        ax.set_ylabel(f"Effective Area [{aeff_array.unit}]", fontsize = 'large')
        ax.set_title(title, fontsize = 'large')
        ax.grid()

        figure_name = output_directory + "IRF_effective_area"+FIGURE_FORMAT
        self.log.info(f"Saving Effective area plot: {figure_name}\n")
        fig.savefig(figure_name, facecolor = 'white')
        
        return None
    
    def compute_exposure_map(self, aeff_array, livetimes):
        """
        Returns the DL4 Exposure Map = Effective Area * Livetime.
        
        Parameters
        ----------
        aeff_array : `astropy.units.Quantity`
            Astropy array of Effective Area values as a function of energy.
        livetimes : `astropy.units.Quantity`
            Astropy Array of the livetimes, needed to set the Exposure Map.
            
        Returns
        -------
        map_exposure : `gammapy.maps.RegionNDMap`
            Map of the Exposure. Axes: 'lon', 'lat', 'energy_true'. Shape: 1x1xlen(aeff_array).
        map_exposure_edisp : `gammapy.maps.RegionNDMap`
            Map of the Exposure. Axes: 'lon', 'lat', 'energy', 'energy_true'. Shape: 1x1x1xlen(aeff_array).
        """
        self.log.info("Compute exposure map.")
        
        expo_array = aeff_array * livetimes[0]
        
        geom_exposure = self.geometry.drop('energy').to_cube([self.axis_energy_true])
        
        map_exposure = RegionNDMap.from_geom(geom = geom_exposure,
                                             data = np.reshape(expo_array.value, geom_exposure.data_shape),
                                             unit = expo_array.unit
                                             )
        
        geom_exposure_edisp = self.geometry.squash('energy').to_cube([self.axis_energy_true])
        
        map_exposure_edisp = RegionNDMap.from_geom(geom = geom_exposure_edisp,
                                                   data = np.reshape(expo_array.value, geom_exposure_edisp.data_shape),
                                                   unit = expo_array.unit
                                                   )
        
        
        return map_exposure, map_exposure_edisp   
    
    def compute_energy_dispersion_map(self):
        raise NotImplementedError("Call funciton from a derived class.")
        pass
    
    def _plot_energy_dispersion(self, edisp, configuration, transient, output_directory):
        """
        Plot and save the Energy Dispersion Matrix.
        
        Parameters
        ----------
        edisp : `gammapy.irf.EDispKernelMap `
            Energy Dispersion Matrix as a DL4.
        configuration : dict
            Dictionary with the parameters from the YAML file.
        transient : `astropy.table.row.Row`
            Row that contains the selected transient from the catalogue.
        output_directory : str
            Output directory where to save a figure of the Effective Area.
        """
        # Prepare Grid
        X, Y = np.meshgrid(self.axis_energy_true.center.value, self.axis_energy_reco.center.value)

        # Copy Data with Masking
        Z = np.ma.masked_where(edisp.edisp_map.data.T[0][0] <= 0, edisp.edisp_map.data.T[0][0])

        # Plot
        fig, ax = plt.subplots(1, figsize=(9,5))

        # Define Levels
        levs = np.linspace(np.floor(np.power(Z.min(),0.3)),
                           np.ceil( np.power(Z.max(),0.3)),
                           num = 50
                          )
        levs = np.power(levs, 1.0/0.3)

        # Plot Data
        cs = ax.contourf(X, Y, Z, levs, norm = PowerNorm(gamma=0.3), cmap = 'plasma')
        ax.contour(X, Y, Z, levs, norm = PowerNorm(gamma=0.3), colors='white', alpha=0.05)
        cbar = fig.colorbar(cs)

        # Labels
        ax.set_facecolor('k')
        ax.set_xscale('log')
        ax.set_yscale('log')
        cbar.set_label('Redistribution Probability', fontsize = 'large')
        ax.set_xlabel(f"True Energy [{self.axis_energy_true.unit}]", fontsize = 'large')
        ax.set_ylabel(f"Energy [{self.axis_energy_reco.unit}]", fontsize = 'large')

        title = f"Energy Dispersion Matrix {configuration['Name_Instrument']}"
        title+= f" {configuration['Name_Detector']}, {transient['name']}."
        ax.set_title(title, fontsize = 'large')

        # Save figure
        figure_name = output_directory+"IRF_energy_dispersion_matrix"+FIGURE_FORMAT
        self.log.info(f"Saving Energy Dispersion Matrix plot : {figure_name}\n")
        fig.savefig(figure_name, facecolor = 'white')

        return None
    
    
    def read_background_spectrum(self, configuration, hdu_ebounds="EBOUNDS", hdu_spectrum="SPECTRUM"):
        """
        Return the background Spectral Model.

        Parameters
        ----------
        configuration : dict
            Dictionary with the parameters from the YAML file.
        hdu_ebounds : str
            Name of the HDU that contains the Reconstructed Energy Axis.
        hdu_spectrum : str
            Name of the HDU that contains the Spectral Model.

        Returns
        -------
        `astropy.units.Quantity`
        """

        self.log.info(f"Read the Background Spectral Model from BAK file: {configuration['Input_bak']}")

        # Define Background Tables
        with fits.open(configuration['Input_bak']) as hdulist:
            table_spectrum = QTable.read(hdulist[hdu_spectrum])
            try:
                table_ebounds = QTable.read(hdulist[hdu_ebounds])
            except:
                self.log.warning(f"Energy Axis not found in BAK. Try to look in {configuration['Input_rmf']}")
                with fits.open(configuration['Input_rmf']) as hdulist_rmf:
                    table_ebounds = QTable.read(hdulist_rmf[hdu_ebounds])

            if 'RATE' in table_spectrum.colnames:
                if table_spectrum['RATE'].unit is None:
                    self.log.warning(f"RATE unit not found. Using 1/{configuration['Time_Unit']}")
                    table_spectrum['RATE'] = table_spectrum['RATE'] / u.Unit(configuration['Time_Unit'])
            elif 'COUNTS' in table_spectrum.colnames:
                self.log.warning(f"Column RATE not found. Using COUNTS / (EXPOSURE [s])")
                Integration_time = hdulist[hdu_spectrum].header['EXPOSURE']*u.s
                self.log.info(f"EXPOSURE (s) = {Integration_time.value}")
            else:
                self.log.error(f"We could not find column RATE nor COUNTS")
                exit()

            # Check Units
            if table_ebounds['E_MIN'].unit is None:
                self.log.warning(f"Energy unit not found. Using {configuration['Energy_Unit']}")
                table_ebounds['E_MIN'] = table_ebounds['E_MIN'] * u.Unit(configuration['Energy_Unit'])
                table_ebounds['E_MAX'] = table_ebounds['E_MAX'] * u.Unit(configuration['Energy_Unit'])

        # Define a new Table with the Columns we need.
        table = QTable()

        table['E_MIN'] = table_ebounds['E_MIN']
        table['E_MAX'] = table_ebounds['E_MAX']

        if 'RATE' in table_spectrum.colnames:
            table['RATE'] = table_spectrum['RATE']
        elif 'COUNTS' in table_spectrum.colnames:
            table['RATE'] = table_spectrum['COUNTS'].value / Integration_time

        # Define the column of the Background Spectral Model    
        table['BKG_MOD'] = table['RATE'] / (table['E_MAX']-table['E_MIN'])

        energy_edges = np.append(table['E_MIN'], table['E_MAX'][-1])
        energy_range = configuration['Energy_Range_Reco'] * u.Unit(configuration['Energy_Unit'])
        i_start, i_stop = 0, -1

        if configuration['Energy_Slice']:
            range_message = f"Original Reco Energy Range:"
            range_message+= f" [{np.round(energy_edges[0].value,3)},"
            range_message+= f" {np.round(energy_edges[-1].value,3)}]"
            range_message+= f" {energy_edges.unit}. Energy bins: {len(table)}."
            self.log.info(range_message)

            dummy_array = energy_edges - energy_range[0]
            i_start = np.argmin(np.abs(dummy_array.value))

            dummy_array = energy_edges - energy_range[1]
            i_stop = np.argmin(np.abs(dummy_array.value))

            table = table[i_start: i_stop]

            self.log.info(f"Slice Energies between indexes [{i_start},{i_stop}]")

        range_message = f"Background Spectral Model defined at Reco Energy Range:"
        range_message+= f" [{np.round(energy_edges[i_start].value,3)},"
        range_message+= f" {np.round(energy_edges[i_stop].value,3)}]"
        range_message+= f" {energy_edges.unit}. Energy bins: {len(table)}."
        self.log.info(range_message)

        self.log.info(f"Background Spectral Model defined with unit {table['BKG_MOD'].unit}\n")

        return table['BKG_MOD']
    
    
    
    def compute_background_map(self):
        raise NotImplementedError("Call funciton from a derived class.")
        pass
    
    
    
    def _plot_background_model(self, bak_model, energy_axis, configuration, transient, output_directory):
        """
        Plot the Background Spectrum (cts/keV/s vs keV).
        
        Parameters
        ----------
        bak_model : `astropy.units.Quantity`
            Background Model as count rate spectrum.
        energy_Axis : `gammapy.maps.MapAxis`
            Reconstructed Energy Axis for Background.
        configuration : dict
            Dictionary with the parameters from the YAML file.
        transient : `astropy.table.row.Row`
            Row that contains the selected transient from the catalogue.
        output_directory : str
            Output directory where to save a figure of the Effective Area.
        """
    
    
        # Plot and save
        fig, ax = plt.subplots(1, figsize=(7,5))

        ax.step(energy_axis.center.value, bak_model, color = 'C3')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(f"Energy [{energy_axis.unit.to_string()}]", fontsize = 'large')
        ax.set_ylabel(f"Background rate [{bak_model.unit.to_string()}]", fontsize = 'large')
        title = f"Background Spectral Model {configuration['Name_Instrument']}"
        title+= f" {configuration['Name_Detector']}, {transient['name']}."
        ax.set_title(title, fontsize = 'large')
        ax.grid()

        # Save figure
        figure_name = output_directory+"IRF_background_spectrum"+FIGURE_FORMAT
        self.log.info(f"Saving Background Spectral Model plot : {figure_name}\n")
        fig.savefig(figure_name, facecolor = 'white')
        
        return None











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
    
    def compute_effective_area_array(self, File_arf, Hdu_aeff, configuration, transient, output_directory):
        """
        Compute the effective area as a 1D array from the Detector Response Matrix, then plot it.
        
        Parameters:
        -----------
        configuration : dict
            Dictionary with the parameters from the YAML file.
        transient : `astropy.table.row.Row`
            Row that contains the selected transient from the catalogue.
        output_directory : str
            Output directory where to save a figure of the Effective Area.
        """
        self.log.info("Compute Effective Area.")
        self.aeff_array = np.sum(self.detector_response_matrix, axis=1)
        self.log.info(f"Total effective area: {np.sum(self.aeff_array)}.")
        
        # Plot and Save
        self._plot_aeff_array(self.aeff_array, configuration, transient, output_directory)
        return None
    
    
    
    def compute_energy_dispersion_map(self, map_exposure_edisp, configuration, transient, output_directory):
        """
        Returns the Energy Dispersion Kernel Map.
        Set the Exposure Map of the Energy Dispersion Matrix.

        Parameters
        ----------
        map_exposure_edisp : `gammapy.maps.RegionNDMap`
            Exposure Map with 4 dimensions: 'lon', 'lat', 'energy', 'energy_true'. Shape 1x1x1xlen(Effective Area).
        configuration : dict
            Dictionary with the parameters from the YAML file.
        transient : `astropy.table.row.Row`
            Row that contains the selected transient from the catalogue.
        output_directory : str
            Output directory where to save a figure of the Effective Area.

        Returns
        -------
        edisp : `gammapy.irf.EDispKernelMap`
            Energy Dispersion Matrix as a DL4 reduced IRF with an Exposure Map.
        """

        self.log.info("Compute Energy Dispersion Matrix")

        edisp = EDispKernel(axes = [self.axis_energy_true, self.axis_energy_reco],
                            data = self.detector_response_matrix.value
                            )
        edisp = EDispKernelMap.from_edisp_kernel(edisp, geom = self.geometry)
        
        # Normalize GBM Data
        DRM = np.zeros(np.shape(edisp.edisp_map.data.T[0][0].T))
        for i, r in enumerate(edisp.edisp_map.data.T[0][0].T):
            norm_row = np.sum(r)
            if norm_row != 0.0:
                DRM[i] = r / norm_row

        DRM = np.reshape(DRM, np.shape(edisp.edisp_map.data))
        edisp.edisp_map.data = DRM
        self.log.warning(f"Normalization to 1 applied: assuming no lost photons.")

        # Set exposure map
        edisp.exposure_map = map_exposure_edisp
        
        # Plot
        self._plot_energy_dispersion(edisp, configuration, transient, output_directory)

        return edisp
    
    
    def compute_background_map(self, bak_model, livetimes, configuration, transient, output_directory):
        """
        Compute Background Map
        
        Parameters
        ----------
        bak_model : `astropy.units.Quantity`
            Background Model as count rate spectrum.
        livetimes : `astropy.units.Quantity`
            Astropy Array of the livetimes, needed to set the Exposure Map.
        
        Returns
        -------
        map : `gammapy.maps.RegionNDMap`
            Background Map. Axes: 'lon', 'lat', 'energy'.
        """
        
        geom = self.geometry.drop('energy').to_cube([self.axis_energy_reco])
        
        # This background is defined over a background axis defined in file BAK.
        self.log.warning("GBM wants another Reco Energy Axis for the Background, taken from BAK, not RSP.")
        axis_energy_reco_bkg = self._define_energy_axis(configuration['Input_bak'],
                                                        "EBOUNDS",
                                                        configuration,
                                                        energy_is_true = False
                                                        )
        
        # We need to interpolate at energy edges taken from RSP.
        f = interp1d(axis_energy_reco_bkg.center.value, bak_model.value, kind='cubic')
        
        energies_for_interpolation = self.axis_energy_reco.center.to(axis_energy_reco_bkg.center.unit)
        bak_model_interp = f(energies_for_interpolation.value) * bak_model.unit
        
        # Plot
        self._plot_background_model(bak_model_interp, self.axis_energy_reco, configuration, transient, output_directory)
        
        # Tha Background Map requires absolute counts
        bak_counts = (bak_model_interp * self.axis_energy_reco.bin_width) * livetimes[0]
        bak_counts.to("")
        
        map = RegionNDMap.from_geom(geom = geom,
                                    data = np.reshape(bak_counts.value, geom.data_shape),
                                    unit = bak_model.unit
                                    )
        
        return map
        
    
    
    
    
    
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
    
    def compute_effective_area_array(self, File_arf, Hdu_aeff, configuration, transient, output_directory):
        """
        Compute the effective area as a 1D array from the Detector Response Matrix, then plot it.
        
        Parameters:
        -----------
        File_arf : str
            Name of the FITS file with Effective Area.
        Hdu_aeff : str
            name of the HDU with Effective Area.
        configuration : dict
            Dictionary with the parameters from the YAML file.
        transient : `astropy.table.row.Row`
            Row that contains the selected transient from the catalogue.
        output_directory : str
            Output directory where to save a figure of the Effective Area.
        """
        self.aeff_array = self._read_effective_area_from_ARF(File_arf, Hdu_aeff,configuration)
        self.log.info(f"Total effective area: {np.sum(self.aeff_array)}.")
        
        # Plot and Save
        self._plot_aeff_array(self.aeff_array, configuration, transient, output_directory)
        return None
    
    
    def compute_energy_dispersion_map(self, map_exposure_edisp, configuration, transient, output_directory):
        """
        Returns the Energy Dispersion Kernel Map.
        Set the Exposure Map of the Energy Dispersion Matrix.

        Parameters
        ----------
        map_exposure_edisp : `gammapy.maps.RegionNDMap`
            Exposure Map with 4 dimensions: 'lon', 'lat', 'energy', 'energy_true'. Shape 1x1x1xlen(Effective Area).
        configuration : dict
            Dictionary with the parameters from the YAML file.
        transient : `astropy.table.row.Row`
            Row that contains the selected transient from the catalogue.
        output_directory : str
            Output directory where to save a figure of the Effective Area.

        Returns
        -------
        edisp : `gammapy.irf.EDispKernelMap`
            Energy Dispersion Matrix as a DL4 reduced IRF with an Exposure Map.
        """

        self.log.info("Compute Energy Dispersion Matrix")

        edisp = EDispKernel(axes = [self.axis_energy_true, self.axis_energy_reco],
                            data = self.detector_response_matrix.value
                            )
        edisp = EDispKernelMap.from_edisp_kernel(edisp, geom = self.geometry)

        # Set exposure map
        edisp.exposure_map = map_exposure_edisp
        
        # Plot
        self._plot_energy_dispersion(edisp, configuration, transient, output_directory)

        return edisp
    
    
    
    def compute_background_map(self, bak_model, livetimes, configuration, transient, output_directory):
        """
        Compute Background Map
        
        Parameters
        ----------
        bak_model : `astropy.units.Quantity`
            Background Model as count rate spectrum.
        livetimes : `astropy.units.Quantity`
            Astropy Array of the livetimes, needed to set the Exposure Map.
        
        Returns
        -------
        map : `gammapy.maps.RegionNDMap`
            Background Map. Axes: 'lon', 'lat', 'energy'.
        """
        
        geom = self.geometry.drop('energy').to_cube([self.axis_energy_reco])
        
        self._plot_background_model(bak_model, self.axis_energy_reco, configuration, transient, output_directory)
        
        # Tha Background Map requires absolute counts
        
        bak_counts = (bak_model * self.axis_energy_reco.bin_width) * livetimes[0]
        bak_counts.to("")
        
        map = RegionNDMap.from_geom(geom = geom,
                                    data = np.reshape(bak_counts.value, geom.data_shape),
                                    unit = bak_model.unit
                                    )
        
        return map
        
        
    
    
    
    
# data_bkg = np.transpose(bak_model.value * np.ones((axis_fovlat.nbin,axis_fovlon.nbin,axis_energy_reco_bkg.nbin)))

# bkg = Background3D(axes = [axis_energy_reco_bkg, axis_fovlon, axis_fovlat],
#                    data = data_bkg,
#                    unit = bak_model.unit,
#                    )