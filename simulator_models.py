############ Not-gammapy imports

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines

import astropy.units as u
from astropy.units import Quantity
from astropy.time import Time
from astropy.table import Table, QTable

from scipy.special import erfinv

# Gammapy imports
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    SkyModel,
    LightCurveTemplateTemporalModel,
    GaussianTemporalModel,
    TemplateSpectralModel
)


FIGURE_FORMAT = ".pdf"

class GBM_Band_Spectral_Function:
    def __init__(self, amplitude, alpha, beta, Epeak, Epiv=100*u.keV):
        self.amplitude = amplitude
        self.alpha     = alpha
        self.beta      = beta
        self.Epeak     = Epeak
        self.Epiv      = Epiv
        self.Estar     = ((self.alpha-self.beta)/(2.0+self.alpha))*self.Epeak

    def _eval_lower(self, energies):
        ene = energies/self.Epiv
        enb = energies/self.Epeak
        return self.amplitude*np.power(ene.to(""), self.alpha)*np.exp(-(2+self.alpha)*enb.to(""))

    def _eval_upper(self, energies):
        ene = energies/self.Epiv
        epp = self.Epeak/self.Epiv
        amp = self.amplitude*np.power(epp.to("")*(self.alpha-self.beta)/(2.0+self.alpha), self.alpha-self.beta)*np.exp(self.beta-self.alpha)
        return amp*np.power(ene.to(""), self.beta)
    
    def evaluate(self, energies):
        """
        Evaluate Band function. It is a step function:
        F(E)= A * [E/Epiv]^alpha*exp[-(2+alpha)*E/Epeak]    if E < Estar
        F(E)= A*[((alpha-beta)/(2+alpha))*(Epeak/Epiv)]^(alpha-beta)*exp(beta-alpha) * (E/Epiv)^beta    if E > Estar
        Estar = ((alpha-beta)/(2+alpha))*Epeak
        """
        band_low_energies  = self._eval_lower(energies[energies<self.Estar])
        band_high_energies = self._eval_lower(energies[energies>self.Estar])
        return np.append(band_low_energies, band_high_energies)

    





def Define_Template_Temporal_Model(trigger_time, configuration, logger):
    """
    Return the Temporal Model and the correction factor for spectral normalization.
    The correct spectral normalization is the fit normalization / correction_factor
    because the fit normalization is a time-integrated average.

    Parameters
    ----------
    trigger_time : `astropy.time.Time`
        Trigger time.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    logger : `logging.Logger`
        Logger from main.

    Returns
    -------
    Temporal_Model : `gammapy.modeling.models.temporal.TemporalModel`
        A Temporal Model for the SkyModel.
    correction_factor : float 
        Correction factor for Spectral Normalization due to time-average of the spectral fit.       
    """

    
    logger.info(f"Read Temporal Model from: {configuration['Light_Curve_Input']}")
    LC_empirical = QTable.read(configuration['Light_Curve_Input'], format = 'fits')

    if LC_empirical['time'].unit in [None, u.Unit("")]:
        logger.warning(f"Time Unit of Template Light Curve not found. Using {configuration['Time_Unit']}")
        LC_empirical['time'] = LC_empirical['time'] * u.Unit(configuration['Time_Unit'])

    Temporal_Model_Table_Metadata = {'MJDREFI' : int(np.modf(trigger_time.mjd)[1]),
                                     'MJDREFF' :     np.modf(trigger_time.mjd)[0],
                                     'TIMEUNIT': LC_empirical['time'].unit.to_string(),
                                     'TIMESYS' : trigger_time.scale
                                    }

    Temporal_Model_Table = Table(meta = Temporal_Model_Table_Metadata)
    Temporal_Model_Table['TIME'] = LC_empirical['time']
    Temporal_Model_Table['NORM'] = LC_empirical['norm']

    Temporal_Model = LightCurveTemplateTemporalModel(Temporal_Model_Table)

    logger.info(Temporal_Model)

    # Evaluate correction factor
    spectral_fit_time_range = configuration['Specfit_time_range']*u.Unit(configuration['Time_Unit'])

    correction_factor = [trigger_time.mjd + spectral_fit_time_range[0].to("day").value,
                         trigger_time.mjd + spectral_fit_time_range[1].to("day").value
                        ]
    correction_factor = Time(correction_factor, format = 'mjd')
    
    correction_factor = Temporal_Model.integral(correction_factor[0], correction_factor[1])
    correction_factor = 1.0 / correction_factor.to("")

    logger.info(f"Spectral fit computed between (wrt Trigger Time): {spectral_fit_time_range}.")
    logger.info(f"Correction factor for time-integration: {np.round(correction_factor,3)}.\n")

    return Temporal_Model, correction_factor


def Define_Gaussian_Pulse(trigger_time, configuration, transient, logger):
    """
    Return the Temporal Model and the correction factor for spectral normalization.
    The correct spectral normalization is the fit normalization / correction_factor
    because the fit normalization is a time-integrated average.
    We assume the model is a Gaussian pulse.
    The average is the temporal bin with highest flux in the GBM spectral analysis.
    Average = pflx_spectrum_stop - pflx_spectrum_start
    Sigma = T90 / (2 * np.sqrt(2)* erfinv(0.90)).

    Parameters
    ----------
    trigger_time : `astropy.time.Time`
        Trigger time.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    logger : `logging.Logger`
        Logger from main.

    Returns
    -------
    Temporal_Model : `gammapy.modeling.models.GaussianTemporalModel`
        A Temporal Model for the SkyModel.
    correction_factor : float 
        Correction factor for Spectral Normalization due to time-average of the spectral fit.       
    """

    # Set Average
    pflx_spectrum_start = transient['pflx_spectrum_start']
    pflx_spectrum_stop  = transient['pflx_spectrum_stop']
    logger.info(f"Highest flux recorded in catalog between [{pflx_spectrum_start},{pflx_spectrum_stop}] from trigger time.")
    
    average = trigger_time.mjd + (pflx_spectrum_stop.to("day").value + pflx_spectrum_start.to("day").value)/2.0
    average = Time(average, format = 'mjd')
    logger.info(f"Center Gaussian Pulse at MJD:{average}, i.e. {average.to_datetime()}")

    avg = average-trigger_time
    logger.info(f"Center Gaussian Pulse at {avg.to(u.s)} wrt Trigger Time.")

    t90 = transient['t90']
    logger.info(f"T90={t90}.")
    sigma = t90 / (2*np.sqrt(2)*erfinv(0.90))
    logger.info(f"Gaussian Pulse sigma={np.round(sigma,4)}.")

    Temporal_Model = GaussianTemporalModel(t_ref = average.mjd * u.d, sigma = sigma)
    logger.info(Temporal_Model)

    # Evaluate correction factor

    tstart_label = configuration['Spectral_Model_Type']+"_spectrum_start"
    tstop_label  = configuration['Spectral_Model_Type']+"_spectrum_stop"

    spectral_fit_time_range = Quantity([transient[tstart_label].unmasked, transient[tstop_label].unmasked])

    correction_factor = [trigger_time.mjd + spectral_fit_time_range[0].to("day").value,
                         trigger_time.mjd + spectral_fit_time_range[1].to("day").value
                        ]
    correction_factor = Time(correction_factor, format = 'mjd')
    
    correction_factor = Temporal_Model.integral(correction_factor[0], correction_factor[1])
    correction_factor = 1.0 / correction_factor.to("")

    logger.info(f"Spectral fit computed between (wrt Trigger Time): {spectral_fit_time_range}.")
    logger.info(f"Correction factor for time-integration: {np.round(correction_factor,3)}.\n")

    return Temporal_Model, correction_factor



def Define_Spectral_Model(configuration, transient, logger, correction_factor, energy_axis):
    """
    Return the Spectral Model.

    Parameters
    ----------
    configuration : dict
        Dictionary with the parameters from the YAML file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    logger : `logging.Logger`
        Logger from main.
    correction_factor : float 
        Correction factor for Spectral Normalization due to time-average of the spectral fit.
    energy_axis: `gammapy.maps.MapAxis`
        True Energy Axis

    Returns
    -------
    Spectral_Model : `gammapy.modeling.models.spectral.SpectralModel`
        A Spectral Model for the SkyModel.       
    """

    logger.info(f"Spectral Model Type: {configuration['Spectral_Model_Type']}")
    logger.info(f"Spectral Model Name: {configuration['Spectral_Model_Name']}")
    label = configuration['Spectral_Model_Type'] +'_' + configuration['Spectral_Model_Name'] + '_'

    if configuration['Spectral_Model_Name'] == "comp":
        
        logger.info(f"Transient Amplitude   : {transient[label+'ampl']}")
        logger.info(f"Transient Index       : {transient[label+'index']}")
        logger.info(f"Transient Peak Energy : {transient[label+'epeak']}")
        logger.info(f"Transient Pivot Energy: {transient[label+'pivot']}")
        logger.info(f"Transient {label} Flux: {transient[label+'phtflux']}")

        amplitude = transient[label+'ampl'] * correction_factor
        index = - transient[label+'index']
        lambda_ = (2.0 + transient[label+'index']) / transient[label+'epeak']
        reference = transient[label+'pivot']

        logger.info("Convert parameters of Comptonized GBM Model into parameters of Gammapy Model")

        Spectral_Model = ExpCutoffPowerLawSpectralModel(amplitude = amplitude,
                                                index     = index,
                                                lambda_   = lambda_,
                                                reference = reference
                                               )
        logger.info(Spectral_Model)


    elif configuration['Spectral_Model_Name'] == "band":

        logger.info(f"Transient Amplitude   : {transient[label+'ampl']}")
        logger.info(f"Transient Alpha Index : {transient[label+'alpha']}")
        logger.info(f"Transient Beta Index  : {transient[label+'beta']}")
        logger.info(f"Transient Peak Energy : {transient[label+'epeak']}")
        logger.info(f"Transient {label} Flux: {transient[label+'phtflux']}")

        ampli = transient[f"{label}ampl"] * correction_factor
        alpha = transient[f"{label}alpha"]
        beta  = transient[f"{label}beta"]
        epeak = transient[f"{label}epeak"]

        Band = GBM_Band_Spectral_Function(amplitude=ampli, alpha=alpha, beta=beta, Epeak=epeak)

        if configuration['Energy_Interpolation'] == 'lin':
            energies = np.linspace(energy_axis.edges[0], energy_axis.edges[-1], num=1000)
        elif configuration['Energy_Interpolation'] == 'log':
            energies = np.linspace(np.log10(energy_axis.edges[0].value),
                                   np.log10(energy_axis.edges[-1].value),
                                   num = 1000
                                  )
            energies = np.power(10,energies)
            energies = energies * energy_axis.unit

        values = Band.evaluate(energies)

        Spectral_Model = TemplateSpectralModel(energy=energies, values=values)

        logger.info(Spectral_Model)

    else:
        raise NotImplementedError("Only Comptonized and Band Spectral Model have been implemented.")


    return Spectral_Model
    










def Plot_Sky_Model(sky_model, configuration, transient, logger, energy_axis, trigger_time,
                    correction_factor, output_directory):
    """
    Plot and save the Sky Model.

    Parameters
    ----------
    sky_model : `gammapy.modeling.models.SkyModel`
        Sky Model to plot.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    logger : `logging.Logger`
        Logger from main.
    correction_factor : float 
        Correction factor for Spectral Normalization due to time-average of the spectral fit.
    energy_axis: `gammapy.maps.MapAxis`
        True Energy Axis
    trigger_time : `astropy.time.Time`
        Trigger time.
    correction_factor : float 
        Correction factor for Spectral Normalization due to time-average of the spectral fit.
    output_directory : str
        Directory used to write and save output.

    Returns
    -------
    None      
    """

    fig, axs = plt.subplots(1,2, figsize = (15,5) )

    # Spectral Model
    if configuration['Energy_Interpolation'] == 'lin':
        energies_to_plot = np.linspace(energy_axis.edges[0], energy_axis.edges[-1], num=100)
    elif configuration['Energy_Interpolation'] == 'log':
        energies_to_plot = np.linspace(np.log10(energy_axis.edges[0].value),
                                       np.log10(energy_axis.edges[-1].value),
                                       num = 100
                                      )
        energies_to_plot = np.power(10,energies_to_plot)
        energies_to_plot = energies_to_plot * energy_axis.unit

    dnde_to_plot, dnde_errors = sky_model.spectral_model.evaluate_error(energies_to_plot)

    axs[0].plot(energies_to_plot.value, dnde_to_plot.value)

    title = f"Spectral Model {transient['name']}, {configuration['Name_Instrument']} {configuration['Name_Detector']}."
    axs[0].set_title(title, fontsize = 'large')
    axs[0].set_xlabel("True Energy ["+energies_to_plot.unit.to_string()+"]", fontsize = 'large')
    axs[0].set_ylabel("dnde ["+dnde_to_plot.unit.to_string()+"]", fontsize = 'large')
    axs[0].set_xscale('log')
    axs[0].set_yscale('log')
    axs[0].grid(which="both")

    # Temporal Model

    obs_time_start = configuration['Observation_Time_Start'] * u.Unit(configuration['Time_Unit'])
    obs_time_stop  = configuration['Observation_Time_Stop' ] * u.Unit(configuration['Time_Unit'])

    times_to_plot = np.linspace(obs_time_start, obs_time_stop, num=300)
    
    if isinstance(sky_model.temporal_model, GaussianTemporalModel):
        tref = (sky_model.temporal_model.t_ref.value - trigger_time.mjd) * u.d
        tref = tref.to("s")
        sigma = sky_model.temporal_model.sigma
        norms_to_plot = sky_model.temporal_model.evaluate(times_to_plot.value, tref.value, sigma.value)
    elif isinstance(sky_model.temporal_model, LightCurveTemplateTemporalModel):
        norms_to_plot = sky_model.temporal_model.evaluate(trigger_time.mjd + times_to_plot.to("day").value)
    else:
        raise NotImplementedError("Only Template Light Curve and Gaussian Pulse implemented")

    axs[1].plot(times_to_plot.value, norms_to_plot)
    
    title = f"Temporal Model {transient['name']}, {configuration['Name_Instrument']} {configuration['Name_Detector']}."
    axs[1].set_title(title, fontsize = 'large')
    axs[1].grid(which="both")
    axs[1].set_xlabel(f"Time since Trigger [{configuration['Time_Unit']}]", fontsize = 'large')
    axs[1].set_ylabel("Temporal Model Norm [A.U.]", fontsize = 'large')

    # Legends
    labels = []
    handles = []
    for par in sky_model.spectral_model.parameters:
        labels.append(par.name+f": {par.value:.2e} [" +par.unit.to_string()+"]")
        handles.append(mlines.Line2D([], []))

    axs[0].legend(handles, labels)

    handles_t = [mlines.Line2D([], []),mlines.Line2D([], [])]
    labels_t = ["Trigger: "+trigger_time.value,f"Correction Factor: {correction_factor:.2f}"]

    if isinstance(sky_model.temporal_model, GaussianTemporalModel):
        labels_t.append(sky_model.temporal_model.t_ref.name+f": {tref.value:.2e} [{tref.unit.to_string()}]")
        labels_t.append(sky_model.temporal_model.sigma.name+f": {sigma.value:.2e} [{sigma.unit.to_string()}]")
        handles_t.append(mlines.Line2D([], []))
        handles_t.append(mlines.Line2D([], []))


    axs[1].legend(handles_t, labels_t)

    # Save Figure
    figure_name = output_directory + "Models"+FIGURE_FORMAT
    logger.info(f"Saving Model plot: {figure_name}\n")
    fig.savefig(figure_name, facecolor = 'white')

    return None


