###########################################
#                                         #
#                                         #
#                                         #
###########################################

############ Not-gammapy imports

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import ticker, cm
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm, PowerNorm

import astropy.units as u
from astropy.units import Quantity
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table, QTable, hstack
from astropy.io import fits

from scipy.special import erfinv


from time import time
from tqdm import tqdm

# Gammapy imports
from gammapy.data import Observation
from gammapy.datasets import SpectrumDataset, Datasets
from gammapy.irf import (
    EffectiveAreaTable2D,
    Background2D, Background3D,
    EnergyDispersion2D, EDispKernel, EDispKernelMap,
)
from gammapy.makers import SpectrumDatasetMaker
from gammapy.maps import MapAxis, RegionGeom, TimeMapAxis
from gammapy.modeling import Fit
from gammapy.modeling.models import (
    ExpCutoffPowerLawSpectralModel,
    SkyModel,
    LightCurveTemplateTemporalModel,
    GaussianTemporalModel
)
from gammapy.utils.random import get_random_state

FIGURE_FORMAT = ".pdf"

############ Functions




def Write_Light_Curves(Time_Centroids, List_of_Datasets, logger, output_directory):
    """
    Write simulated light curves in fits file.

    Parameters
    ----------
    Time_Centroids : `astropy.units.quantity.Quantity`
        Array of Time Centroid of each observation of the Light Curves.
    List_of_Datasets : list
        List of `gammapy.datasets.core.Datasets`, simulated Datasets.
    logger : `logging.Logger`
        Logger from main.
    output directory : str
        Directory used to write and save output. 
    """
    hdu_list = []
    hdu_list.append(fits.PrimaryHDU())

    # qtable = QTable(datasets_generic.info_table())
    # qtable['Time_Centroids'] = Time_Centroids
    # hdu_list.append(fits.table_to_hdu(qtable))

    for i_LC in tqdm(range(len(List_of_Datasets)), desc='Writing Light Curves'):
        qtable = QTable(List_of_Datasets[i_LC].info_table())
        qtable['Time_Centroids'] = Time_Centroids
        hdu_list.append(fits.table_to_hdu(qtable))

    hdu_list = fits.HDUList(hdu_list)

    logger.info(f"Write Lightcurves: {output_directory}lightcurves.fits")
    hdu_list.writeto(output_directory+"lightcurves.fits", overwrite=True)

    return None


def Print_Light_Curves(Time_Centroids, List_of_Datasets, output_directory, configuration, transient):
    """
    Print plots of the simulated light curves.

    Parameters
    ----------
    Time_Centroids : `astropy.units.quantity.Quantity`
        Array of Time Centroid of each observation of the Light Curves.
    List_of_Datasets : list
        List of `gammapy.datasets.core.Datasets`, simulated Datasets.
    output directory : str
        Directory used to write and save output.
    configuration : dict
        Dictionary with the parameters from the YAML file.
    transient : `astropy.table.row.Row`
        Row that contains the selected transient from the catalogue.
    """

    # Simulated Light Curves
    figure_name = f"{output_directory}light_curves"+FIGURE_FORMAT
    pp = PdfPages(figure_name)

    for i_LC in tqdm(range(len(List_of_Datasets)), desc='Print Light Curve Plots'):
    
        fig, ax = plt.subplots(1, figsize = (15, 5), constrained_layout=True )
        title = f"Simulated Light {i_LC+1}/{len(List_of_Datasets)} Curve from {configuration['Name_Instrument']}"
        title+= f" {configuration['Name_Detector']}, {transient['name']}. "
        title+= f"Energy range [{configuration['Energy_Range_Reco'][0]}, {configuration['Energy_Range_Reco'][1]}] {u.Unit(configuration['Energy_Unit'])}."

        counts = List_of_Datasets[i_LC].info_table()['counts']
        uncertainties = np.sqrt(counts)
        widths = List_of_Datasets[i_LC].info_table()['livetime'].value

        ax.bar(Time_Centroids.value,
            height = 2.0 * uncertainties,
            width = widths,
            bottom = counts - uncertainties,
            alpha=0.4, color='grey'
           )
        ax.step(Time_Centroids.value, counts,
            label = f"Simulated counts", color = 'C0', where = 'mid'
            )
        ax.set_xlabel('Time [s]', fontsize = 'large')
        ax.set_ylabel('Counts', fontsize = 'large')
        ax.set_title(title, fontsize = 'large')
        ax.grid()
        ax.legend()
        pp.savefig(fig)

    pp.close()

    # Predicted Background and Signal


    figure_name = f"{output_directory}model_prediction"+FIGURE_FORMAT
    pp = PdfPages(figure_name)

    fig, ax = plt.subplots(1, figsize = (15, 5), constrained_layout=True )
    title = f"Predicted counts from Models and IRFs. Curve from {configuration['Name_Instrument']}"
    title+= f" {configuration['Name_Detector']}, {transient['name']}. "
    title+= f"Energy range [{configuration['Energy_Range_Reco'][0]}, {configuration['Energy_Range_Reco'][1]}] {u.Unit(configuration['Energy_Unit'])}."

    signal = List_of_Datasets[0].info_table()['npred_signal']
    background = List_of_Datasets[0].info_table()['npred_background']
    counts = List_of_Datasets[0].info_table()['npred']
    widths = List_of_Datasets[0].info_table()['livetime'].value
    ax.step(Time_Centroids.value, counts,label = f"Counts: signal+bkgd", color = 'C1', where = 'mid')
    ax.step(Time_Centroids.value, signal,label = f"Signal", color = 'C2', where = 'mid')
    ax.step(Time_Centroids.value, background,label = f"Background", color = 'C0', where = 'mid')
    ax.set_xlabel('Time [s]', fontsize = 'large')
    ax.set_ylabel('Predicted Counts', fontsize = 'large')
    ax.set_title(title, fontsize = 'large')
    ax.grid()
    ax.legend()
    pp.savefig(fig)

    pp.close()


    return None






