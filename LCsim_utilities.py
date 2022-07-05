###########################################
#
#
###########################################

############ Imports

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from matplotlib import ticker, cm
from matplotlib.colors import LogNorm, PowerNorm

import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.time import Time
from astropy.table import Table, QTable, hstack
from astropy.io import fits

import logging
import os
import yaml
from yaml import SafeLoader
from time import time

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
    LightCurveTemplateTemporalModel
)
from gammapy.utils.random import get_random_state

############ Functions

def Welcome():
    """
    Welcome function. This is used to print info about the code on execution.
    """
    print('Welcome! A proper Welcome must be implemented.\n')
    return None

def Goodbye(execution_time_start):
    """
    Goodbye function. Mostly counts time.
    """
    print('\nTotal Runtime = ', time()-execution_time_start, '(s).')
    print('Goodbye!')
    return 0



