#!/usr/bin/env python


import os


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # src/interpolation
DATASETS_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUTS_DIR = os.path.join(ROOT_DIR, 'outputs')
SRC_DIR = os.path.join(ROOT_DIR, 'src')
CONFIG_DIR = os.path.join(SRC_DIR, 'config')
APPLIANCES_FILE_PATH = os.path.join(CONFIG_DIR, 'appliances.yaml')
CFG_FILE_PATH = os.path.join(CONFIG_DIR, 'interpolation.yaml')
AMPDS2_FILE_PATH = os.path.join(DATASETS_DIR, 'ampds2', 'AMPds2.h5')
SMARTDS_LOADSHAPES_DIR = os.path.join(DATASETS_DIR, 'smartds', '2018', 'GSO', 'rural', 'profiles')
SMARTDS_LOADS_DIR = os.path.join(DATASETS_DIR, 'smartds', '2018', 'GSO', 'rural', 'scenarios', 'base_timeseries', 'opendss')