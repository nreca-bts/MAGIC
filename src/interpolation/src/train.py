#!/usr/bin/env python


import re, itertools, yaml
from pprint import pprint
from pathlib import Path
import scipy
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import timeVAE.data_utils
import timeVAE.vae.vae_utils
import timeVAE.vae_pipeline
import timeVAE.paths
import paths
import visualize


def main():
    with open(paths.APPLIANCES_FILE_PATH) as f:
        appliances = yaml.safe_load(f)
    with open(timeVAE.paths.CFG_FILE_PATH) as f:
        config = yaml.safe_load(f)
    appliance_name = appliances['appliance_name']
    # - If an appliance was specified, only train that appliance and ignore the rest
    if appliance_name:
        appliance = appliances['appliances'][appliance_name]
        train_model(appliance, config['show_training_figs'])
    # - If no appliance was specified, train all appliances
    else:
        for appliance in appliances['appliances'].values():
            train_model(appliance, config['show_training_figs'])


def train_model(appliance, show_graph):
    '''
    Create a timeVAE model for the given appliance

    :param show_graph: whether to show a graph of some generated shapelet samples
    :type show_graph: bool
    :param appliance: a dictionary definition of an appliance
    :type appliance: dict
    '''
    assert isinstance(appliance, dict)
    df_appliance = get_appliance_data(Path(paths.AMPDS2_FILE_PATH), appliance['name'])
    ary_shapelet, df_shapelet = get_shapelets(df_appliance, appliance['shapelet_start_W'], appliance['shapelet_end_W'])
    if appliance['shapelet_sliding_window'] is True:
        ary_shapelet = get_sliding_window_shapelets(ary_shapelet, appliance['shapelet_length'])
    ary_shapelet = filter_shapelets(ary_shapelet, appliance['shapelet_length'], appliance['shapelet_max_W'])
    with open(timeVAE.paths.HYPERPARAMETERS_FILE_PATH) as f:
        config = yaml.safe_load(f)
    model_name = f'shapelets_{appliance["name"]}_len-{ary_shapelet.shape[1]}_lat-{config["timeVAE"]["latent_dim"]}_epo-{appliance["timeVAE_epochs"]}_win-{appliance["shapelet_sliding_window"]}'
    write_shapelets(model_name, ary_shapelet)
    timeVAE.vae_pipeline.run_vae_pipeline(model_name, 'timeVAE', appliance['timeVAE_epochs'])
    model_dir = Path(timeVAE.paths.MODELS_DIR) / model_name
    df_prior = get_prior_sampled_shapelets(model_dir, 100)
    # - Choose how much data to graph
    #end = 100000
    end = None
    if show_graph:
        visualize.graph_appliance(appliance['name'], df_appliance, df_shapelet, df_prior, end, model_name)


def get_appliance_data(h5_filepath, appliance_name):
    '''
    - https://www.nature.com/articles/sdata201637
    - https://makonin.com/doc/EPEC_2013.pdf
        - The only column I care about is active power (i.e. real power). It's in Watts

    :param h5_filepath: the path to the AMPds2 .h5 file
    :type h5_filepath: Path
    :param appliance_name: the name of the appliance
    :type appliance_name: str
    :return: the appliance data 
    :rtype: DataFrame
    '''
    assert isinstance(h5_filepath, Path)
    assert isinstance(appliance_name, str)
    assert h5_filepath.is_file()
    name_to_meter = {
        'house': 'meter1',
        'light2': 'meter2',
        'light3': 'meter3',
        'light4': 'meter4',
        'dryer': 'meter5',
        'washer': 'meter6',
        'sockets7': 'meter7',
        'dishwasher': 'meter8',
        'workbench': 'meter9',
        'security': 'meter10',
        'fridge': 'meter11',
        'hvac': 'meter12',
        'garage': 'meter13',
        'heat_pump': 'meter14',
        'water_heater': 'meter15',
        'light16': 'meter16',
        'sockets17': 'meter17',
        'rental_suite': 'meter18',
        'television': 'meter19',
        'sockets20': 'meter20',
        'oven': 'meter21'
    }
    assert appliance_name in name_to_meter.keys()
    meter = name_to_meter[appliance_name]
    # - Every .h5 file is organized according to a domain-specific directory hierarchy
    with pd.HDFStore(h5_filepath) as data_store:
        #key = '/building1/'
        #node = data_store.get_node(key)
        #pprint(node._v_attrs.metadata['appliances'])
        df = data_store.get(f'/building1/elec/{meter}')
    df = df[[('power', 'active')]]
    df.columns = [0]
    # - Adjust from local time to UTC time
    df.index = pd.date_range(start='2012-04-01 07:00:00-00:00', end='2014-04-01 06:59:00-00:00', freq='1min')
    return df


def get_shapelets(data, shapelet_start_W, shapelet_end_W):
    '''
    # - TODO: work on better shapelet detection for constant loads like light16 in this function or in other functions. Perhaps something that
        involves min/max start and end threshold AND min/max delta changes

    - Return all of the shapelets in the data. The longest shapelet will not have any 0 values. All shapelets shorter than the longest shapelet will
      be padded at the end with 0s. If I want to arbitrarily reduce shapelets to a particular length (e.g. 15 minutes) and/or create sliding windows
      that may sample from the same shapelet multiple times, those are separate operations

    :param shapelet_start_W: the minimum value that a reading can have which will mark the start of an "on" event. For example, if
        shapelet_start_W is 100 W, then 0 W -> 99 W would not trigger an "on" event. While an "on" event is in-progress, all subsequent "on" events
        are ignored until an "off" event is found (and the current "on" event is terminated). For example, 0 W -> 100 W for a fridge creates an "on"
        event. Subsequent values > 100 W are ignored until the "on" event is terminated.
    :type shapelet_start_W: int
    :param shapelet_end_W: the maximum value that a reading can have which will mark the end of the "on" event. For example, if shapelet_end_W
        is 1 W, then 128 W -> 5 W would not trigger an "off" event, but 5 W -> 1 W would trigger an "off" event and mark the end of a shapelet.
    :type shapelet_end_W: int
    :return: all detected shapelets as both an ndarray and as a DataFrame
    :rtype: tuple
    '''
    assert isinstance(data, pd.DataFrame)
    assert isinstance(shapelet_start_W, int)
    assert isinstance(shapelet_end_W, int)
    assert shapelet_start_W > shapelet_end_W
    ary_data = data.iloc[:, 0].values
    is_on = False
    max_len = 0
    i = 0
    # - max_start_idx, max_end_idx, and shapelet_lengths are just for data examination
    max_start_idx = 0
    max_end_idx = 0
    shapelet_lengths = []
    shapelet_indexes = []
    while i < len(ary_data):
        if ary_data[i] >= shapelet_start_W and not is_on:
            is_on = True
            start_idx = i
        if ary_data[i] <= shapelet_end_W and is_on:
            is_on = False
            end_idx = i
            # - Here, indexes are inclusive
            shapelet_indexes.append((start_idx, end_idx))
            shapelet_length = end_idx - start_idx + 1
            shapelet_lengths.append(shapelet_length)
            if shapelet_length > max_len:
                max_len = shapelet_length 
                max_start_idx = start_idx
                max_end_idx = end_idx
        i += 1
    # - Pad all shapelets at the end with 0s to match the length of the longest shapelet
    ary = np.zeros((len(shapelet_indexes), max_len), dtype=ary_data.dtype)
    for i, (start, end) in enumerate(shapelet_indexes):
        segment = ary_data[start:end + 1]
        ary[i, :len(segment)] = segment
    # - Create a DataFrame for graphing
    data_masked = np.zeros_like(ary_data)
    for start, end in shapelet_indexes:
        data_masked[start:end + 1] = ary_data[start:end + 1]
    df = pd.DataFrame(data_masked)
    df.index = pd.date_range(start='2012-04-01 07:00:00-00:00', periods=len(df), freq='1min')
    #print(f'Max shapelet length: {max_len}')
    #print(f'Mode shapelet length: {scipy.stats.mode(shapelet_lengths)}')
    #print(f'Mean shapelet length: {np.mean(shapelet_lengths)}')
    #print(f'Raw shapelets shape: {ary.shape}')
    return ary, df
    

def filter_shapelets(shapelets, length, shapelet_max_W=None):
    '''
    Remove shapelets that are shorter than the specified length and truncate shapelets that are longer than specified length. Call this AFTER
    get_sliding_window_shapelets()

    :param length: the length that the shapelets should be truncated to
    :type length: int
    :param shapelet_max_W: the maximum value allowed in a shapelet. A shapelet with a value greater than this is removed. Helps with light2
    :type shapelet_max_W: int
    :return: the filtered shapelets
    :rtype: ndarray
    '''
    assert isinstance(shapelets, np.ndarray)
    # - Truncate all shapelets to the length
    shapelets = np.array([s[:length] for s in shapelets])
    # - Remove any shapelets with 0s because they're too short
    shapelets = shapelets[~np.any(shapelets <= 0, axis=1)]
    if shapelet_max_W is not None:
        shapelets = shapelets[~np.any(shapelets > shapelet_max_W, axis=1)]
    print(f'Filtered (with or without sliding windows) shapelet count: {shapelets.shape}')
    return shapelets


def get_sliding_window_shapelets(shapelets, window_length):
    '''
    Call this BEFORE filter_shapelets() because otherwise there will be nothing to slide over! Don't use windows.reshape() because it crashes for
    large arrays (e.g. hvac).

    :return: the shapelets 
    :rtype: ndarray
    '''
    assert isinstance(shapelets, np.ndarray)
    assert shapelets.ndim == 2
    n_samples, time_steps = shapelets.shape
    if time_steps < window_length:
        raise Exception(f'Cannot create sliding windows because time_steps "{time_steps}" is less than window_length "{window_length}".')
    all_windows = []
    for i in range(n_samples):
        row = shapelets[i]
        # - Skip fully zero rows (optional)
        if np.all(row == 0):
            continue
        row_windows = sliding_window_view(row, window_length)
        # - Remove shapelets that only contain 0s
        row_windows = row_windows[~np.all(row_windows == 0, axis=1)]
        all_windows.append(row_windows)
    if not all_windows:
        raise Exception(f'All sliding windows were empty.')
    return np.vstack(all_windows)


def write_shapelets(model_name, shapelets):
    '''
    :rtype: None
    '''
    assert isinstance(model_name, str)
    # - Structure the data in the format the timeVAE expects
    shapelets = np.array([[[i] for i in a] for a in shapelets])
    output_path = Path(timeVAE.paths.DATASETS_DIR) / model_name
    # - timeVAE expects the .npz file to name the ndarray "data" instead of "arr_0"
    #   - If savez isn't given a kwarg to name the ndarray, it defaults to using the name "arr_0"
    np.savez(output_path, data=shapelets)


def get_prior_sampled_shapelets(model_dir, sample_count):
    '''
    Return shapelets for graphing (don't filter the shapelets)

    :return: some shapelets sampled from the prior distribution of the given model
    :rtype: DataFrame
    '''
    assert isinstance(model_dir, Path)
    assert isinstance(sample_count, int)
    vae = timeVAE.vae.vae_utils.load_vae_model('timeVAE', model_dir)
    samples = timeVAE.vae.vae_utils.get_prior_samples(vae, sample_count)
    scaler = timeVAE.data_utils.load_scaler(model_dir)
    inverse_scaled_prior_samples = timeVAE.data_utils.inverse_transform_data(samples, scaler)
    # - Sometimes shapelets have values < 0. There are different ways to handle this. This error correction occurs after the t-SNE plot and other built-in
    #   visualizations are generated:
    #   - Find the min value and add it to the shapelet to shift it upward
    inverse_scaled_prior_samples = [a - np.min(a) if np.min(a) < 0 else a for a in inverse_scaled_prior_samples]
    # - Use this print statement to see values for settting the shapelet_prior_max_W and shapelet_prior_min_W
    #[print(f'{i + 1} np.mean: {np.mean(a.flatten())}') for i, a in enumerate(inverse_scaled_prior_samples)]
    # - Format the shapelets for graphing
    shapelets = np.array([np.concatenate([[[0]] * 50, a, [[0]] * 50]) for a in inverse_scaled_prior_samples])
    shapelets = shapelets.reshape(shapelets.shape[0] * shapelets.shape[1])
    index = pd.date_range(start='2012-04-01 07:00:00-00:00', end='2014-04-01 06:59:00-00:00', freq='1min')
    if len(index) > len(shapelets):
        index = index[:len(shapelets)]
    elif len(index) < len(shapelets):
        shapelets = shapelets[:len(index)]
    df = pd.DataFrame(shapelets)
    df.index = index
    return df


def get_smartds_initial_load_data(csv_dir, load_dir, mode):
    '''
    Return a DataFrame of containing load shapes that represent the initial 15-minute state of each load

    :param csv_dir: the path to the directory that contains the per-unit load shape CSVs
    :type csv_dir: str
    :param load_dir: the path to the directory that contains the Loads.dss file and LoadShapes.dss file
    :type load_dir: str
    :param mode: whether we are interpolating individual OpenDSS loads or physical loads
    :type mode: str
    :return: a DataFrame of all the load load shapes in W scale
    :rtype: DataFrame
    '''
    assert isinstance(csv_dir, Path)
    assert isinstance(load_dir, Path)
    assert isinstance(mode, str)
    csv_dir = Path(csv_dir)
    load_dir = Path(load_dir)
    assert csv_dir.is_dir()
    assert load_dir.is_dir()
    assert mode == 'opendss' or mode == 'physical'
    # - Load the CSVs into a DataFrame
    df_csvs = pd.DataFrame()
    csv_glob_patterns = ['res_kw*.csv', 'com_kw*.csv']
    for pattern in csv_glob_patterns:
        for p in csv_dir.glob(pattern):
            df_csvs[p.stem] = pd.read_csv(p, header=None)
    df_csvs.index = pd.date_range(start='2012-04-01 07:00:00-00:00', end='2013-04-01 06:59:00-00:00', freq='15min')
    df_csvs = df_csvs.resample('1min').interpolate(method='linear')
    trailing_df = pd.DataFrame(np.tile(df_csvs.iloc[-1, :].values, (14, 1)))
    trailing_df.columns = df_csvs.columns
    trailing_df.index = pd.date_range(start='2013-04-01 06:46:00-00:00', end='2013-04-01 06:59:00-00:00', freq='1min')
    df_csvs = pd.concat([df_csvs, trailing_df], axis=0, ignore_index=False)
    # - Load the loads into a DataFrame
    with open(load_dir / 'Loads.dss') as f:
        lines = f.readlines()
    records = []
    for l in lines:
        mo = re.search(r'New Load.(\w+)', l)
        if not mo:
            continue
        if mode == 'opendss':
            load_name = mo.group(1)
        elif mode == 'physical':
            load_name = mo.group(1)[:-2] if mo.group(1)[-2] == '_' else mo.group(1)
        else:
            raise Exception()
        records.append({
            'load_name': load_name,
            'kw': float(re.search(r'kW=([\d.]+)', l).group(1)),
            'loadshape': re.search(r'yearly=(\w+)', l).group(1),
            'multiplier': 1
        })
    df_loads = pd.DataFrame(records).set_index('load_name')

    def update_multiplier(s):
        '''
        - Most loads in Loads.dss are split across two lines. Raise an exception if a load has different property values across its two lines
        '''
        rows = df_loads.loc[s.name]
        if rows.ndim > 1:
            if not (rows.nunique() == 1).all():
                raise Exception(f'The load {s.name} is split across {rows.ndim} lines, but those lines have at least one differet property value.')
            else:
                s.multiplier = rows.ndim
        return s

    df_loads = df_loads.apply(update_multiplier, axis=1)
    df_loads = df_loads.loc[~df_loads.index.duplicated(keep='first'), :]
    
    def apply_kw_mulitplier(s):
        '''
        - Transform the per-unit load shapes in the CSVs into actual load shapes for the loads. We also scale from kW to W here
        '''
        load_shape = df_csvs[s.loadshape] * s.kw * 1000 * s.multiplier
        load_shape.name = s.name
        return load_shape

    df_csvs = df_loads.apply(apply_kw_mulitplier, axis=1).T
    return df_csvs


def get_data(csv_dir):
    '''
    Return a DataFrame containing the load shapes in the csvs

    :param csv_dir: the path to the directory that contains the per-unit load shape CSVs
    :type csv_dir: str
    :return: a DataFrame containing the load shapes in the csvs
    :rtype: DataFrame
    '''
    assert isinstance(csv_dir, Path)
    csv_dir = Path(csv_dir)
    assert csv_dir.is_dir()
    # - Load the CSVs into a DataFrame
    df_csvs = pd.DataFrame()
    for p in csv_dir.glob('*.csv'):
        df_csvs[p.stem] = pd.read_csv(p, header=None)
    df_csvs.index = pd.date_range(start='2012-04-01 07:00:00-00:00', end='2013-04-01 06:59:00-00:00', freq='1min')
    return df_csvs


def get_models(sample_count):
    '''
    - TODO: this function ASSUMES all of the models have been trained according to the current configs. Need to ensure this is always the case

    :param sample_count: the number of samples to obtain from each vae model. A higher number will result in longer program execution time but greater
        interpolation variability. Runs in O(n) time: 10,000 samples: 13 seconds. 100,000 samples: 84 seconds. 1,000,000 samples: 852 seconds
    :type sample_count: int
    :return: all of the models described by shapelet length and shapelet mean
    :rtype: DataFrame
    '''
    assert isinstance(sample_count, int)
    assert sample_count > 0
    models_dir = Path(timeVAE.paths.MODELS_DIR)
    models = {}
    for path in models_dir.glob('shapelets_*'):
        models[path.stem] = path
    with open(paths.APPLIANCES_FILE_PATH) as f:
        appliances = yaml.safe_load(f)['appliances']
    records = []
    for appliance in appliances.values():
        pattern = f'shapelets_{appliance["name"]}_len-{appliance["shapelet_length"]}_lat-{appliance["timeVAE_latent_dim"]}_epo-{appliance["timeVAE_epochs"]}_win-{str(appliance["shapelet_sliding_window"]).capitalize()}'
        path = models.get(pattern)
        if path is None:
            raise Exception(f'Found no matching trained model for the appliance configuration {appliance}.')
        appliance_name = appliance['name']
        samples = timeVAE.data_utils.inverse_transform_data(timeVAE.vae.vae_utils.get_prior_samples(timeVAE.vae.vae_utils.load_vae_model('timeVAE', path), sample_count), timeVAE.data_utils.load_scaler(path))
        # - First, find the min value, and if it's less than 0, and add its inverse to the shapelet to shift it upward
        # - Next, filter shapelets to only include those that fit within the desired range, and flatten each filtered shapelet
        samples = np.array([b.flatten() for b in [a - np.min(a) if np.min(a) < 0 else a for a in samples] if appliances[appliance_name]['shapelet_prior_min_W'] <= np.mean(b) <= appliances[appliance_name]['shapelet_prior_max_W']])
        records.append({
            'name': appliance_name,
            # - This is a circular iterator. Eventually, it will loop back around and grab the same sample
            'sample_generator': itertools.cycle(samples),
            'mean': round(np.mean(samples), 0),
            'std': round(np.std(samples), 0),
            'length': int(appliance['shapelet_length']),
            'load_consumption': appliances[appliance_name]['load_consumption'],
            'load_timing': appliances[appliance_name]['load_timing'],
            'shapelet_prior_max_W': appliances[appliance_name]['shapelet_prior_max_W'],
            'shapelet_prior_min_W': appliances[appliance_name]['shapelet_prior_min_W']
        })
    df = pd.DataFrame(records).set_index('name')
    return df


if __name__ == '__main__':
    main()