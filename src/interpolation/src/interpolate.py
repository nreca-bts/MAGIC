#!/usr/bin/env python


import concurrent, logging, yaml
from multiprocessing import current_process
from pathlib import Path
import numpy as np
import pandas as pd
import train
import paths


def main():
    logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
    logging.info(f'Process name {current_process()} is running main()...')
    with open(paths.CFG_FILE_PATH) as f:
        config = yaml.safe_load(f)
    load_category = config['load_category']
    feeder_path = config['feeder_path']
    is_smooth = config['is_smooth']
    is_fast = config['is_fast']
    threshold = config['threshold']
    output_dir = Path(paths.OUTPUTS_DIR) / feeder_path / f'thr-{threshold}_smo-{is_smooth}_fas-{is_fast}_mode-{load_category}'
    output_dir.mkdir(parents=True, exist_ok=True)
    data = train.get_smartds_initial_load_data(
        Path(paths.SMARTDS_LOADSHAPES_DIR),
        Path(paths.SMARTDS_LOADS_DIR) / feeder_path,
        load_category)
    if config['load_name']:
        data = data[[config['load_name']]]
    if config['multiprocessing']:
        interpolate_data_multiprocessing(data, threshold, is_smooth, is_fast, output_dir)
    else:
        interpolate_data_singlethreaded(data, threshold, is_smooth, is_fast, output_dir)
    logging.info(f'Process name {current_process()} finished running main().')


def interpolate_data_singlethreaded(data, threshold, is_smooth, is_fast, output_dir):
    '''
    :param data: the DataFrame of load shapes to interpolate
    :type data: DataFrame
    :param threshold: a limit that constrains the mean of the interpolated data to a range around the mean of the original data
    :type threshold: int
    :param is_smooth: whether to filter models by std to ensure a smooth fit. This has the biggest impact on the interpolated shape by far
    :type is_smooth: bool
    :param is_fast: whether to prioritize speed over variability. is_fast=True reduces execution time by 37.5% (125 seconds vs. 200 seconds) by using
        as many of one model as possible. Doing this results in a different interpolated shape which may be more repetitive
    :type is_fast: bool
    :param output_dir: the directory where the interpolated files should be output
    :type output_dir: Path
    :rtype: None
    '''
    assert isinstance(data, pd.DataFrame)
    assert isinstance(threshold, int)
    assert isinstance(is_smooth, bool)
    assert isinstance(is_fast, bool)
    assert isinstance(output_dir, Path)
    data.apply(lambda s: _interpolate_data(s.values, s.name, threshold, is_smooth, is_fast, output_dir))
    

def interpolate_data_multiprocessing(data, threshold, is_smooth, is_fast, output_dir):
    '''
    Runtime approximately 13 hours for rhs0_1247/rhs0_1247--rdt1527 in physical mode

    :param data: the DataFrame of load shapes to interpolate
    :type data: DataFrame
    :param threshold: a limit that constrains the mean of the interpolated data to a range around the mean of the original data
    :type threshold: int
    :param is_smooth: whether to filter models by std to ensure a smooth fit. This has the biggest impact on the interpolated shape by far
    :type is_smooth: bool
    :param is_fast: whether to prioritize speed over variability. is_fast=True reduces execution time by at least 37.5% (125 seconds vs. 200 seconds)
        by using as many of one model as possible. Doing this results in a different interpolated shape which may be more repetitive
    :type is_fast: bool
    :param output_dir: the directory where the interpolated files should be output
    :type output_dir: Path
    :rtype: None
    '''
    assert isinstance(data, pd.DataFrame)
    assert isinstance(threshold, int)
    assert isinstance(is_smooth, bool)
    assert isinstance(is_fast, bool)
    assert isinstance(output_dir, Path)
    process_argument_tuples = [(data[col].values, col, threshold, is_smooth, is_fast, output_dir) for col in data.columns]
    with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
        future_list = []
        for process_argument_list in process_argument_tuples:
            future_list.append(executor.submit(_interpolate_data, *process_argument_list))
        for f in concurrent.futures.as_completed(future_list):
            if f.exception() is None:
                logging.info(f'{f.result()} was returned')
            else:
                raise f.exception()
    

def _interpolate_data(data, name, threshold, is_smooth, is_fast, output_dir, sample_count=50000):
    '''
    :param data: the data of the load shape to be interpolated
    :type data: ndarray
    :param name: the name of the load
    :type name: str
    :param threshold: a limit that constrains the mean of the interpolated data to a range around the mean of the original data
    :type threshold: int
    :param is_smooth: whether to filter models by std to ensure a smooth fit. This has the biggest impact on the interpolated shape by far
    :type is_smooth: bool
    :param is_fast: whether to prioritize speed over variability. is_fast=True reduces execution time by 37.5% (125 seconds vs. 200 seconds) by using
        as many of one model as possible. Doing this results in a different interpolated shape which may be more repetitive
    type is_fast: bool
    :param output_dir: the directory where the interpolated files should be output
    :type output_dir: Path
    :param sample_count: the number of samples to obtain from each vae model. A higher number will result in longer program execution time but greater
        interpolation variability. 360 * 24 * 60 = 525600, 525600 / 10 = 52560, so 50,000 samples seems reasonable.
    :type sample_count: int
    :rtype: None
    '''
    assert isinstance(data, np.ndarray)
    assert isinstance(name, str)
    assert isinstance(threshold, int)
    assert isinstance(is_smooth, bool)
    assert isinstance(is_fast, bool)
    assert isinstance(output_dir, Path)
    assert isinstance(sample_count, int)
    assert sample_count > 0
    logging.basicConfig(filename='log.txt', level=logging.INFO, format='%(levelname)s: %(asctime)s: %(message)s')
    logging.info(f'Process name {current_process()} is running _interpolate_data() for "{name}"...')
    models = train.get_models(sample_count)
    # - Sort the models by mean in descending order because every data group relies on this
    models = models.sort_values(by='mean', axis=0, ascending=False)
    groups = _group_data(data, threshold)
    interpolated_data = []
    for group in groups:
        interpolated_data.append(_interpolate_group(models, data, group, threshold, is_smooth, is_fast))
    interpolated_data = np.concatenate(interpolated_data)
    np.savetxt(output_dir / f'{name}.csv', interpolated_data, delimiter=',', fmt='%.15f')


def _group_data(data, threshold):
    '''
    # - TODO: the way that I group data means that big loads with large standard deviations will never be used

    :param data: the data of the load shape to be interpolated
    :type data: ndarray
    :param threshold: the maximum difference that two adjacent readings can have before being grouped separately
    :type threshold: int
    :return: the data organized into groups. Each group is itself a tuple of (<start index>, <end index> (inclusive))
    :rtype: list
    '''
    assert isinstance(data, np.ndarray)
    assert isinstance(threshold, int)
    data_groups = []
    i = 0
    n = len(data)
    while i < n:
        start_value = np.round(data[i])
        group_end = i
        for j in range(i + 1, n):
            current_value = np.round(data[j])
            if abs(current_value - start_value) <= threshold:
                group_end = j
            else:
                break
        data_groups.append((i, group_end))
        i = group_end + 1
    return data_groups


def _interpolate_group(models, data, group, threshold, is_smooth, is_fast):
    '''

    :return: the interpolated data
    :rtype: ndarray
    '''
    assert isinstance(models, pd.DataFrame)
    assert isinstance(data, np.ndarray)
    assert isinstance(group, tuple)
    assert isinstance(threshold, int)
    assert isinstance(is_smooth, bool)
    assert isinstance(is_fast, bool)
    length = group[1] - group[0] + 1
    interpolated_data = np.array([0.0] * length)
    diff = np.mean(data[group[0]:group[1] + 1])
    std = np.std(data[group[0]:group[1] + 1])
    if is_smooth:
        models = models[(models['mean'] < diff + threshold) & (models['std'] < std + threshold)]
    else:
        models = models[(models['mean'] < diff + threshold)]
    while diff > 0 and diff > threshold:
        for row in models.itertuples():
            instances = int(diff // row.mean)
            if instances == 0:
                continue
            if not is_fast:
                instances = 1
            # - instances: how many samples can be vertically stacked to approach the group
            # - 1 + length // row.length: how many samples need to be horizontally appended to match the length of the group
            samples = np.array([next(row.sample_generator) for _ in range(instances * (1 + length // row.length))])
            # - Reshape the samples how I need them, then vertically sum them
            samples = samples.reshape(instances, int((samples.shape[0] * samples.shape[1]) // instances))
            samples = np.sum(samples, axis=0)[:length]
            interpolated_data += samples
            diff -= np.mean(samples)
            if diff <= 0 or diff <= threshold:
                break 
    return interpolated_data


if __name__ == '__main__':
    main()