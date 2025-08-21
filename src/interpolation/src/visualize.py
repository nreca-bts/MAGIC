#!/usr/bin/env python


from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import paths
import train


def main():
    # - Pre-inspect various appliances
    #appliance_name = 'oven'
    #df_appliance = train.get_appliance_data(Path(paths.AMPDS2_FILE_PATH), appliance_name)
    #graph_appliance(appliance_name, df_appliance, df_appliance, df_appliance)

    load_name = 'load_p1rlv1359' # residential
    #load_name = 'load_p1rlv1359_1' # residential
    #load_name = 'load_p1rlv1864' # commercial
    #load_name = 'load_p1rlv5841' # big commercial load
    #load_name = 'load_p1rlv2032' # big commercial load

    mode = 'physical'
    #mode = 'opendss'

    circuit_path = Path('rhs0_1247/rhs0_1247--rdt1527')
    #circuit_path = Path('rhs2_1247/rhs2_1247--rdt1262')
    #circuit_path = Path('rhs2_1247/rhs2_1247--rdt1264')

    view_interpolation_results(load_name, mode, circuit_path)


def view_interpolation_results(load_name, mode, circuit_path):
    '''
    View the interpolated data for a given load

    :param load_name: the name of the load and corresponding csv
    :type load_name: str
    :param mode: either "physical" or "opendss"
    :type mode: str
    :param circuit_path: the path segment to the feeder that contains the load
    :type circuit_path: str
    :rtype: None
    '''
    threshold = 10
    is_smooth = True
    is_fast = False
    df_vae = train.get_data(Path(paths.OUTPUTS_DIR) / circuit_path / f'thr-{threshold}_smo-{is_smooth}_fas-{is_fast}_mode-{mode}')
    df_vae = df_vae[[f'{load_name}']]
    df_smartds = train.get_smartds_initial_load_data(
        Path(paths.SMARTDS_LOADSHAPES_DIR),
        Path(paths.SMARTDS_LOADS_DIR) / circuit_path,
        mode)
    df_smartds = df_smartds[[load_name]]
    df_ampds2 = train.get_appliance_data(Path(paths.AMPDS2_FILE_PATH), 'house')
    graph_ampds2_and_smartds(load_name, df_ampds2, df_smartds, df_vae)


def graph_appliance(appliance_name, df_actual, df_shapelet, df_prior, end=None, text=None):
    '''
    '''
    assert isinstance(appliance_name, str)
    assert isinstance(df_actual, pd.DataFrame)
    assert isinstance(df_shapelet, pd.DataFrame)
    assert isinstance(df_prior, pd.DataFrame)
    assert end is None or isinstance(end, int)
    if end is not None:
        df_actual = df_actual.iloc[:end]
        df_shapelet = df_shapelet.iloc[:end]
        df_prior = df_prior.iloc[:end]
    _graph_dataframes(
        [
            {
                'name': f'{appliance_name} actual 1-minute active power',
                'df': df_actual,
                'column': 0,
                'color': 'blue'
            },
            {
                'name': f'{appliance_name} shapelet 1-minute active power',
                'df': df_shapelet,
                'column': 0,
                'color': 'red'
            },
            {
                'name': f'{appliance_name} prior samples 1-minute active power',
                'df': df_prior,
                'column': 0,
                'color': 'green'
            },
        ],
        {
            'title': {'text': f'{appliance_name} actual 1-minute active power vs. shapelet 1-minute active power vs. prior samples 1-minute active power', 'font': {'size': 25}},
            'xaxis':  {'title': {'text': 'Timestamp'}},
            'yaxis': {'title': {'text': 'Power (Watts)'}},
            'legend': {'title': {'text': 'Legend'}},
            'font': {'size': 18}    
        },
        [
            {
                'text': text,
                'x_offset': 1.30, # 1.32
                'y_offset': -0.075,
            }
        ])


def graph_ampds2_and_smartds(load_name, df_ampds2, df_smartds, df_vae, end=None):
    '''
    :param end: the end DataFrame row index at which to stop graphing
    :type end: int
    '''
    assert isinstance(load_name, str)
    assert isinstance(df_ampds2, pd.DataFrame)
    assert isinstance(df_smartds, pd.DataFrame)
    assert isinstance(df_vae, pd.DataFrame)
    assert end is None or isinstance(end, int)
    if end is not None:
        df_ampds2 = df_ampds2.iloc[:end]
        df_smartds = df_smartds.iloc[:end]
        df_vae = df_vae.iloc[:end]
    _graph_dataframes(
        [
            {
                'name': 'AMPds2 1-minute active power',
                'df': df_ampds2,
                'column': 0,
                'color': 'blue'
            },
            {
                'name': f'{load_name} 15-minute active power',
                'df': df_smartds,
                'column': load_name,
                'color': 'red'
            },
            {
                'name': f'{load_name} 1-minute interpolated active power',
                'df': df_vae,
                'column': f'{load_name}',
                'color': 'green'
            },
        ],
        {
            'title': {'text': f'AMPds2 1-minute active power vs. {load_name} 15-minute active power vs. {load_name} 1-minute interpolated active power', 'font': {'size': 25}},
            'xaxis':  {'title': {'text': 'Timestamp'}},
            'yaxis': {'title': {'text': 'Power (Watts)'}},
            'legend': {'title': {'text': 'Legend'}},
            'font': {'size': 18}    
        },
        [
            {
                'text': 'AMPds2 mean: ' + '{0:.2f}'.format(df_ampds2[0].mean()) + '<br>' +
                    'AMPds2 stdev: ' + '{0:.2f}'.format(df_ampds2[0].std()),
                'x_offset': 1.32,
                'y_offset': -0.075,
            },
            {
                'text': 'SMART-DS mean: ' + '{0:.2f}'.format(df_smartds[load_name].mean()) + '<br>' +
                    'SMART-DS stdev: ' + '{0:.2f}'.format(df_smartds[load_name].std()),
                'x_offset': 1.32,
                'y_offset': -0.1,
            },
            {
                'text': 'SMART-DS VAE mean: ' + '{0:.2f}'.format(df_vae[f'{load_name}'].mean()) + '<br>' +
                    'SMART-DS VAE stdev: ' + '{0:.2f}'.format(df_vae[f'{load_name}'].std()),
                'x_offset': 1.32,
                'y_offset': -0.125,
            },
        ])


def _graph_dataframes(dataframes, fig_layout, annotations=None):
    '''
    Utility function for graphing dataframes. Not meant to be called directly

    :param dataframes: a list of dicts that contain DataFrames to graph
    :type dataframes: list
    :param fig_layout: a dict of figure layout settings
    :type fig_layout: dict
    :param annotations:
    :type annotations: list

    - Make nice graphs of the ampds2 dataset for some slides
    '''
    assert isinstance(dataframes, list)
    assert isinstance(fig_layout, dict)
    assert isinstance(annotations, list) or annotations is None
    fig = go.Figure()
    x_hover_template = 'Timestamp: %{x|%Y-%m-%d %H:%M}'
    for d in dataframes:
        fig.add_trace(
            go.Scatter(
                name=d['name'],
                x=d['df'].index,
                y=d['df'][d['column']],
                hovertemplate='<br>'.join([
                    x_hover_template,
                    'Energy: %{y} W',
                    '<extra></extra>']),
                line=dict(color=d['color'])))
    fig.update_layout(fig_layout)
    y_offset = .85
    if annotations is not None:
        for d in annotations:
            y_offset += d['y_offset']
            fig.add_annotation(
                text=d['text'],
                align='left',
                showarrow=False,
                xref='paper',
                yref='paper',
                x=d['x_offset'],
                y=y_offset,
                bordercolor='black',
                borderwidth=1)
    fig.show()


if __name__ == '__main__':
    main()