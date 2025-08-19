#!/usr/bin/env python


import re, math, datetime
from pathlib import Path
import pandas as pd
import plotly.graph_objects as go
import paths


def main():
    circuit_path = Path('rhs0_1247/rhs0_1247--rdt1527')
    #circuit_path = Path('rhs2_1247/rhs2_1247--rdt1262')
    #circuit_path = Path('rhs2_1247/rhs2_1247--rdt1264')
    loads = get_loads(paths.SMARTDS_LOADS_DIR / circuit_path / 'Loads.dss')
    loadshape_prefix = None
    #loadshape_prefix = 'res_'
    #loadshape_prefix = 'com_'
    print_demand_metrics(loads, loadshape_prefix)
    loadshapes = get_loadshapes(loads)

    #calculate_daily_peak_demand_time(loadshapes, loadshape_prefix, output_png=False)

    # - Create a line graph
    # - Line graphs to create:
    #   - res and commercial daily-aggregated summer (done)
    #   - res and commercial daily-aggregated winter (done)
    #   - res daily-aggregated summer (done)
    #   - res daily-aggregated winter (done)
    #   - com daily-aggregated summer (done)
    #   - com daily-aggregated winter (done)
    calculate_peak_demand(
        loads,
        loadshapes,
        loadshape_prefix,
        single_day_aggreation=False,
        individual_traces=False,
        season=None,
        output_png=False)


def get_loads(loads_path):
    '''
    - Summary_data.csv states that total peak time circuit power (kW) is 5356 and total peak time real customer power (kW) is 5100
        - If I use the regex that assumes two load objects suffixed with _1 and _2 are actually the same load, then I get a total peak time circuit
          power (kVA) of 2944 and a total peak time real customer power (kW) of 2884.
        - If I use the regex that assumes every line in Load.dss is a different load, I get a total peak time circuit power (kVA) of 5206 and a total
          peak time real customer power (kW) of 5100
            - This matches the value in Summary_data.csv exactly, so this MUST be the correct regex
    :returns: a dict mapping each load to various properties
    :rtype: dict
    '''
    loads = {}
    with open(loads_path) as f:
        for line in f:
            # - Use this regex when it is assumed that two load objects suffixed with _1 and _2 are actually the SAME load (i.e. detect physical loads)
            load_name_mo = re.search(r'Load.load_([A-Za-z0-9]+_?)', line)
            # - Use this regex when it is assumed that every line in Loads.dss is a different load (i.e. OpendSS loads)
            #load_name_mo = re.search(r'Load.(\w+)', line)
            if load_name_mo:
                load_name = load_name_mo.group(1)
                if load_name not in loads:
                    loads[load_name] = {}
            kw_mo = re.search(r'kW=(\d+.\d+)', line)
            if kw_mo:
                kw = float(kw_mo.group(1))
                if 'kw' not in loads[load_name]:
                    loads[load_name]['kw'] = kw
                # - I could average the kW properties of the two load objects (which we assume to represent the same load), but I'm interested if
                #   there's ever a case where the kW values are different, so raise an exception instead
                elif loads[load_name]['kw'] != kw:
                    raise Exception
            kvar_mo = re.search(r'kvar=(\d+.\d+)', line)
            if kvar_mo:
                kvar = float(kvar_mo.group(1))
                if 'kvar' not in loads[load_name]:
                    loads[load_name]['kvar'] = kvar
                elif loads[load_name]['kvar'] != kvar:
                    raise Exception
            yearly_mo = re.search(r'yearly=(\w+)', line)
            if yearly_mo:
                yearly = yearly_mo.group(1)
                if 'yearly' not in loads[load_name]:
                    loads[load_name]['yearly'] = yearly
                elif loads[load_name]['yearly'] != yearly:
                    raise Exception
            phases_mo = re.search(r'Phases=(\d)', line)
            if phases_mo:
                phases = phases_mo.group(1)
                if 'phases' not in loads[load_name]:
                    loads[load_name]['phases'] = phases
                elif loads[load_name]['phases'] != phases:
                    raise Exception
    for v in loads.values():
        # kVA^2 = kW^2 + kVAR^2 (Pythagorean theorem)
        v['kva'] = math.sqrt(v['kw']**2 + v['kvar']**2)

    # - There are either 2212 or 1114 loads
    #   - If there are 2212 loads, 16 of them are not suffixed with _<digit>
    #   - If there are 1114 loads, 16 of them are defined on single line (i.e. they aren't suffixed with _<digit>) 
    #print(loads.keys())
    #print(f'len(loads.keys()): {len(loads.keys())}') # 2212
    #print('Total number of keys without a numeric suffix: {}'.format(len(list(filter(lambda key: not re.search(r"_\d$", key), loads.keys())))))
    return loads


def get_loadshapes(loads):
    '''
    :param loads: a dict mapping each load to various properties
    :type loads: dict
    :return: a dict mapping each load shape to various properties
    :rtype: dict
    '''
    loadshapes = {}
    for v in loads.values():
        if 'yearly' not in v:
            raise Exception('Cannot retrieve loadshapes for a dataset that does not have a LoadShapes.dss file')
        if v['yearly'] not in loadshapes:
            loadshapes[v['yearly']] = {'count': 1}
        else:
            loadshapes[v['yearly']]['count'] += 1
    # - Find the peak time of each load shape
    for k in loadshapes.keys():
        df = pd.read_csv(Path(paths.SMARTDS_LOADSHAPES_DIR) / f'{k}.csv', names=[k])#magic.root_dir.parent / f'data/loadshapes/{k}.csv', names=[k])
        df.index = pd.date_range(start='2018-01-01 00:00:00', end='2018-12-31 23:45:00', freq='15min')
        loadshapes[k]['df'] = df
        # - These properties are nice but aren't used anywhere yet
        #loadshapes[k]['peak_kw_pu'] = df.max().values[0]
        #loadshapes[k]['peak_kw_pu_ts'] = df.idxmax().values[0]
        # - This groups a year into a single day and ignores summer vs winter seasonality
        loadshapes[k]['peak_daily_kw_pu_time'] = df.groupby(lambda ts: ts.time()).mean().idxmax().values[0]
        loadshapes[k]['peak_daily_kw_pu_avg'] = df.groupby(lambda ts: ts.time()).mean().max().values[0]

    # - Only 218 load shapes are ever used, regardless of how many loads there are
    #print(len(loadshapes))

    return loadshapes


def print_demand_metrics(loads, loadshape_prefix=None):
    '''
	:param loadshape_prefix: either 'res_' for residential or 'com_' for commercial or None for both
	:type loadshape_prefix: str
    :param loads: a dict mapping each load to various properties
    :type loads: dict
    :rtype: None
    '''
    load_kw_sum = 0
    load_kva_sum = 0
    load_count = 0
    non_suffixed_loads = 0
    for k in loads.keys():
        # - This if-statement is only true if Load.dss references a LoadShapes.dss file, which is not true for the peak/ directory
        if 'yearly' in loads[k]:
            if loadshape_prefix is None or loads[k]['yearly'].startswith(loadshape_prefix):
                load_kw_sum += loads[k]['kw']
                load_kva_sum += loads[k]['kva']
                load_count += 1
                mo = re.search(r'_\d$', k)
                if mo is None:
                    non_suffixed_loads += 1
        elif loadshape_prefix is None:
            load_kw_sum += loads[k]['kw']
            load_kva_sum += loads[k]['kva']
            load_count += 1
            mo = re.search(r'_\d$', k)
            if mo is None:
                non_suffixed_loads += 1
        else:
            # - Even though I can't use peak_base_timeseries/Load.dss, these calculations are still correct because I AM getting the peak kW and kVA
            #   values directly off of the load objects anyway
            raise Exception('The only way to differentiate between residential and commercial loads is by using a Load.dss file with loads that have a "yearly" property')
    if loadshape_prefix is None:
        loadshape_prefix = 'residential and consumer'
    print(f'Number of "{loadshape_prefix}" loads: {load_count}')
    print(f'"{loadshape_prefix}" consumer total peak kW demand (real power): {load_kw_sum}')
    print(f'"{loadshape_prefix}" consumer total peak kVA demand (apparent power): {load_kva_sum}')
    print(f'"{loadshape_prefix}" consumer average peak kW demand (real power): {load_kw_sum/load_count}')
    print(f'"{loadshape_prefix}" consumer average peak kVA demand (apparent power): {load_kva_sum/load_count}')
    print(f'{non_suffixed_loads} loads of the "{loadshape_prefix}" type did not have a numeric suffix')


def calculate_daily_peak_demand_time(loadshapes, loadshape_prefix=None, output_png=False):
    '''
    We decided that this histogram is somewhat misleading so we shouldn't use it

    :param loadshapes: a dict mapping each load shape to various properties
    :type loadshapes: dict
	:param loadshape_prefix: either 'res_' for residential or 'com_' for commercial or None for both
	:type loadshape_prefix: str
	:rtype: None 
    '''
    # - Determine what the average kW consumption is at a given peak time to populate the customdata argument
    #for v in loads.values():
    #    if 'peak_daily_kw_avg' not in loadshapes[v['yearly']]:
    #        loadshapes[v['yearly']]['peak_daily_kw_avg'] = []
    #    loadshapes[v['yearly']]['peak_daily_kw_avg'].append(loadshapes[v['yearly']]['peak_daily_kw_pu_avg'] * v['kw'])
    #df = pd.DataFrame()

    # - Count the frequency of peak_daily_kw_pu_time values to create data for a histogram    
    times = []
    for k, v in loadshapes.items():
        if loadshape_prefix is None or k.startswith(loadshape_prefix):
            # - Plotly doesn't play nice with pure time objects, to create fake datetime objects
            times.extend([datetime.datetime(2018, 1, 1, v['peak_daily_kw_pu_time'].hour, v['peak_daily_kw_pu_time'].minute)] * v['count'])
            print(f'{k} - {v["peak_daily_kw_pu_time"]} : {v["peak_daily_kw_pu_avg"]} : {v["count"]}')
    times.sort()
    # - Create the histogram
    fig = go.Figure(
        go.Histogram(
            x=times,
            # - xbins.size is specified in milliseconds by default
            #   - E.g. to get 15-minute bins, set xbins.size to 15 * 60 * 1000 = 900,000 (15 minutes is equivalent to 900000 milliseconds)
            xbins=dict(start='2018-01-01 00:00', end='2018-01-02 00:00', size='900000'),
            autobinx=False,
            # - The "customdata" argument MUST be a DataFrame 
            #customdata=[[1], [2], [3]],
            #hovertemplate='<br>'.join(['Time: %{x}', 'Count: %{y}', 'kW: %{customdata[0]}', '<extra></extra>']),
            hovertemplate='<br>'.join(['Time: %{x}', 'Count: %{y}', '<extra></extra>']),
        ),
        layout={'xaxis': {'range': ['2018-01-01 00:00', '2018-01-02 00:00']}}
    )
    # - Don't show date. Only time (even though we're using fake datetimes)
    fig.update_xaxes(tickformat='%H:%M')

    if loadshape_prefix == 'res_':
        load_type = 'residential'
    elif loadshape_prefix == 'com_':
        load_type = 'commercial'
    elif loadshape_prefix is None:
        load_type = 'residential and commercial'
    fig.update_layout(
        title={'text': f'Histogram of daily peak demand times of "{load_type}" loads', 'font': {'size': 15}},
        xaxis={'title': {'text': 'Time'}},
        yaxis={'title': {'text': 'Count'}},
        legend={'title': {'text': 'Legend Title'}},
        font={'size': 18})
    if output_png:
        fig.write_image('histogram.png')
    else:
        fig.show()


def calculate_peak_demand(loads, loadshapes, loadshape_prefix=None, single_day_aggreation=False, individual_traces=False, season=None, output_png=False):
    '''
    :param loads: a dict mapping each load to various properties
    :type loads: dict
    :param loadshapes: a dict mapping each load shape to various properties
    :type loadshapes: dict
    :param loadshape_prefix: an optional string to fitler loads
        If loadshape_prefix == "res_", calculate peak demand only for residential loads
        If loadshape_prefix == "com_", calculate peak demand only for commercial loads
        If loadshape_prefix == None, calculate peak demand for all loads
    :type loadshape_prefix: str
    :param single_day_aggregation: whether to aggregate all of the data for the year into a single day
    :type single_day_aggregation: bool
    :param individual_traces: whether to show the individual load shapes
    :type individual_traces: bool
    :param season: the season to show (e.g. "summer", "winter" or None for both seasons)
    :type season: str
    '''
    loads_df = pd.DataFrame(loads).transpose()
    if loadshape_prefix:

        def filter_func(s):
            '''
            :param s: a series corresponding to a load
            :type s: Series
            '''
            return s['yearly'].startswith(loadshape_prefix)

        results = loads_df.apply(filter_func, axis=1)
        loads_df = loads_df[results]

    #print('loads_df')
    #print(loads_df)
    #print()

    loadshapes_pu_df = pd.concat([v['df'] for v in loadshapes.values()], axis=1)
    #print('loadshapes_pu_df')
    #print(loadshapes_pu_df)
    #print()

    # - Print a particular load shape if I want to see pu values (e.g. com_kw_12824_pu is the first commercial load shape and it doesn't appear in the
    #   above print statement
    #print('Show particular pu load shape values')
    #print(loadshapes_pu_df['com_kw_12824_pu'])
    #print()

    def create_load_shape(s):
        '''
        :param s: a series corresponding to a load
        :type s: Series
        '''
        # - We can choose which unit we are using to measuring peak demand: kw, kva, kvar
        unit = 'kw' # kva
        return loadshapes_pu_df[s['yearly']] * s[unit]

    loadshapes_df = loads_df.apply(create_load_shape, axis=1).transpose()
    #print('loadshapes_df')
    #print(loadshapes_df)
    #print()

    total_circuit_demand_s = loadshapes_df.sum(axis=1)
    #print('total_circuit_demand_s')
    #print(total_circuit_demand_s)
    #print()

    if loadshape_prefix == 'res_':
        load_type = 'residential'
    elif loadshape_prefix == 'com_':
        load_type = 'commercial'
    elif loadshape_prefix is None:
        load_type = 'residential and commercial'
    
    # - Optionally, show a certain season
    if season == 'summer':
        # - Summer is considered May - September
        loadshapes_df = loadshapes_df.loc[datetime.datetime(2018, 5, 1):datetime.datetime(2018, 9, 30, 23, 59)]
        total_circuit_demand_s = total_circuit_demand_s.loc[datetime.datetime(2018, 5, 1):datetime.datetime(2018, 9, 30, 23, 59)]
        title = 'Summer'
    elif season == 'winter':
        # - Winter is considered October - April
        #   - Plotly draws a line to connect each dot to the next. That's why there's a weird horizontal line 
        loadshapes_df = pd.concat([
            loadshapes_df.loc[datetime.datetime(2018, 1, 1):datetime.datetime(2018, 4, 30, 23, 59)],
            loadshapes_df.loc[datetime.datetime(2018, 10, 1):datetime.datetime(2018, 12, 31, 23, 59)]],
            axis=0)
        total_circuit_demand_s = pd.concat([
            total_circuit_demand_s.loc[datetime.datetime(2018, 1, 1):datetime.datetime(2018, 4, 30, 23, 59)],
            total_circuit_demand_s.loc[datetime.datetime(2018, 10, 1):datetime.datetime(2018, 12, 31, 23, 59)]],
            axis=0)
        title = 'Winter'
    elif season is None:
        title = 'Yearly'
    else:
        raise ValueError()

    #print(total_circuit_demand_s.loc[pd.Timestamp('2018-06-19 16:00:00'):pd.Timestamp('2018-06-19 18:00:00')])
    print(f'Peak demand for "{load_type}" loads was {total_circuit_demand_s.max()} which occured at {total_circuit_demand_s.idxmax()}') 

    # - Optionally, bin into 1-day intervals instead of 15-minute intervals
    #total_circuit_demand_s = total_circuit_demand_s.resample('1d').mean()

    # - Generate a line graph
    # - Optionally, group all data by time into a single day
    if single_day_aggreation:
        total_circuit_demand_s = total_circuit_demand_s.groupby(lambda ts: ts.time()).mean()
        title = f'{title} daily-aggregated {load_type} demand (kW) on the circuit at 15-minute intervals'
        x_axis_text = 'Time'
        x_hover_template = 'Time: %{x}'
    else:
        title = f'{title} {load_type} demand (kW) on the circuit at 15-minute intervals'
        x_axis_text = 'Datetime'
        x_hover_template = 'Timestamp: %{x|%Y-%m-%d %H:%M}'
    fig = go.Figure()
    # - Add individual traces if desired
    if individual_traces:
        # - Show a particular individual load shape if desired
        #loadshapes_df = loadshapes_df[['load_p1rlv1359_1']] # residential
        #loadshapes_df = loadshapes_df[['load_p1rlv3771_1']] # commercial
        for col in loadshapes_df.columns:
            s = loadshapes_df[col]
            if single_day_aggreation:
                s = s.groupby(lambda ts: ts.time()).mean()
            fig.add_trace(
                go.Scatter(
                    x=s.index,
                    y=s,
                    hovertemplate='<br>'.join([
                        x_hover_template,
                        'Power: %{y} kW',
                        '<extra></extra>']),
                    line=dict(color='gray')))
    # - Add a line that is the sum of all the individual traces
    fig.add_trace(
        go.Scatter(
            x=total_circuit_demand_s.index,
            y=total_circuit_demand_s,
            hovertemplate='<br>'.join([
                x_hover_template,
                'Power: %{y} kW',
                '<extra></extra>']),
            line=dict(color='blue')))
    # - Format the figure
    title_font_size = 20
    show_legend = True
    if output_png:
        title_font_size = 12
        show_legend = False
    fig.update_layout(
        title={'text': title, 'font': {'size': title_font_size}},
        xaxis={'title': {'text': x_axis_text}},
        yaxis={'title': {'text': 'Power (kW)'}},
        legend={'title': {'text': 'Legend Title'}},
        font={'size': 15},
        showlegend=show_legend)
    if output_png:
        fig.write_image('line-graph.png')
    else:
        fig.show()
    

if __name__ == '__main__':
    main()