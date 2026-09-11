# Dictionary of NERC events. Datetimes are ISO-8601 strings so this module has
# no runtime dependency on the configuration object's datetime imports.
nerc_events = {
    '2024-05-10': {
        'nerc_prefix': '2024E04',
        'data_limits': ['2024-05-10T15:00', '2024-05-12T06:00'],
        'plot_limits': ['2024-05-10T11:00', '2024-05-12T06:00'],
        'paper_dir': '../../2024-May-Storm-paper',
        'sid_duplicates': {
            '10197': 'Sullivan', '10204': 'Shelby', '10208': 'Rutherford',
            '10203': 'Raccoon Mountain', '10212': 'Pinhook',
            '10201': 'Montgomery', '10660': 'Gleason', '10200': 'East Point',
            '10207': 'Bull Run',
        },
        'paper_sids': {
            'GIC': {
                'timeseries': {'Bull Run': 'a)', 'Montgomery': 'c)', 'Union': 'e)', 'Widows Creek': 'g)'},
                'scatter': {'Bull Run': 'b)', 'Montgomery': 'd)', 'Union': 'f)', 'Widows Creek': 'h)'},
            },
            'B': {
                'timeseries': {'Bull Run': 'a)', '50116': 'c)'},
                'scatter': {'Bull Run': 'b)', '50116': 'd)'},
            },
        },
    },
    '2013-05-31': {'nerc_prefix': '2013E01', 'data_limits': ['2013-05-31T16:00', '2013-06-01T18:00'], 'plot_limits': ['2013-05-31T14:00', '2013-06-01T18:00']},
    '2013-10-02': {'nerc_prefix': '2013E02', 'data_limits': ['2013-10-02T01:00', '2013-10-02T20:00'], 'plot_limits': ['2013-10-01T23:00', '2013-10-02T20:00']},
    '2015-03-17': {'nerc_prefix': '2015E01', 'data_limits': ['2015-03-17T04:00', '2015-03-18T12:00'], 'plot_limits': ['2015-03-17T02:00', '2015-03-18T12:00']},
    '2015-06-22': {'nerc_prefix': '2015E02', 'data_limits': ['2015-06-22T05:00', '2015-06-23T14:00'], 'plot_limits': ['2015-06-22T03:00', '2015-06-23T14:00']},
    '2015-09-11': {'nerc_prefix': '2015E03', 'data_limits': ['2015-09-11T04:00', '2015-09-12T18:00'], 'plot_limits': ['2015-09-11T02:00', '2015-09-12T18:00']},
    '2015-09-19': {'nerc_prefix': '2015E04', 'data_limits': ['2015-09-19T22:00', '2015-09-20T18:00'], 'plot_limits': ['2015-09-19T20:00', '2015-09-20T18:00']},
    '2015-10-06': {'nerc_prefix': '2015E05', 'data_limits': ['2015-10-07T00:00', '2015-10-08T18:00'], 'plot_limits': ['2015-10-06T22:00', '2015-10-08T18:00']},
    '2015-10-07': {'nerc_prefix': '2015E05', 'data_limits': ['2015-10-07T00:00', '2015-10-08T18:00'], 'plot_limits': ['2015-10-06T22:00', '2015-10-08T18:00']},
    '2015-12-20': {'nerc_prefix': '2015E06', 'data_limits': ['2015-12-20T04:00', '2015-12-21T10:00'], 'plot_limits': ['2015-12-20T02:00', '2015-12-21T10:00']},
    '2017-05-27': {'nerc_prefix': '2017E01', 'data_limits': ['2017-05-27T21:00', '2017-05-28T18:00'], 'plot_limits': ['2017-05-27T19:00', '2017-05-28T18:00']},
    '2017-09-07': {'nerc_prefix': '2017E02', 'data_limits': ['2017-09-07T23:00', '2017-09-08T18:00'], 'plot_limits': ['2017-09-07T21:00', '2017-09-08T18:00']},
    '2017-09-27': {'nerc_prefix': '2017E03', 'data_limits': ['2017-09-27T18:00', '2017-09-28T18:00'], 'plot_limits': ['2017-09-27T16:00', '2017-09-28T18:00']},
    '2018-08-25': {'nerc_prefix': '2018E01', 'data_limits': ['2018-08-25T20:00', '2018-08-26T20:00'], 'plot_limits': ['2018-08-25T18:00', '2018-08-26T20:00']},
    '2021-05-12': {'nerc_prefix': '2021E01', 'data_limits': ['2021-05-12T06:00', '2021-05-13T04:00'], 'plot_limits': ['2021-05-12T04:00', '2021-05-13T04:00']},
    '2021-11-03': {'nerc_prefix': '2021E02', 'data_limits': ['2021-11-03T19:00', '2021-11-04T14:00'], 'plot_limits': ['2021-11-03T17:00', '2021-11-04T14:00']},
    '2023-03-23': {'nerc_prefix': '2023E02', 'data_limits': ['2023-03-23T10:00', '2023-03-24T12:00'], 'plot_limits': ['2023-03-23T08:00', '2023-03-24T12:00']},
    '2023-04-23': {'nerc_prefix': '2023E03', 'data_limits': ['2023-04-24T00:00', '2023-04-24T20:00'], 'plot_limits': ['2023-04-23T22:00', '2023-04-24T20:00']},
    '2023-04-24': {'nerc_prefix': '2023E03', 'data_limits': ['2023-04-24T00:00', '2023-04-24T20:00'], 'plot_limits': ['2023-04-23T22:00', '2023-04-24T20:00']},
    '2024-03-23': {'nerc_prefix': '2024E01', 'data_limits': ['2024-03-24T12:00', '2024-03-25T00:00'], 'plot_limits': ['2024-03-24T10:00', '2024-03-25T00:00']},
    '2024-08-11': {'nerc_prefix': '2024E07', 'data_limits': ['2024-08-11T06:00', '2024-08-12T18:00'], 'plot_limits': ['2024-08-11T04:00', '2024-08-12T18:00']},
    '2024-10-07': {'nerc_prefix': '2024E10', 'data_limits': ['2024-10-07T12:00', '2024-10-08T12:00'], 'plot_limits': ['2024-10-07T10:00', '2024-10-08T12:00']},
    '2024-10-10': {'nerc_prefix': '2024E11', 'data_limits': ['2024-10-10T14:00', '2024-10-11T14:00'], 'plot_limits': ['2024-10-10T12:00', '2024-10-11T14:00']},
    '2024-12-31': {'nerc_prefix': '2024E12', 'data_limits': ['2024-12-31T15:00', '2025-01-02T00:00'], 'plot_limits': ['2024-12-31T13:00', '2025-01-02T00:00']},
    '2025-04-15': {'nerc_prefix': '2025E01', 'data_limits': ['2025-04-15T15:00', '2025-04-17T09:00'], 'plot_limits': ['2025-04-15T13:00', '2025-04-17T09:00']},
    '2025-06-01': {'nerc_prefix': '2025E02', 'data_limits': ['2025-06-01T00:00', '2025-06-03T12:00'], 'plot_limits': ['2025-05-31T22:00', '2025-06-03T12:00']},
}

def set_storm_limits(event):
    # Use Dst from HAPI to determine storm start and stop times
    from hapiclient import hapi, hapitime2datetime
    from datetime import datetime, timedelta
    import pandas

    if isinstance(event, str):
        event_dt = datetime.strptime(event, '%Y-%m-%d')
    else:
        event_dt = event
    
    server     = 'https://cdaweb.gsfc.nasa.gov/hapi'
    dataset    = 'OMNI2_H0_MRG1HR'  
    parameters = 'T1800,Mgs_mach_num1800,DST1800,AE1800,AL_INDEX1800,AU_INDEX1800'
    start      = (event_dt - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00Z')
    stop       = (event_dt + timedelta(days=5)).strftime('%Y-%m-%dT00:00:00Z')
    data_hapi, meta = hapi(server, dataset, parameters, start, stop, logging=True)

    parameters_dict = {
        'Time': 'Time',
        'DST1800': 'Dst'
      }

    dfs = []
    for param, name in parameters_dict.items():
        if param == 'Time':
            dfs.append(pandas.DataFrame(hapitime2datetime(data_hapi[param])))
        else:
            dfs.append(pandas.DataFrame(data_hapi[param]))
    
    df = pandas.concat(dfs, axis=1)
    df.columns = list(parameters_dict.values())

    #find time of max dst before dst drops to min Dst
    min_idx = df['Dst'].idxmin()
    df_before_min = df.loc[:min_idx]
    max_dst_time = df_before_min.loc[df_before_min['Dst'].idxmax(), 'Time']

    import matplotlib.pyplot as plt
    from datetick import datetick
    plt.figure()
    plt.plot(df['Time'], df['Dst'], color='k')
    plt.axvline(max_dst_time, color='m', linestyle='--', label='Max Dst')
    plt.axvline(datetime.strptime(nerc_events[event]['data_limits'][0], '%Y-%m-%dT%H:%M').replace(tzinfo=timezone.utc), color='c', linestyle=':', label='Data Limit Start')
    diff = datetime.strptime(nerc_events[event]['data_limits'][0], '%Y-%m-%dT%H:%M').replace(tzinfo=timezone.utc) - max_dst_time
    plt.scatter([], [], facecolors='none', edgecolors='none', label=f'Difference: {diff.total_seconds()/3600:.2f} hours')
    plt.grid()
    plt.legend()
    plt.ylabel('Dst [nT]')
    datetick('x')
    plt.savefig(f'./dst_check_plots/{event}_dst.png', dpi=300, bbox_inches='tight')
    plt.show()
    exit()
    return max_dst_time

diffs = []
from datetime import datetime, timezone, timedelta
for event in nerc_events.keys():
    """if event == '2024-10-07':
        continue"""
    max_dst_time = set_storm_limits(event)
    diff = datetime.strptime(nerc_events[event]['data_limits'][0], '%Y-%m-%dT%H:%M').replace(tzinfo=timezone.utc) - max_dst_time
    diffs.append(diff)
mean_diff = sum(diffs, timedelta(0)) / len(diffs)
print(f'mean difference between data_limits[0] and max_dst_time: {mean_diff}')
#print(diffs)