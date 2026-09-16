test = False # if True, runs tests to compare manually determined storm times
# to automated determined storm times for NERC storms

def storm_time(event, test=False):
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
    start      = (event_dt).strftime('%Y-%m-%dT00:00:00Z')
    stop       = (event_dt + timedelta(days=3)).strftime('%Y-%m-%dT00:00:00Z')
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

    # Find time of max dst before dst drops to min Dst
    min_idx = df['Dst'].idxmin()
    df_before_min = df.loc[:min_idx]
    max_dst_time = df_before_min.loc[df_before_min['Dst'].idxmax(), 'Time']

    # Find time of min dst/2 after dst drops to min Dst
    half_min_dst = df.loc[min_idx, 'Dst']/2
    df_after_min = df.loc[min_idx:]
    half_min_dst_time = df_after_min[df_after_min['Dst'] >= half_min_dst]['Time'].iloc[0]

    # Make times timezone naive for consistency
    max_dst_time = max_dst_time.replace(tzinfo=None)
    half_min_dst_time = half_min_dst_time.replace(tzinfo=None)

    if test:
        #plotting for testing:
        import matplotlib.pyplot as plt
        from datetick import datetick
        import os
        plt.figure()
        plt.plot(df['Time'], df['Dst'], color='k')
        plt.axvline(max_dst_time, color='m', linestyle='--', label='Automated data limits')
        plt.axvline(datetime.strptime(events[event]['data_limits'][0], '%Y-%m-%dT%H:%M'), color='c', linestyle=':', label='Manual data limits')
        diff = datetime.strptime(events[event]['data_limits'][0], '%Y-%m-%dT%H:%M') - max_dst_time
        plt.scatter([], [], facecolors='none', edgecolors='none', label=f'Difference in start: {diff.total_seconds()/3600:.2f} hours')
        plt.axvline(half_min_dst_time, color='m', linestyle='--')
        plt.axvline(datetime.strptime(events[event]['data_limits'][1], '%Y-%m-%dT%H:%M'), color='c', linestyle=':')
        diff = datetime.strptime(events[event]['data_limits'][1], '%Y-%m-%dT%H:%M') - half_min_dst_time
        plt.scatter([], [], facecolors='none', edgecolors='none', label=f'Difference in end: {diff.total_seconds()/3600:.2f} hours')
        plt.grid()
        plt.legend()
        plt.ylabel('Dst [nT]')
        plt.title(event)
        datetick('x')
        file_dir = os.path.join(os.path.dirname(__file__), '..', 'dst_check_plots')
        if not os.path.exists(file_dir):
            os.makedirs(file_dir)
        print(f"Saving Dst plot for event {event} to {file_dir}")
        plt.savefig(os.path.join(file_dir, f'{event}_dst.png'), dpi=300, bbox_inches='tight')
        #plt.show()

    return [max_dst_time, half_min_dst_time]

if test:
    import json
    import os
    with open(os.path.join(os.path.dirname(__file__), '..', 'swerve', 'config_nerc.json'), 'r') as f:
        events_config = json.load(f)
    events = events_config
    start_diffs = []
    end_diffs = []
    from datetime import datetime, timezone, timedelta
    for event in events.keys():
        max_dst_time, half_min_dst_time = storm_time(event, test=test)
        start_diff = datetime.strptime(events[event]['data_limits'][0], '%Y-%m-%dT%H:%M') - max_dst_time
        start_diffs.append(start_diff)
        end_diff = datetime.strptime(events[event]['data_limits'][1], '%Y-%m-%dT%H:%M') - half_min_dst_time
        end_diffs.append(end_diff)
    mean_diff = sum(start_diffs, timedelta(0)) / len(start_diffs)
    print(f'mean difference between data_limits[0] and max_dst_time: {mean_diff.total_seconds()/3600:.2f} hours')
    mean_diff = sum(end_diffs, timedelta(0)) / len(end_diffs)
    print(f'mean difference between data_limits[1] and half_min_dst_time: {mean_diff.total_seconds()/3600:.2f} hours')
    # histogram of diffs
    import matplotlib.pyplot as plt
    import os
    plt.figure()
    diff_hours = [diff.total_seconds() / 3600 for diff in start_diffs]
    hour_bins = range(int(min(diff_hours)), int(max(diff_hours)) + 2)
    plt.hist(diff_hours, bins=hour_bins, color='c', edgecolor='c', alpha=0.6, label='start times')
    diff_hours = [diff.total_seconds() / 3600 for diff in end_diffs]
    hour_bins = range(int(min(diff_hours)), int(max(diff_hours)) + 2)
    plt.hist(diff_hours, bins=hour_bins, color='y', edgecolor='y', alpha=0.6, label='end times')
    plt.xlabel('Difference [hours]')
    plt.ylabel('Count')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(os.path.dirname(__file__), '..', 'dst_check_plots', 'diff_histogram.png'), dpi=300, bbox_inches='tight')