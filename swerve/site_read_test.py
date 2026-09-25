import os
import csv
import numpy
import pandas
import pickle
import datetime

from swerve import config
from swerve.site_read import _site_read_orig

run_tests = True

CONFIG = config()
logger = CONFIG['logger'](**CONFIG['logger_kwargs'])

# Test functions for all readers in site_read!

def _test_AlvesRibeiro_GIC_measured():
    data_source = 'AlvesRibeiro'
    data_type = 'GIC'
    data_class = 'measured'
    sid = 'Paraimo'
    event = '2021-09-17'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('16/09/2021 00:00', '%d/%m/%Y %H:%M')
    raw_start_data = [0.075586081]
    assert read_start_time == raw_start_time and read_start_data == raw_start_data

def _test_AlvesRibeiro_GIC_calculated():
    data_source = 'AlvesRibeiro'
    data_type = 'GIC'
    data_class = 'calculated'
    sid = 'Paraimo'
    event = '2021-09-17'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('16/09/2021 00:00', '%d/%m/%Y %H:%M')
    raw_start_data = [0.040223038]
    assert read_start_time == raw_start_time and read_start_data == raw_start_data

def _test_Blake_GIC_measured():
    data_source = 'Blake'
    data_type = 'GIC'
    data_class = 'measured'
    sid = 'Woodland'
    event = '2016-03-06'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('2016-03-06', '%Y-%m-%d')
    raw_start_data = [0.0333]
    assert read_start_time == raw_start_time and read_start_data == raw_start_data

def _test_Blake_GIC_calculated():
    data_source = 'Blake'
    data_type = 'GIC'
    data_class = 'calculated'
    sid = 'Woodland'
    event = '2016-03-06'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('2016-03-06', '%Y-%m-%d')
    raw_start_data = [0.0037]
    assert read_start_time == raw_start_time and read_start_data == raw_start_data

def _test_Espinosa_GIC_measured():
    data_source = 'Espinosa'
    data_type = 'GIC'
    data_class = 'measured'
    sid = 'Itumbiara'
    event = '2013-10-08'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('08-Oct-2013 00:00:00', '%d-%b-%Y %H:%M:%S')
    raw_start_data = -0.00915
    # Using numpy.isclose because Excel file reader adds small diff
    assert read_start_time == raw_start_time and numpy.isclose(read_start_data[0], raw_start_data)

def _test_Marsal2021_DMM_measured():
    data_source = 'Marsal2021'
    data_type = 'DMM'
    data_class = 'measured'
    sid = 'TRA line'
    event = '2021-03-24'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('2021 03 24 00 00 00', '%Y %m %d %H %M %S')
    raw_start_data = numpy.array([24983.567,-35.650,37609.175])
    assert read_start_time == raw_start_time and numpy.array_equal(read_start_data, raw_start_data)

def _test_Marsal2025_DMM_measured():
    data_source = 'Marsal2025'
    data_type = 'DMM'
    data_class = 'measured'
    sid = 'TRA line'
    event = '2023-04-23'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('2023 04 23 00 00 37', '%Y %m %d %H %M %S')
    raw_start_data = numpy.array([25750.733,31.041,37277.206])
    assert read_start_time == raw_start_time and numpy.array_equal(read_start_data, raw_start_data)

def _test_NERC_GIC_measured():
    data_source = 'NERC'
    data_type = 'GIC'
    data_class = 'measured'
    sid = '10052'
    event = '2024-05-10'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('05/10/2024 12:00:00 AM', '%m/%d/%Y %I:%M:%S %p')
    raw_start_data = 1.55
    assert read_start_time == raw_start_time and read_start_data == raw_start_data

def _test_NERC_B_measured():
    data_source = 'NERC'
    data_type = 'B'
    data_class = 'measured'
    sid = '50100'
    event = '2024-05-10'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('05/10/2024 12:00:00 AM', '%m/%d/%Y %I:%M:%S %p')
    raw_start_data = numpy.array([23829.33,-671.19,40090.02])
    assert read_start_time == raw_start_time and numpy.array_equal(read_start_data, raw_start_data)

def _test_Parry2024_DMM_measured():
    data_source = 'Parry2024'
    data_type = 'DMM'
    data_class = 'measured'
    sid = 'Alberta line'
    event = '2021-10-12'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('20211012000000', '%Y%m%d%H%M%S')
    raw_start_data = numpy.array([16005.519,-68.122,55409.691])
    assert read_start_time == raw_start_time and numpy.array_equal(read_start_data, raw_start_data)

def _test_Parry2024_GIC_measured():
    from datetime import timezone
    from zoneinfo import ZoneInfo
    data_source = 'Parry2024'
    data_type = 'GIC'
    data_class = 'measured'
    sid = 'Ellerslie 1'
    event = '2021-10-12'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time_utc = datetime.datetime.strptime('10/11/2021 18:00', '%m/%d/%Y %H:%M').replace(tzinfo=ZoneInfo("America/Denver")).astimezone(timezone.utc)
    raw_start_time = raw_start_time_utc.replace(tzinfo=None)
    raw_start_data = [-5.517001]
    assert read_start_time == raw_start_time and read_start_data == raw_start_data

def _test_Parry2025_GIC_measured():
    data_source = 'Parry2025'
    data_type = 'GIC'
    data_class = 'measured'
    sid = 'Alberta 1'
    event = '2023-04-24'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('2023/04/24 00:00:00', '%Y/%m/%d %H:%M:%S')
    raw_start_data = [-2.049549818000000]
    assert read_start_time == raw_start_time and read_start_data == raw_start_data

def _test_Parry2025_B_measured():
    data_source = 'Parry2025'
    data_type = 'B'
    data_class = 'measured'
    sid = 'FCHP'
    event = '2023-04-24'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('20230424000000', '%Y%m%d%H%M%S')
    raw_start_data = numpy.array([11332.991,3537.687,56488.975])
    assert read_start_time == raw_start_time and numpy.array_equal(read_start_data, raw_start_data)

def _test_TVA_GIC_measured():
    data_source = 'TVA'
    data_type = 'GIC'
    data_class = 'measured'
    sid = 'Bull Run'
    event = '2024-05-10'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('05/10/2024 12:00:00 AM', '%m/%d/%Y %I:%M:%S %p')
    raw_start_data = [-0.2]
    assert read_start_time == raw_start_time and read_start_data == raw_start_data

def _test_TVA_B_measured():
    data_source = 'TVA'
    data_type = 'B'
    data_class = 'measured'
    sid = 'Bull Run'
    event = '2024-05-10'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('05/09/2024 12:00:00 AM', '%m/%d/%Y %I:%M:%S %p')
    raw_start_data = numpy.array([15609.5,-1063.38,46980.79])
    assert read_start_time == raw_start_time and numpy.array_equal(read_start_data, raw_start_data)

def _test_TVA_GIC_calculated():
    data_source = 'TVA'
    data_type = 'GIC'
    data_class = 'calculated'
    sid = 'Bull Run'
    event = '2024-05-10'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('05/10/2024 12:00:00 AM', '%m/%d/%Y %I:%M:%S %p')
    raw_start_data = [0.193325]
    assert read_start_time == raw_start_time and read_start_data == raw_start_data

def _test_Zhang2020_GIC_measured():
    data_source = 'Zhang2020'
    data_type = 'GIC'
    data_class = 'measured'
    sid = 'Huangmeishan'
    event = '2015-12-14'
    # Reading in one site
    read_data = _site_read_orig(sid, data_type, data_class, data_source, event, logger)
    read_start_time = read_data['time'][0]
    read_start_data = read_data['data'][0]
    # Comparing to values in site file
    raw_start_time = datetime.datetime.strptime('2015  12  14       000000', '%Y  %m  %d       %H%M%S')
    raw_start_data = [0.06385]
    assert read_start_time == raw_start_time and read_start_data == raw_start_data


if run_tests:
    _test_AlvesRibeiro_GIC_measured()
    _test_AlvesRibeiro_GIC_calculated()

    _test_Blake_GIC_measured()
    _test_Blake_GIC_calculated()

    _test_Espinosa_GIC_measured()

    _test_Marsal2021_DMM_measured()

    _test_Marsal2025_DMM_measured()

    _test_NERC_GIC_measured()
    _test_NERC_B_measured()

    _test_Parry2024_DMM_measured()
    _test_Parry2024_GIC_measured()

    _test_Parry2025_GIC_measured()
    _test_Parry2025_B_measured()

    _test_TVA_GIC_measured()
    _test_TVA_B_measured()
    _test_TVA_GIC_calculated()

    _test_Zhang2020_GIC_measured()