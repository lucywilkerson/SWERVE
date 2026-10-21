# Looks for data in SWERVE-data/data_processed between start and stop times
# Then writes available data to Numpy array of form [datetime, value]
# Given data type, data class, data source, start time, and stop time

import numpy as np
import os
import pandas as pd