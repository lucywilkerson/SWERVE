import argparse
from swerve import config, cadence
import os

CONFIG = config()

data_dir = CONFIG['dirs']['original']
dat_file = os.path.join(data_dir, 'caraballo', 'gicdata2024.dat.db')
if not os.path.exists(dat_file):
    raise FileNotFoundError(f"Data file not found: {dat_file}")

import struct
import pandas as pd
import numpy as np
from collections import Counter


TS_START = 1715212800  # May 9, 2024
TS_END = 1715558399    # May 12, 2024

print("--- EXECUTING EXACT TIMESTAMP TRACKING AUDIT ---")

found_offsets = []

with open(dat_file, 'rb') as f:
    content = f.read()
    
    # Scan every single byte offset (not stepping by 4) to find the absolute positioning
    for offset in range(0, len(content) - 4):
        val = struct.unpack("<I", content[offset:offset+4])[0]
        if TS_START <= val <= TS_END:
            found_offsets.append(offset)

print(f"Total raw timestamp matches found: {len(found_offsets)}")

if len(found_offsets) > 1:
    # 1. Analyze the exact byte differences between adjacent timestamps
    gaps = [found_offsets[i+1] - found_offsets[i] for i in range(len(found_offsets)-1)]
    common_gaps = Counter(gaps).most_common(5)
    
    print("\nTop 5 absolute byte gaps between timestamps:")
    for gap, count in common_gaps:
        print(f"  Gap of {gap} bytes occurs {count} times")
        
    # 2. Print out the first 5 raw offsets to inspect the start block pattern
    print("\nFirst 5 raw byte locations of May 2024 timestamps:")
    for i, off in enumerate(found_offsets[:5]):
        ts_val = struct.unpack("<I", content[off:off+4])[0]
        print(f"  Match {i}: Byte Offset {off} -> Timestamp: {ts_val} ({pd.to_datetime(ts_val, unit='s')})")
else:
    print("\n[!] No explicit 4-byte little-endian timestamps found in the target window.")


exit()



import sqlite3

# 1. Connect to the database file
conn = sqlite3.connect(dat_file)

# 2. Create a cursor object to execute SQL commands
cursor = conn.cursor()

# 3. (Optional) Find out the names of the tables inside the DB file
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = cursor.fetchall()
print("Tables in database:", tables)

# 4. Read data from a specific table (replace 'your_table_name' with an actual table)
# Let's assume you have a table from the list above
try:
    table_name = tables[0][0] # Grab the first table found
    cursor.execute(f"SELECT * FROM {table_name}")
    
    # Fetch all rows from the result of the query
    rows = cursor.fetchall()
    
    for row in rows:
        print(row)
except IndexError:
    print("The database file has no tables.")

# 5. Always close the connection when finished
conn.close()
