import os

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from icecream import ic
from scipy.interpolate import interp1d

os.chdir(os.path.dirname(os.path.abspath(__file__)))

df_trips = pd.read_csv('inputs/trips.txt')  
df_stop_times = pd.read_csv('inputs/stop_times.txt')
df_stops = pd.read_csv('inputs/stops.txt')
df_transfers = pd.read_csv('inputs/transfers.txt')
ic(df_stops)
unique_stops = df_stops[['stop_id', 'stop_name', 'stop_lat', 'stop_lon']].drop_duplicates()
ic(unique_stops)

