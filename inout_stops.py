import pandas as pd
from pandas import DataFrame
import numpy as np

stops_i = pd.read_csv('inputs/stops.txt')
stops_i['stop_id']  = [str(stop)+'i' for stop in stops_i['stop_id']]
stops_i['stop_name']  = [str(stop)+'_i' for stop in stops_i['stop_name']]

stops_o = pd.read_csv('inputs/stops.txt')
stops_o['stop_id']  = [str(stop)+'o' for stop in stops_o['stop_id']]
stops_o['stop_name']  = [str(stop)+'_o' for stop in stops_o['stop_name']]

stops =pd.concat([stops_i,stops_o])
stops.reset_index()

stops.to_csv('inputs/stops_io.txt',index=False)
