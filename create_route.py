import pandas as pd
from pandas import DataFrame
import numpy as np
import os
import glob

files = glob.glob('inputs/routes/*')
for file in files:
    os.remove(file)

def replace_spaces(file):
    file.columns = file.columns.str.replace(' ', '')
    return file

stops = pd.read_csv('inputs/stops.txt')

stoptimes = pd.read_csv('inputs/stop_times.txt')
replace_spaces(stops)
replace_spaces(stoptimes)
stoptimes = stoptimes[stoptimes['trip_id'].str.startswith('2024-03-11')]
stoptimes = stoptimes[stoptimes['departure_time'] < '24:00']
#stoptimes = stoptimes[stoptimes['departure_time']>'14:00']
trip_ids = stoptimes['trip_id'].unique()

stoptimes = pd.merge(stoptimes,stops,on="stop_id",how="left").drop(["stop_lon","stop_lat","stop_IBNR"],axis=1)
#stoptimes['departure_time']=pd.to_datetime(stoptimes['departure_time'])
#stoptimes['arrival_time']=pd.to_datetime(stoptimes['arrival_time'])

def timedelta_to_minutes(timedelta):
    return timedelta.components[2]+timedelta.components[1]*60



#for i in range(len(trip_ids)):
for i in range(10):
    route = stoptimes[stoptimes['trip_id']==trip_ids[i]]
    route = route.reset_index()
    starttime = route['arrival_time'].iloc[0]
    starttime = starttime[:2]+starttime[3:5]
    route['departure_time']=pd.to_datetime(route['departure_time'])
    route['arrival_time']=pd.to_datetime(route['arrival_time'])
    times_at_stop = [route['departure_time'][j]-route['arrival_time'][j] for j in range(len(route))]
    times_at_stop = [timedelta_to_minutes(time) for time in times_at_stop]
    motion_times = [route['arrival_time'][j+1]-route['departure_time'][j] for j in range(len(route)-1)]
    motion_times = [timedelta_to_minutes(time) for time in motion_times]
    motion_times.append(starttime)
    route_data = {"Station Name": route['stop_name'], "Travel Time": motion_times}
    route_data = DataFrame(route_data)
    route_data.to_csv(f'inputs/routes/{trip_ids[i]}.csv',index=False)

DataFrame({'trip_id':trip_ids[:i+1]}).to_csv('inputs/generated_trip.csv',index=False)




