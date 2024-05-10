import pandas as pd
from pandas import DataFrame
import numpy as np

def replace_spaces(file):
    file.columns = file.columns.str.replace(' ', '')
    return file

def get_all_edges(trains):
    all_edges = set()
    for train in trains:
        edges = [(train['Station Name'][i], train['Station Name'][i+1]) for i in range(len(train['Station Name']) - 1)]
        all_edges.update(edges)
    return list(all_edges)

def name_to_id(trains,stops):
        train_with_id = [None for i in range(len(trains))]
        for i in range(len(trains)):
            ids = []
            for j in range(len(trains[i])):
                #ids.append(int(*stops[stops['stop_name'].str.replace(' ', '') == trains[i]['Station Name'][j].replace(' ', '')]['stop_id'].values))
                ids.append(*stops[stops['stop_name'].str.replace(' ', '') == trains[i]['Station Name'][j].replace(' ', '')]['stop_id'].values)

            train_with_id[i] = DataFrame({'Station Name': ids, 'Travel Time': trains[i]['Travel Time']})
        return train_with_id

if __name__ == '__main__':
    X = pd.read_csv('traces/AF.csv')
    Y = pd.read_csv('traces/BG.csv')
    Z = pd.read_csv('traces/CH.csv')



trip_ids = pd.read_csv('inputs/generated_trip.csv')
trains = [pd.read_csv(f'inputs/routes/{trip_ids.iloc[i][0]}.csv') for i in range(len(trip_ids))]
stops = pd.read_csv('inputs/stops_io.txt')
replace_spaces(stops)

#trains = [X, Y, Y]


edges = get_all_edges(trains)
capacity = np.ones(len(edges), dtype=int)*10


data = {"Edge": edges, "Capacity": capacity}
df = DataFrame(data)
df.to_csv('inputs/capacities.csv')

trains_with_id = name_to_id(trains,stops)
edges2=get_all_edges(trains_with_id)
data2 = {"Edge_start": [edges2[i][0] for i in range(len(edges2))], "Edge_end":[edges2[i][1] for i in range(len(edges2))], "Capacity": capacity}

df2 = DataFrame(data2)
df2.to_csv('inputs/id_capacities.csv')