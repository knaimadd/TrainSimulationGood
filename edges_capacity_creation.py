import pandas as pd
from pandas import DataFrame
import numpy as np

def get_all_edges(trains):
    all_edges = set()
    for train in trains:
        edges = [(train['Station Name'][i], train['Station Name'][i+1]) for i in range(len(train['Station Name']) - 1)]
        all_edges.update(edges)
    return list(all_edges)


if __name__ == '__main__':
    X = pd.read_csv('traces/AF.csv')
    Y = pd.read_csv('traces/BG.csv')
    Z = pd.read_csv('traces/CH.csv')

trains = [X, Y, Z]

edges = get_all_edges(trains)
capacity = np.ones(len(edges), dtype=int)

data = {"Edge": edges, "Capacity": capacity}

df = DataFrame(data)
df.to_csv('inputs/capacities0.csv')