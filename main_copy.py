import os
from dataclasses import dataclass

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from icecream import ic
from pandas import DataFrame
from scipy.interpolate import interp1d
from matplotlib.animation import FuncAnimation
from typing import Any
from collections import defaultdict

from constants import *

@dataclass
class TrainSimulation:
#    stops: DataFrame 
#    stop_times: DataFrame 
    trains: list[DataFrame]

    def __post_init__(self) -> None:
        self.total_times = [sum(train['Travel Time']) for train in self.trains]
        self.last_stops = [len(train['Station Name'])-1 for train in self.trains]
        self.step_positions = self.create_step_positions()
        self.positions = [[0] for i in range(len(self.trains))]
        self.occupied_edges = np.array([None]*len(self.trains))
        self.current_steps = [0]*len(self.trains)

    # stworzenie listy pozycji w kolejnych krokach pociągów bez zatrzymywania
    def create_step_positions(self):
        step_positions = [np.zeros(self.total_times[i]+1) for i in range(len(self.trains))]
        for i in range(len(self.trains)):
            train = self.trains[i]
            distance = 0
            for j in range(len(train['Travel Time'])):
                t = train['Travel Time'][j]
                step_positions[i][distance:distance+t] = [j+k/t for k in range(t)]
                distance += t
            step_positions[i][-1] = j

        return step_positions       

    def occupied_edge(self, train_number):
        prev_station_index = np.floor(self.step_positions[train_number][self.current_steps[train_number]])
        next_station_index = np.ceil(self.step_positions[train_number][self.current_steps[train_number]])
        prev_station = self.trains[train_number]['Station Name'][prev_station_index]
        next_station = self.trains[train_number]['Station Name'][next_station_index]
        return [prev_station,next_station]
    
    def next_occupied_edge(self, train_number):
        prev_station_index = np.floor(self.step_positions[train_number][self.current_steps[train_number]+1])
        next_station_index = np.ceil(self.step_positions[train_number][self.current_steps[train_number]+1])
        prev_station = self.trains[train_number]['Station Name'][prev_station_index]
        next_station = self.trains[train_number]['Station Name'][next_station_index]
        return [prev_station,next_station]

    def occupy_edge(self, train_number):
        self.occupied_edges[train_number] = self.occupied_edge(train_number)

    def is_arrived(self, train_number):
        return self.current_steps[train_number] == self.total_times[train_number]
    
    def get_current_position(self, train_number):
        return self.step_positions[train_number][self.current_steps[train_number]]


    def step(self):
        for i in range(len(self.trains)):
            if self.is_arrived(i):
                self.occupied_edges[i] = None
                continue #nwm czy continue czy pass
            if self.next_occupied_edge(i) in list(np.concatenate((self.occupied_edges[:i],self.occupied_edges[i+1:]))):
                pass
            else:
                self.current_steps[i] += 1
            self.positions[i].append(self.get_current_position(i))


    
    def simulation(self):
        while not all([self.is_arrived(i) for i in range(len(self.trains))]):
            self.step()


            

if __name__ == '__main__':
    X = pd.read_csv('traces/AF.csv')
    Y = pd.read_csv('traces/BG.csv')
    Z = pd.read_csv('traces/CH.csv')


    A = TrainSimulation([X, Y, Z])

    A.simulation()



