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
                self.positions[i].append(self.get_current_position(i))
                continue #nwm czy continue czy pass
            if self.next_occupied_edge(i) in list(np.concatenate((self.occupied_edges[:i],self.occupied_edges[i+1:]))):
                pass
            else:
                self.current_steps[i] += 1
                self.occupy_edge(i)
            self.positions[i].append(self.get_current_position(i))

    def simulation(self):
        while not all([self.is_arrived(i) for i in range(len(self.trains))]):
            self.step()


@dataclass
class TrainSimulationAnimation:
    trains:list[DataFrame]
    positions: list[list]
    stops: DataFrame
    def __post_init__(self):
        self.no_trains = len(self.trains)
        self.fig, self.ax = self.create_plot()
        self.no_steps = len(self.positions[0])
        #self.stops_x = dict(zip(self.stops['stop_id'] ,self.stops['stop_lon'] ))
        #self.stops_y = dict(zip(self.stops['stop_id'] ,self.stops['stop_lat'] ))
        self.stops_x = self.stops['stop_lon']
        self.stops_y = self.stops['stop_lat']
        self.trains_id = self.name_to_id()
        self.trains_x = [np.zeros(self.no_steps) for i in range(self.no_trains)]
        self.trains_y = [np.zeros(self.no_steps) for i in range(self.no_trains)]
        #self.colors = [list(np.random.choice(range(256), size=3)) for i in range(self.no_trains)]
        self.eval_plot_positions()

    def create_plot(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        return fig,ax
    
    def name_to_id(self):
        stops = self.stops
        train_with_id = [None for i in range(self.no_trains)]
        for i in range(self.no_trains):
            ids = []
            for j in range(len(self.trains[i])):
                ids.append(int(*stops[stops['stop_name'].str.replace(' ', '') == self.trains[i]['Station Name'][j].replace(' ', '')]['stop_id'].values))
            train_with_id[i] = DataFrame({'stop_id': ids, 'Travel Time': self.trains[i]['Travel Time']})
        return train_with_id

    def eval_plot_positions(self):
        for i in range(self.no_trains):
            for step in range(self.no_steps):
                prev_stop_index = np.floor(self.positions[i][step])
                next_stop_index = np.ceil(self.positions[i][step])
                prev_stop_id = self.trains_id[i]['stop_id'][prev_stop_index]
                next_stop_id = self.trains_id[i]['stop_id'][next_stop_index]
                x_move = self.stops_x[next_stop_id]-self.stops_x[prev_stop_id]
                print(x_move)
                y_move = self.stops_y[next_stop_id]-self.stops_y[prev_stop_id]
                self.trains_x[i][step] = self.stops_x[prev_stop_id]+(self.positions[i][step]-prev_stop_index)*x_move
                self.trains_y[i][step] = self.stops_y[prev_stop_id]+(self.positions[i][step]-prev_stop_index)*y_move


    def draw_stops(self,ax):
        ax.clear()
        ax.plot(self.stops_x,self.stops_y,'o')
        #plt.plot(stops_x,stops_y,'o')

    def draw_train_positions(self,step_number,ax):
        for i in range(self.no_trains):
            ax.plot(self.trains_x[i][step_number], self.trains_y[i][step_number],'o')
        
        #return self.fig
    
    def draw_step(self,step_number,ax):
        self.draw_stops(ax)
        self.draw_train_positions(step_number,ax)

    def animate(self):
        animation = FuncAnimation(self.fig, self.draw_step, fargs=(self.ax,),
                                frames=int(self.no_steps), interval=1000, repeat=False)
        return animation


def replace_spaces(file):
    file.columns = file.columns.str.replace(' ', '')
    return file
def save_anim(animation: FuncAnimation) -> None:
    animation.save('outputs/anim3.gif', writer='imagemagick', fps=5,dpi=200)

if __name__ == '__main__':
    X = pd.read_csv('traces/AF.csv')
    Y = pd.read_csv('traces/BG.csv')
    Z = pd.read_csv('traces/CH.csv')
    A = TrainSimulation([X, Y, Z])
    A.simulation()
    sim_positions = A.positions

    B = TrainSimulationAnimation([X,Y,Z], sim_positions, pd.read_csv('inputs/stops0.txt'))
    save_anim(B.animate())


