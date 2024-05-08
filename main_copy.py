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
import datetime

from constants import *


def multiply_time(file, n):
    file.loc[:(len(file['Travel Time'])-2), 'Travel Time'] = n*file.loc[:(len(file['Travel Time'])-2), 'Travel Time']
    return file

def get_all_edges(trains):
    all_edges = set()
    for train in trains:
        edges = [(train['Station Name'][i], train['Station Name'][i+1]) for i in range(len(train['Station Name']) - 1)]
        all_edges.update(edges)
    return all_edges


@dataclass
class TrainSimulation:
#    stops: DataFrame 
#    stop_times: DataFrame 
    trains: list[DataFrame]
    n: int

    def __post_init__(self) -> None:
        self.trains = [self.trains[i].copy() for i in range(len(self.trains))]
        self.trains = [multiply_time(train,self.n) for train in self.trains]
        self.no_trains = len(self.trains)
        self.last_stops = [len(train['Station Name'])-1 for train in self.trains]
        self.trains_starttime = self.create_starttime()
        self.start = min(self.trains_starttime)
        self.create_0lasttime()
        self.total_times = [sum(train['Travel Time']) for train in self.trains]
        self.trains_startstep = self.create_startstep()
        self.step_positions = self.create_step_positions()
        self.positions = [[0] for i in range(len(self.trains))]
        self.occupied_edges = np.array([None]*len(self.trains))
        self.current_steps = [0]*len(self.trains)
        self.capacities = self.get_capacities()

    def get_capacities(self): 
        df = pd.read_csv('inputs/capacities0.csv')
        cap = {df['Edge'][i]: df['Capacity'][i] for i in range(len(df['Edge']))}
        return cap

    def create_starttime(self): #czas startu
        trains_starttime = [pd.to_datetime(str(train['Travel Time'].iloc[-1])[:2]+':'+str(train['Travel Time'].iloc[-1])[2:]) for train in self.trains]
        return trains_starttime
    
   
    def create_0lasttime(self): #zamiana czasu startu na 0 w csv
        for i in range(self.no_trains):
            self.trains[i].loc[len(self.trains[i])-1, "Travel Time"] = 0

    def create_startstep(self): #krok startowy
        differences =  [(self.trains_starttime[i]-self.start).components[1:3] for i in range(self.no_trains)]
        startstep = [(differences[i][0]*60+differences[i][1])*self.n for i in range(self.no_trains)]
        return startstep
    
    # stworzenie listy pozycji w kolejnych krokach pociągów bez zatrzymywania
    def create_step_positions(self):
        start_delays = self.trains_startstep
        step_positions = [np.zeros(self.total_times[i]+1+start_delays[i]) for i in range(len(self.trains))]
        for i in range(len(self.trains)):
            train = self.trains[i]
            distance = 0
            for j in range(len(train['Travel Time'])):
                t = train['Travel Time'][j]
                step_positions[i][distance+start_delays[i]:distance+t+start_delays[i]] = [j+k/t for k in range(t)]
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
        e = self.occupied_edge(train_number)
        if self.is_edge(e):
            self.occupied_edges[train_number] = e
        else:
            self.occupied_edges[train_number] = None
    
    def not_on_edge(self, train_number):
        self.occupied_edges[train_number] = None

    def is_arrived(self, train_number):
        return self.current_steps[train_number] == self.total_times[train_number]+self.trains_startstep[train_number]
    
    def get_current_position(self, train_number):
        return self.step_positions[train_number][self.current_steps[train_number]]
    
    def is_edge(self, v):
        return v[0] != v[1]

    def step(self):
        for i in range(len(self.trains)):
            if self.is_arrived(i):
                self.occupied_edges[i] = None
                self.positions[i].append(self.get_current_position(i))
                continue #nwm czy continue czy pass
            next = self.next_occupied_edge(i)
            occupied = list(np.concatenate((self.occupied_edges[:i],self.occupied_edges[i+1:])))
            cnt = occupied.count(next)
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
    starttime: Any
    def __post_init__(self):
        self.starttime = str(self.starttime)[11:16]
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
                prev_stop_xy = self.stops[self.stops['stop_id']==prev_stop_id][['stop_lon','stop_lat']].iloc[0]
                next_stop_xy = self.stops[self.stops['stop_id']==next_stop_id][['stop_lon','stop_lat']].iloc[0]
                #x_move = self.stops_x[next_stop_id]-self.stops_x[prev_stop_id]
                x_move = next_stop_xy[0]-prev_stop_xy[0]
                y_move = next_stop_xy[1]-prev_stop_xy[1]
                #y_move = self.stops_y[next_stop_id]-self.stops_y[prev_stop_id]
                #self.trains_x[i][step] = self.stops_x[prev_stop_id]+(self.positions[i][step]-prev_stop_index)*x_move
                #self.trains_y[i][step] = self.stops_y[prev_stop_id]+(self.positions[i][step]-prev_stop_index)*y_move
                self.trains_x[i][step] = prev_stop_xy[0]+(self.positions[i][step]-prev_stop_index)*x_move
                self.trains_y[i][step] = prev_stop_xy[1]+(self.positions[i][step]-prev_stop_index)*y_move


    def draw_stops(self,ax):
        ax.clear()
        ax.plot(self.stops_x,self.stops_y,'.')
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
    animation.save('outputs/anim3.gif', writer='imagemagick', fps=24,dpi=200)
def multiply(file, n):
    file.loc[:(len(file['Travel Time'])-2), 'Travel Time'] = n*file.loc[:(len(file['Travel Time'])-2), 'Travel Time']

if __name__ == '__main__':
    n = 2
    X = pd.read_csv('traces/AF.csv')
    #X = pd.read_csv('traces/katowice_poznan.csv')
    #multiply(X, n)
    Y = pd.read_csv('traces/BG.csv')
    #Y = pd.read_csv('traces/wroclaw_warsaw2.csv')
    #multiply(Y, n)
    Z = pd.read_csv('traces/CH.csv')
    #multiply(Z, n)
    #A = TrainSimulation([X, Y, Z])
    trains = [X,Y,Y]
    A = TrainSimulation(trains,n)
    A.simulation()
    sim_positions = A.positions
    
    B = TrainSimulationAnimation(trains, sim_positions, pd.read_csv('inputs/stops0.txt'), A.start)
#    save_anim(B.animate())


