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
    capacities_file: DataFrame

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
        self.current_steps = np.zeros(len(self.trains), dtype=int)
        self.capacities = self.get_capacities()
        self.arrived_step = np.zeros(len(self.trains))
        self.previous = [() for i in range(len(self.trains))]
        # Statystyki
        self.delay = [[0] for i in range(len(self.trains))]
        self.delay_caused = self.initial_delay()
        self.no_arrived = []
        self.no_stopped = []
        self.count_edge = self.initial_delay()

    def reset(self):
        self.positions = [[0] for i in range(len(self.trains))]
        self.occupied_edges = np.array([None]*len(self.trains))
        self.current_steps = np.zeros(len(self.trains), dtype=int)
        self.delay = [[0] for i in range(len(self.trains))]
    
    def initial_delay(self):
        df = self.capacities_file
        cap = {}
        for i in range(len(df['Edge'])):
            edge = eval(df['Edge'][i])
            if (edge[0].endswith(('_i', '_o'))) and (edge[0][:-2] != edge[1][:-2]):
                edge = (edge[0][:-2], edge[1][:-2])
            inv = (edge[1], edge[0])
            if inv in cap:
                cap[inv] = min(df['Capacity'][i], 0)
            else:
                cap[edge] = 0
        return cap

    def get_capacities(self): 
        df = self.capacities_file
        cap = {}
        for i in range(len(df['Edge'])):
            edge = eval(df['Edge'][i])
            if (edge[0].endswith(('_i', '_o'))) and (edge[0][:-2] != edge[1][:-2]):
                edge = (edge[0][:-2], edge[1][:-2])
            inv = (edge[1], edge[0])
            if inv in cap:
                cap[inv] = min(df['Capacity'][i], cap[inv])
            else:
                cap[edge] = df['Capacity'][i]
        return cap

    def create_starttime(self): #czas startu
        trains_starttime = [None for train in self.trains]
        for i in range(self.no_trains):
            if len(str(self.trains[i]['Travel Time'].iloc[-1])) ==3:
                trains_starttime[i] = pd.to_datetime('0'+str(self.trains[i]['Travel Time'].iloc[-1])[:1]+':'+str(self.trains[i]['Travel Time'].iloc[-1])[1:])
            elif len(str(self.trains[i]['Travel Time'].iloc[-1]))==2:
                trains_starttime[i] = pd.to_datetime('00:'+str(self.trains[i]['Travel Time'].iloc[-1]))
            else:
                trains_starttime[i] = pd.to_datetime(str(self.trains[i]['Travel Time'].iloc[-1])[:2]+':'+str(self.trains[i]['Travel Time'].iloc[-1])[2:])
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
        if prev_station[:-2] != next_station[:-2]:
            prev_station = prev_station[:-2]
            next_station = next_station[:-2]
        return (prev_station,next_station)
    
    def next_occupied_edge(self, train_number):
        prev_station_index = np.floor(self.step_positions[train_number][self.current_steps[train_number]+1])
        next_station_index = np.ceil(self.step_positions[train_number][self.current_steps[train_number]+1])
        prev_station = self.trains[train_number]['Station Name'][prev_station_index]
        next_station = self.trains[train_number]['Station Name'][next_station_index]
        if prev_station[:-2] != next_station[:-2]:
            prev_station = prev_station[:-2]
            next_station = next_station[:-2]
        return (prev_station,next_station)

    def occupy_edge(self, train_number, edge):
        e = edge
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

    def step(self, arrived, s): 
        for i in range(len(self.trains)):
            if arrived[i]:
                self.occupied_edges[i] = None
                self.positions[i].append(self.get_current_position(i))
                self.delay[i].append(self.delay[i][-1])
                if self.arrived_step[i] == 0:
                    self.arrived_step[i] = s
                continue #nwm czy continue czy pass
            next = self.next_occupied_edge(i)
            if next not in self.capacities:
                next = (next[1], next[0])
            am_i_edge = self.is_edge(next)
            if am_i_edge and next != self.previous[i]:
                self.count_edge[next] += 1
            occupied = list(np.concatenate((self.occupied_edges[:i],self.occupied_edges[i+1:])))
            cnt = occupied.count(next)
            if am_i_edge and cnt == self.capacities[next]:
                self.delay[i].append(self.delay[i][-1]+1)
                self.delay_caused[next] += 1
            else:
                self.current_steps[i] += 1
                self.occupy_edge(i, next)
                self.delay[i].append(self.delay[i][-1])
            self.previous[i] = next
            self.positions[i].append(self.get_current_position(i))

    def simulation(self):
        self.reset()
        arrived = [self.is_arrived(i) for i in range(len(self.trains))]
        s = 0
        while not all(arrived):
            self.no_arrived.append(sum(arrived))
            self.no_stopped.append(sum([self.trains_startstep[i] > s for i in range(len(self.trains))]))
            self.step(arrived, s)
            arrived = [self.is_arrived(i) for i in range(len(self.trains))]
            s += 1



@dataclass
class TrainSimulationAnimation:
    trains:list[DataFrame]
    positions: list[list]
    stops: DataFrame
    n: int
    starttime: Any
    edges: DataFrame
    arrived_step: list
    starstep:list
    delays: list
    def __post_init__(self):
        self.starttime =self.starttime
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
        self.capacities = self.edges["Capacity"]
        self.edges_start_xy, self.edges_end_xy,self.edge_width = self.eval_edges()
        #self.colors = [list(np.random.choice(range(256), size=3)) for i in range(self.no_trains)]
        self.arrived_step_fix()
        self.eval_plot_positions()
        self.delays_colors = self.eval_colors()
        self.border = pd.read_csv('inputs/border.csv')

    def arrived_step_fix(self):
        for i in range(self.no_trains):
            if self.arrived_step[i] ==0:
                self.arrived_step[i] = self.no_steps
    def eval_colors(self):
        self.max_delay = max([max(self.delays[i]) for i in range(self.no_trains)])
        no_delay = (0,0.58,0)
        delays_gb = 0.62
        if self.max_delay == 0:
            return [no_delay]
        else:
            color_step = delays_gb/self.max_delay
            delays = [(1,0.62-color_step*i,0.62-color_step*i) for i in range(self.max_delay)]
            return [no_delay]+delays

    def create_plot(self):
        fig, ax = plt.subplots(figsize=(10, 8))
        return fig,ax
    
    def name_to_id(self):
        stops = self.stops
        train_with_id = [None for i in range(self.no_trains)]
        for i in range(self.no_trains):
            ids = []
            for j in range(len(self.trains[i])):
                #ids.append(int(*stops[stops['stop_name'].str.replace(' ', '') == self.trains[i]['Station Name'][j].replace(' ', '')]['stop_id'].values))
                ids.append(*stops[stops['stop_name'].str.replace(' ', '') == self.trains[i]['Station Name'][j].replace(' ', '')]['stop_id'].values)
            train_with_id[i] = DataFrame({'stop_id': ids, 'Travel Time': self.trains[i]['Travel Time']})
        return train_with_id
    
    def edges_name_to_id(self):
        edges = self.edges
        edges_with_id = [None for i in range(len(edges))]
        for i in range(len(edges)):
            ids = []
            for j in range(len(self.trains[i])):
                ids.append(*edges[edges['Edge'].str.replace(' ', '') == self.trains[i]['Station Name'][j].replace(' ', '')]['stop_id'].values)
                #ids.append(int(*edges[edges['Edge'].str.replace(' ', '') == self.trains[i]['Station Name'][j].replace(' ', '')]['stop_id'].values))
            edges_with_id[i] = DataFrame({'stop_id': ids, 'Travel Time': self.trains[i]['Travel Time']})
        return edges_with_id

    def eval_plot_positions(self):
        for i in range(self.no_trains):
            for step in range(self.no_steps):
                prev_stop_index = np.floor(self.positions[i][step])
                next_stop_index = np.ceil(self.positions[i][step])
                prev_stop_id = self.trains_id[i]['stop_id'][prev_stop_index]
                next_stop_id = self.trains_id[i]['stop_id'][next_stop_index]
                prev_stop_xy = self.stops[self.stops['stop_id']==prev_stop_id][['stop_lon','stop_lat']].values[0]
                next_stop_xy = self.stops[self.stops['stop_id']==next_stop_id][['stop_lon','stop_lat']].values[0]
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
        ax.set_xticks([])
        ax.set_yticks([])
        #plt.plot(stops_x,stops_y,'o')
    
    
    
    def eval_edges(self):
        edges_start = self.edges["Edge_start"]
        edges_end = self.edges["Edge_end"]
        edges_start_xy = [self.stops[self.stops['stop_id']==edges_start[i]][['stop_lon','stop_lat']].iloc[0] for i in range(len(edges_start))]
        edges_end_xy = [self.stops[self.stops['stop_id']==edges_end[i]][['stop_lon','stop_lat']].iloc[0] for i in range(len(edges_end))]
        cap_min = min(self.capacities)
        cap_max = max(self.capacities)
        w_max = 2
        w_min = 0.5
        if cap_max == cap_min:
            width = [0.7 for i in self.capacities]
        else:
            a = (w_max-w_min)/(cap_max-cap_min)
            b = w_min-a*cap_min
            width = [a*capacity+b for capacity in self.capacities]
        return edges_start_xy,edges_end_xy,width
    
    def draw_border(self,ax):
        ax.plot(self.border['0'],self.border['1'],color='black',linewidth=0.8,alpha=0.8)
   

    def draw_edges(self,ax):
        for i in range(len(self.edges)):
            ax.plot([self.edges_start_xy[i].values[0],self.edges_end_xy[i].values[0]],[self.edges_start_xy[i].values[1],self.edges_end_xy[i].values[1]],color="black",alpha=0.3,linewidth=self.edge_width[i])

    def draw_train_positions(self,step_number,ax):
        for i in range(self.no_trains):
            if step_number >= self.starstep[i] and step_number <= self.arrived_step[i]:
                ax.plot(self.trains_x[i][step_number], self.trains_y[i][step_number],'o',color=self.delays_colors[self.delays[i][step_number]])
        
        #return self.fig
    
    def draw_step(self,step_number,ax):
        self.draw_stops(ax)
        self.draw_edges(ax)
        self.draw_border(ax)
        #print(str(datetime.timedelta(seconds=60/self.n*step_number)))
        ax.set_title(str(self.starttime+datetime.timedelta(seconds=60/self.n*step_number))[11:])
        self.draw_train_positions(step_number,ax)

    def animate(self):
        animation = FuncAnimation(self.fig, self.draw_step, fargs=(self.ax,),
                                frames=int(self.no_steps), interval=1000, repeat=False)
        return animation


def replace_spaces(file):
    file.columns = file.columns.str.replace(' ', '')
    return file
def save_anim(animation: FuncAnimation,name='anim3.gif',fps=30) -> None:
    animation.save(f'outputs/{name}', writer='imagemagick', fps=fps,dpi=100)
def multiply(file, n):
    file.loc[:(len(file['Travel Time'])-2), 'Travel Time'] = n*file.loc[:(len(file['Travel Time'])-2), 'Travel Time']

def create_anim(A,stops,id_capacities):
    B =TrainSimulationAnimation(A.trains,A.positions,stops,A.n,A.start,id_capacities,A.arrived_step,A.trains_startstep,A.delay)
    return B

if __name__ == '__main__':
    """n = 10
    X = pd.read_csv('traces/AF.csv', index_col=False)
    #X = pd.read_csv('traces/katowice_poznan.csv')
    #multiply(X, n)
    Y = pd.read_csv('traces/BG.csv', index_col=False)
    #Y = pd.read_csv('traces/wroclaw_warsaw2.csv')
    #multiply(Y, n)
    Z = pd.read_csv('traces/CH.csv')
    #A = TrainSimulation([X, Y, Z])
    cap = pd.read_csv('inputs/capacities0.csv')
    trains = [X,Y,Z]
    A = TrainSimulation(trains,n,cap)
    A.simulation()
    sim_positions = A.positions"""


    trip_ids = pd.read_csv('inputs/generated_trip.csv')
    trains = [pd.read_csv(f'inputs/routes/{trip_ids.iloc[i][0]}.csv') for i in range(len(trip_ids))]
    n=1
    A = TrainSimulation(trains,n, pd.read_csv('inputs/capacities_estimated.csv'))
    A.simulation()

    print(A.count_edge)
    B = create_anim(A,pd.concat([pd.read_csv('inputs/stops.txt'), pd.read_csv('inputs/stops_io.txt')]),pd.read_csv('inputs/id_capacities.csv'))
    save_anim(B.animate(), fps=30)

