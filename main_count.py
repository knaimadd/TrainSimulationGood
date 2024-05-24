from main_copy import *
from edges_capacity_creation import *

class TrainSimulationCount(TrainSimulation):
    
    def __init__(self, trains, n, capacity_file):
        super().__init__(trains, n, capacity_file)
        self.max_capacity_counter = self.initial_delay()

    def count_max_trains(self, edge, cnt):
        if cnt + 1 > self.max_capacity_counter[tuple(edge)]:
            self.max_capacity_counter[tuple(edge)] = cnt + 1

    def step(self, arrived, s):
        for i in range(len(self.trains)):
            if arrived[i]:
                self.occupied_edges[i] = None
                self.positions[i].append(self.get_current_position(i))
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
            if self.is_edge(next):
                self.count_max_trains(next, cnt)
            if self.is_edge(next) and cnt == self.capacities[tuple(next)]:
                self.delay[i].append(self.delay[i][-1]+1)
                self.delay_caused[str(tuple(next))] += 1
            else:
                self.current_steps[i] += 1
                self.occupy_edge(i, next)
                self.delay[i].append(self.delay[i][-1])
            self.positions[i].append(self.get_current_position(i))

def key_to_id(key, stops):
    e1 = stops[stops['stop_name'].str.replace(' ', '') == key[0].replace(' ', '')]['stop_id'].values[0]
    e2 = stops[stops['stop_name'].str.replace(' ', '') == key[1].replace(' ', '')]['stop_id'].values[0]
    return e1, e2
    
def keys_to_ids(keys, stops):
    ids = []
    for key in keys:
        ids.append(key_to_id(key, stops))
    return ids

if __name__ == '__main__':
    trip_ids = pd.read_csv('inputs/generated_trip.csv')
    trains = [pd.read_csv(f'inputs/routes/{trip_ids.iloc[i][0]}.csv') for i in range(len(trip_ids))]
    stops = pd.concat([pd.read_csv('inputs/stops.txt'), pd.read_csv('inputs/stops_io.txt')])
    n=1
    A = TrainSimulationCount(trains,n, pd.read_csv('inputs/capacities.csv'))
    A.simulation()
    
    cap = A.max_capacity_counter
    edges = list(A.max_capacity_counter.keys())
    vals = list(A.max_capacity_counter.values())
    cap = DataFrame({'Edge': edges, 'Capacity': vals})
    cap.to_csv('inputs/capacties_estimated.csv')
    print(A.count_edge)

    
    edges2 = keys_to_ids(edges, stops)
    #trains_with_id = name_to_id(trains,stops)

    #edges2=get_all_edges(trains_with_id)

    data2 = {"Edge_start": [edges2[i][0] for i in range(len(edges2))], "Edge_end":[edges2[i][1] for i in range(len(edges2))], "Capacity": vals}

    df2 = DataFrame(data2)
    df2.to_csv('inputs/id_capacities_estimated.csv')
    
    
