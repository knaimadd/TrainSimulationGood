from main_copy import *

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
            occupied = list(np.concatenate((self.occupied_edges[:i],self.occupied_edges[i+1:])))
            cnt = occupied.count(next)
            if self.is_edge(next):
                self.count_max_trains(next, cnt)
            if self.is_edge(next) and cnt == self.capacities[tuple(next)]:
                self.delay[i].append(self.delay[i][-1]+1)
                self.delay_caused[str(tuple(next))] += 1
            else:
                self.current_steps[i] += 1
                self.occupy_edge(i)
                self.delay[i].append(self.delay[i][-1])
            self.positions[i].append(self.get_current_position(i))


if __name__ == '__main__':
    trip_ids = pd.read_csv('inputs/generated_trip.csv')
    trains = [pd.read_csv(f'inputs/routes/{trip_ids.iloc[i][0]}.csv') for i in range(len(trip_ids))]
    n=1
    A = TrainSimulationCount(trains,n, pd.read_csv('inputs/capacities.csv'))
    A.simulation()
