from main_copy import *
    
class TrainSimulationRNG(TrainSimulation):

    def __init__(self, trains, n, capacity_file, likelyhood, intensity):
        super().__init__(trains, n, capacity_file)
        # ten pierwiastek z 3 zupełnie na oko wyznaczony do normowania, trzeba dla innej symulacji sprawdzić czy będzie ok
        self.likelyhood = [likelyhood*(1-i/len(self.trains))*np.sqrt(3) for i in range(len(self.trains))][::-1]
        self.intensity = intensity
        self.edge_random_delay = []
        self.resistant = []
        self.direct_random = [[0] for i in range(len(self.trains))]

    def reset(self):
        self.positions = [[0] for i in range(len(self.trains))]
        self.occupied_edges = np.array([None]*len(self.trains))
        self.current_steps = np.zeros(len(self.trains), dtype=int)
        self.delay = [[0] for i in range(len(self.trains))]
        self.resistant = []
        self.direct_random = [[0] for i in range(len(self.trains))]

    def step(self, arrived, s):
        if len(self.edge_random_delay) > 0:
            edges_delays = np.ones(len(self.edge_random_delay))
            for i in range(len(self.edge_random_delay)):
                if self.edge_random_delay[i][1] == 0:
                    self.capacities[self.edge_random_delay[i][0]] = self.edge_random_delay[i][2]
                    edges_delays[i] = 0
                else:
                    self.edge_random_delay[i][1] -= 1
            self.edge_random_delay = [self.edge_random_delay[i] for i in np.where(edges_delays)[0]]

        not_none_occupied = [x for x in self.occupied_edges if x is not None]
        no_occupied = len(not_none_occupied)
        if no_occupied > 0:
            p = np.random.rand()
            if p < self.likelyhood[no_occupied]:
                edge = not_none_occupied[np.random.randint(0, no_occupied)]
                if edge not in self.resistant:
                    self.resistant.append(edge)
                    self.edge_random_delay.append([edge, int(np.floor(np.random.exponential(self.intensity))), self.capacities[edge]])
                    self.capacities[edge] = 0
        for i in range(len(self.trains)):
            if arrived[i]:
                self.occupied_edges[i] = None
                self.positions[i].append(self.get_current_position(i))
                self.delay[i].append(self.delay[i][-1])
                self.direct_random[i].append(self.direct_random[i][-1])
                if self.arrived_step[i] == 0:
                    self.arrived_step[i] = s
                continue #nwm czy continue czy pass
            next = self.next_occupied_edge(i)
            if next not in self.capacities:
                next = (next[1], next[0])
            occupied = list(np.concatenate((self.occupied_edges[:i],self.occupied_edges[i+1:])))
            cnt = occupied.count(next)
            if self.is_edge(next) and cnt >= self.capacities[next]:
                self.delay[i].append(self.delay[i][-1]+1)
                self.delay_caused[next] += 1
                if self.capacities[next] == 0:
                    self.direct_random[i].append(self.direct_random[i][-1]+1)
                else:
                    self.direct_random[i].append(self.direct_random[i][-1])        
            else:
                self.current_steps[i] += 1
                self.occupy_edge(i, next)
                self.delay[i].append(self.delay[i][-1])
                self.direct_random[i].append(self.direct_random[i][-1])
            self.positions[i].append(self.get_current_position(i))

def adjust_length(lists):
    m = len(max(lists, key=len))
    for i in range(len(lists)):
        last = lists[i][-1]
        length = len(lists[i])
        lists[i].extend([last]*(m-length))

if __name__ == '__main__':
    n = 1
    #X = pd.read_csv('traces/AF.csv', index_col=False)
    #Y = pd.read_csv('traces/BG.csv', index_col=False)
    #Z = pd.read_csv('traces/CH.csv', index_col = False)

    cap = pd.read_csv('inputs/capacities_estimated.csv')
    #trains = [X,Y,Z]
    trip_ids = pd.read_csv('inputs/generated_trip.csv')
    trains = [pd.read_csv(f'inputs/routes/{trip_ids.iloc[i][0]}.csv') for i in range(len(trip_ids))]
    A = TrainSimulationRNG(trains, n, cap, 0.3, 20)

    N = 10
    m = 0
    delay_times = [[] for i in range(N)]
    delay_random = [[] for i in range(N)]
    for i in range(N):
        A.simulation()
        delay_times[i] = A.delay
        delay_random[i] = A.direct_random
        print(f'symulacja {i}')
        m = max(m, len(max(A.delay, key=len)))
            
    # wydłużanie do wspólnej długości
    summed_delays = [[] for i in range(N)]
    summed_delays_random = [[] for i in range(N)]
    for i in range(N):
        for j in range(len(delay_times[i])):
            last = delay_times[i][j][-1]
            length = len(delay_times[i][j])
            delay_times[i][j].extend([last]*(m-length))
            lastRNG = delay_random[i][j][-1]
            lengthRNG = len(delay_random[i][j])
            delay_random[i][j].extend([lastRNG]*(m-lengthRNG))
        summed_delays[i] = np.sum(np.array(delay_times[i]), axis=0)
        summed_delays_random[i] = np.sum(np.array(delay_random[i]), axis=0)
    summed_delays = np.array(summed_delays)

    
    #A.simulation()

    #B = create_anim(A,pd.read_csv('inputs/stops_io.txt'),pd.read_csv('inputs/id_capacities.csv'))
    #save_anim(B.animate())

