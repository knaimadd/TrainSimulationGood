from main_copy import *
    
class TrainSimulationRNG(TrainSimulation):

    def __init__(self, trains, n, capacity_file, likelyhood, intensity):
        super().__init__(trains, n, capacity_file)
        self.likelyhood = likelyhood
        self.intensity = intensity
        self.edge_random_delay = []

    def step(self, arrived, s):
        if len(self.edge_random_delay) > 0:
            for i in range(len(self.edge_random_delay)):
                if self.edge_random_delay[i][1] == 0:
                    self.capacities[self.edge_random_delay[i][0]] = self.edge_random_delay[i][3]
                    self.edge_random_delay.pop(i)
                else:
                    self.edge_random_delay[i][1] -= 1

        p = np.random.rand()
        if p < self.likelyhood:
            edge = np.random.choice(list(self.capacities.keys()))
            self.edge_random_delay.append([edge, int(np.floor(np.random.exponential(self.intensity))), self.capacities[edge]])
            self.capacities[edge] = 0
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
            if self.is_edge(next) and cnt == self.capacities[next]:
                self.delay[i].append(self.delay[i][-1]+1)
                self.delay_caused[next] += 1
            else:
                self.current_steps[i] += 1
                self.occupy_edge(i)
                self.delay[i].append(self.delay[i][-1])
            self.positions[i].append(self.get_current_position(i))

if __name__ == '__main__':
    n = 2
    X = pd.read_csv('traces/AF.csv', index_col=False)
    Y = pd.read_csv('traces/BG.csv', index_col=False)
    Z = pd.read_csv('traces/CH.csv', index_col = False)

    cap = pd.read_csv('inputs/capacities0.csv')
    trains = [X,Y,Z]
    A = TrainSimulationRNG(trains, n, cap, 0.1, 10)
    A.simulation()
    sim_positions = A.positions
    det = TrainSimulation(trains, n, cap)
    det.simulation()

    #B = create_anim(A,pd.read_csv('inputs/stops0.txt'),pd.read_csv('inputs/id_capacities0.csv'))
    #save_anim(B.animate())

