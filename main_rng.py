from main_copy import *


@dataclass
class TrainSimulationRNG:
    trains: list[DataFrame]
    n: int
    likelyhood: float
    intensity: float

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
        self.random_delays = np.zeros(len(self.trains))
        # Statystyki
        self.delay = [[0] for i in range(len(self.trains))]
        self.delay_caused = self.initial_delay()
        self.no_arrived = []
        self.no_stopped = []

    def reset(self):
        self.positions = [[0] for i in range(len(self.trains))]
        self.occupied_edges = np.array([None]*len(self.trains))
        self.current_steps = np.zeros(len(self.trains), dtype=int)
        self.delay = [[0] for i in range(len(self.trains))]
    
    def initial_delay(self):
        df = pd.read_csv('inputs/capacities0.csv')
        init_delay = {df['Edge'][i]: 0 for i in range(len(df['Edge']))}
        return init_delay

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

    def random_delay(self):
        rng = np.random.default_rng()
        p = rng.random()
        if self.likelyhood >= p:
            return int(np.floor(rng.exponential(self.intensity)))
        return 0
    
    def step(self, arrived):
        for i in range(len(self.trains)):
            if arrived[i]:
                self.occupied_edges[i] = None
                self.positions[i].append(self.get_current_position(i))
                continue #nwm czy continue czy pass

            if self.random_delays[i] == 0 and np.random.rand() <= self.intensity:
                self.random_delays[i] = self.random_delay()
            
            next = self.next_occupied_edge(i)
            occupied = list(np.concatenate((self.occupied_edges[:i],self.occupied_edges[i+1:])))
            cnt = occupied.count(next)
            is_next_edge = self.is_edge(next)
            if is_next_edge and cnt == self.capacities[str(tuple(next))]:
                self.delay[i].append(self.delay[i][-1]+1)
                self.delay_caused[str(tuple(next))] += 1
            elif is_next_edge and self.random_delays[i] > 0:
                self.delay[i].append(self.delay[i][-1]+1)
                self.delay_caused[str(tuple(next))] += 1
                self.random_delays[i] -= 1
            else:
                self.current_steps[i] += 1
                self.occupy_edge(i)
            self.positions[i].append(self.get_current_position(i))

    def simulation(self):
        self.reset()
        arrived = [self.is_arrived(i) for i in range(len(self.trains))]
        s = 0
        while not all(arrived):
            self.no_arrived.append(sum(arrived))
            self.no_stopped.append(sum([self.trains_startstep[i] > s for i in range(len(self.trains))]))
            self.step(arrived)
            arrived = [self.is_arrived(i) for i in range(len(self.trains))]
            s += 1

if __name__ == '__main__':
    n = 2
    X = pd.read_csv('traces/AF.csv', index_col=False)
    Y = pd.read_csv('traces/BG.csv', index_col=False)
    Z = pd.read_csv('traces/CH.csv', index_col = False)

    trains = [X,Y,Z]
    A = TrainSimulationRNG(trains, n, 0.1, 10)
    A.simulation()
    sim_positions = A.positions
    det = TrainSimulation(trains, n)
    det.simulation()

    B = TrainSimulationAnimation(trains, sim_positions, pd.read_csv('inputs/stops0.txt'), n,A.start,pd.read_csv('inputs/id_capacities0.csv'))
    save_anim(B.animate())

