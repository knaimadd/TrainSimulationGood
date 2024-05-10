from main_copy import *
import pandas as pd

trip_ids = pd.read_csv('inputs/generated_trip.csv')
trains = [pd.read_csv(f'inputs/routes/{trip_ids.iloc[i][0]}.csv') for i in range(len(trip_ids))]
n=1
A = TrainSimulation(trains,n, pd.read_csv('inputs/capacities.csv'))
A.simulation()

sim_positions = A.positions

#B = TrainSimulationAnimation(trains, sim_positions, pd.read_csv('inputs/stops.txt'), n,A.start,pd.read_csv('inputs/id_capacities.csv'),A.arrived_step,A.trains_startstep)
#save_anim(B.animate())

B = create_anim(A,pd.read_csv('inputs/stops_io.txt'),pd.read_csv('inputs/id_capacities.csv'))
save_anim(B.animate())