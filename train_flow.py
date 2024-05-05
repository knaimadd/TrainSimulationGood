import networkx as nx
import pandas as pd
import os

# Przygotowanie grafu
G = nx.Graph()

os.chdir(os.path.dirname(os.path.abspath(__file__)))
# Wczytanie danych o przystankach
stops = pd.read_csv('traces/wroclaw_warsaw.csv')
for index, row in stops.iterrows():
    G.add_node(row['stop_id'], name=row['stop_name'])

# Wczytanie połączeń między przystankami (przykładowe, zakładając że mamy takie dane)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 4)

def is_section_free(current_section, occupied_sections):
    return current_section not in occupied_sections

def move_train(train_id, path, occupied_sections, delays):
    print(f"Rozpoczynamy trasę pociągu {train_id}")
    for section in path:
        if is_section_free(section, occupied_sections):
            occupied_sections.append(section)
            print(f"Pociąg {train_id} przemieszcza się do sekcji {section}")
            delays[train_id] = 0
        else:
            print(f"Pociąg {train_id} czeka na zwolnienie sekcji {section}")
            delays[train_id] += 1
            break
        occupied_sections.remove(section)

occupied_sections = []
delays = {1: 0, 2: 0}

# Załóżmy, że pociąg 1 i pociąg 2 mają część wspólną trasy
path_train_1 = [1, 2, 3]
path_train_2 = [2, 3, 4]

# Symulacja ruchu pociągów
move_train(1, path_train_1, occupied_sections, delays)
move_train(2, path_train_2, occupied_sections, delays)

print(f"Opóźnienia pociągów: {delays}")
