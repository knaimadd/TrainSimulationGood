import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Definiowanie grafu
G = nx.Graph()
G.add_edges_from([("Warszawa", "Kraków"), ("Warszawa", "Wrocław"), ("Kraków", "Wrocław"), ("Wrocław", "Poznań"), ("Gdańsk", "Warszawa")])

# Parametry symulacji
liczba_krokow = 10  # Całkowita liczba kroków symulacji

# Stan początkowy symulacji
pociagi = {
    "Pociąg 1": "Warszawa",
    "Pociąg 2": "Kraków"
}

# Funkcja do wizualizacji stanu grafu
def rysuj_stan(G, pociagi, krok):
    plt.figure(figsize=(10, 7))
    pos = nx.spring_layout(G)  # Układ wierzchołków grafu
    nx.draw(G, pos, with_labels=True, node_size=2000, node_color="lightblue", font_size=10, font_weight="bold")
    
    # Rysowanie pociągów na ich aktualnych pozycjach
    for pociag, pozycja in pociagi.items():
        nx.draw_networkx_nodes(G, pos, nodelist=[pozycja], node_size=2500, node_color="red")
        nx.draw_networkx_labels(G, pos, labels={pozycja: pociag}, font_color="white")
    
    plt.title(f"Krok symulacji: {krok}")
    plt.show()

# Symulacja
for krok in range(1, liczba_krokow + 1):
    # Przemieszczanie pociągów
    for pociag, pozycja in pociagi.items():
        # Losowe wybieranie nowej pozycji z sąsiadów aktualnej pozycji
        nowa_pozycja = np.random.choice(list(G.neighbors(pozycja)))
        pociagi[pociag] = nowa_pozycja
    
    # Wizualizacja stanu symulacji
    rysuj_stan(G, pociagi, krok)
