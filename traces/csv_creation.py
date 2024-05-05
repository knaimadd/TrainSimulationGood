import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

route_stations = [
    "Katowice",
    "Chorzów Batory",
    "Zabrze",
    "Gliwice",
    "Lubliniec",
    "Olesno Śląskie",
    "Kluczbork",
    "Byczyna Kluczborska",
    "Kępno",
    "Ostrzeszów",
    "Ostrów Wielkopolski",
    "Pleszew",
    "Jarocin",
    "Środa Wielkopolska",
    "Poznań Główny"
]
#travel_times_minutes = [5, 10, 8, 39, 20, 15, 13, 17, 14, 18, 18, 14, 20, 23, 0]
travel_times_minutes = [5, 10, 8, 39, 80, 15, 13, 17, 14, 180, 18, 14, 20, 23, 0]
header = "Station Name,Travel Time\n"
with open('katowice_poznan.csv', 'w', encoding='utf-8-sig') as file:
    file.write(header)
    for station, travel_time in zip(route_stations, travel_times_minutes): 
        file.write(f"{station},{travel_time}\n")
