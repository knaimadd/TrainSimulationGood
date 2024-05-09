import pandas as pd
from pandas import DataFrame
import numpy as np

def replace_spaces(file):
    file.columns = file.columns.str.replace(' ', '')
    return file

df = pd.read_csv('inputs/stop_times.txt')
replace_spaces(df)

df = df[df['trip_id'].str.startswith('2024-03-11')]
