import pandas as pd
import numpy as np
#import os
#cur_path = os.path.dirname(__file__)

data = pd.read_csv('Data/Fine-tuning/dataset.csv',delimiter=';')

for index, row in data.iterrows():
    id = row['id']
    string_to_write = ""
    for col in data.columns:
        if row[col] is not np.nan:
            string_to_write += f"{col}:{row[col]}\n"
    
    with open(f"patient{id}.txt",'a') as f:
        f.write(string_to_write)
