

import pandas as pd
import numpy as np

df = pd.read_csv('glass_data.csv')

df.columns = [ "id","indice_refrativo","sodio",'Magnesio',"Aluminio","Silicone","Potasion","Calcio","Barium","Ferro","n_columns"]


print(df.head())