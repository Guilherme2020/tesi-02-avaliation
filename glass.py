import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC

# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier


# from sklearn.metrics import confusion_matrix


# Descrição do DataSet
# nome: Glass Identification ( Identificação de Vidro)
# Volume de dados: 213 amostras
# Descrição:

# O estudo da classificação dos tipos de vidro foi motivado pela investigação criminológica, pois fornece
# evidências  nessas investigações forenses
# Na cena do crime, o vidro deixado pode ser usado como prova se for corretamente identificado.
# O conjunto de dados é comumente usado para demonstrar algoritmos de classificação de aprendizado de máquina em ambientes acadêmicos.

# Objetivo

# O objetivo é prever com base na análise dos componentes quimicos
# em qual classe o tipo de vidro se encontra.
"""
Type of glass: (class attribute)
      -- 1 building_windows_float_processed
      -- 2 building_windows_non_float_processed
      -- 3 vehicle_windows_float_processed
      -- 4 vehicle_windows_non_float_processed (none in this database)
      -- 5 containers
      -- 6 tableware
      -- 7 headlamps
Class Distribution: (out of 214 total instances)
    -- 163 Window glass (building windows and vehicle windows)
       -- 87 float processed  
          -- 70 building windows
          -- 17 vehicle windows
       -- 76 non-float processed
          -- 76 building windows
          -- 0 vehicle windows
    -- 51 Non-window glass
       -- 13 containers
       -- 9 tableware
       -- 29 headlamps
"""
# Colunas
#1 - ID
# 2 - Indice Refreativo 
#3 - Sodio
#4 - Magnesio
#5 - Aluminio
#6 -  Silicone
#7 - Potassio
#8 - Calcio
#9 - Barium
#10 - Ferro
#11 - Tipo

df = pd.read_csv('glass_data.csv')

df.columns = ["id", "indice_refrativo", "sodio", 'Magnesio', "Aluminio",
              "Silicone", "Potasion", "Calcio", "Barium", "Ferro", "tipo"]

#
df['tipo'] = df['tipo'].apply(lambda x: (0, 1)[x <= 4])

# Campos nulos: Nenhum
# nulos = df.isna().sum()
# print("Campos nulos: ",nulos)

# tipos_colunas
#print("Tipos colunas: ",tipos_colunas)
#   df.dtypes

# Linha x Colunas
print(df.shape)

print(df.head(100))
#info:
print("Info: ",df.info( ))
# Descricao
describe = df.describe()
print(describe)

media_sodio = df.sodio.mean()
max_sodio = df.sodio.max()


# # print("Dados estatisticos das colunas: ")
# print(media_sodio)
# print(max_sodio)

