

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt


# from sklearn.model_selection import train_test_split
# from sklearn.naive_bayes import MultinomialNB
# from sklearn.ensemble import RandomForestClassifier

# from sklearn.ensemble import AdaBoostClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier


# from sklearn.metrics import confusion_matrix



df = pd.read_csv('glass_data.csv')

df.columns = [ "id","indice_refrativo","sodio",'Magnesio',"Aluminio","Silicone","Potasion","Calcio","Barium","Ferro","tipo"]


df['tipo'] = df['tipo'].apply(lambda x : (0,1)[x <= 4])



# nulos = df.isna().sum()

# tipos_colunas = df.dtypes

media_sodio = df.sodio.mean()
max_sodio = df.sodio.max()
describe = df.describe()

#print(df.head(100))
print(df.shape)
# print("Campos nulos: ",nulos)

# # print("Dados estatisticos das colunas: ")
# print(describe)
# print(media_sodio)
# print(max_sodio)
# #print("Tipos colunas: ",tipos_colunas)