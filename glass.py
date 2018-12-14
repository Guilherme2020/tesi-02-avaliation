#!/usr/bin/env python
# coding: utf-8

# In[70]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.decomposition import PCA
from sklearn import metrics

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



# In[ ]:



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


# In[99]:


df = pd.read_csv('glass_data.csv')

df.columns = ["id", "indice_refrativo", "sodio", 'Magnesio', "Aluminio",
              "Silicone", "Potasio", "Calcio", "Barium", "Ferro", "tipo"]

#
# df['tipo'] = df['tipo'].apply(lambda x: (0, 1)[x <= 4])

#df_tipo = df['tipo']
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


# print("Dados estatisticos das colunas: ")
print(media_sodio)
print(max_sodio)


# In[103]:



media_sodio = df.sodio.mean()
max_sodio = df.sodio.max()


print("Dados estatisticos das colunas: ")
print("Media do sodio",media_sodio)
print("Max sodio",max_sodio)
print("tipo",df['tipo'].mean())


# In[20]:


#sns.set(style="whitegrid", font_scale=1.8)
plt.subplots(figsize = (15,8))
#df = pd.read_csv('glass_data.csv')
# ax = plt.plot(df['tipo'])
sns.countplot(x="tipo", data=df).set_title('Count of Glass Types')


# In[75]:


grouBy = df.groupby('tipo',as_index=False).mean()


# In[76]:


grouBy


# In[23]:


sns.set(style="whitegrid", font_scale=1.2)
plt.subplots(figsize=(20, 15))
plt.subplot(3, 3, 1)
sns.boxplot(x='tipo', y='indice_refrativo', data=df)
plt.subplot(3, 3, 2)
sns.boxplot(x='tipo', y='sodio', data=df)
plt.subplot(3, 3, 3)
sns.boxplot(x='tipo', y='Magnesio', data=df)
plt.subplot(3, 3, 4)
sns.boxplot(x='tipo', y='Aluminio', data=df)
plt.subplot(3, 3, 5)
sns.boxplot(x='tipo', y='Silicone', data=df)
plt.subplot(3, 3, 6)
sns.boxplot(x='tipo', y='Potasio', data=df)
plt.subplot(3, 3, 7)
sns.boxplot(x='tipo', y='Calcio', data=df)
plt.subplot(3, 3, 8)
sns.boxplot(x='tipo', y='Barium', data=df)
plt.subplot(3, 3, 9)
sns.boxplot(x='tipo', y='Ferro', data=df)
plt.show()


# In[89]:


# Performing PCA
X = df[['indice_refrativo','sodio','Magnesio','Aluminio','Silicone','Potasio','Calcio','Barium','Ferro']]
pca = PCA(random_state = 1)
pca.fit(X)
variancia_expl = pca.explained_variance_ratio_
cum_var_expl = np.cumsum(variancia_expl)
variancia_df = pd.DataFrame(pca.explained_variance_.round(2), index=["P" + str(i) for i in range(1,10)],
                      columns=["Variancia por Componente"])
print(variancia_df.T)
plt.figure(figsize=(12,7))
plt.bar(range(1,len(cum_var_expl)+1), variancia_expl, align= 'center', label= 'individual variancia', color='teal', alpha = 0.8)
plt.step(range(1,len(cum_var_expl)+1), cum_var_expl, where = 'mid' , label= 'cumulativa variance', color='red')
plt.ylabel('Relacao de variancia Explicada ')
plt.xlabel('Principais componentes')
plt.xticks(np.arange(1,len(variancia_expl)+1,1))
plt.legend(loc='center right')
plt.show()


# In[92]:


pca_red = PCA(n_components=5)
X_reduced = pca_red.fit_transform(X)

x = X_reduced
y = df['tipo'].values

X_train, X_test,y_train,y_test = train_test_split(x,y,test_size=.3,random_state=25)

print(np.unique(y_train))
print(np.unique(y_test))


# In[93]:



# modelo = MultinomialNB()
# modelo.fit(X_train,y_train)


#Random Forest

classificador = RandomForestClassifier(n_estimators=100,max_depth=2,random_state=0)

classificador.fit(X_train,y_train)
classificador.score(X_test,y_test)



#Gaussi
gausiNb = GaussianNB()

gausiNb.fit(X_train,y_train)
gausiNb.score(X_test,y_test)


adaBoost =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

adaBoost.fit(X_train,y_train)
adaBoost.score(X_test,y_test) 


# In[94]:


#Algoritmo SVC

svm = SVC()
svm.fit(X_train,y_train)
y_predict  = svm.predict(X_test)

metrica_svm = metrics.accuracy_score(y_predict,y_test)


#algoritmo Decision Tree
dec_tree = DecisionTreeClassifier()
dec_tree.fit(X_train, y_train)
y_predict = dec_tree.predict(X_test)

metrica_dctree = metrics.accuracy_score(y_predict, y_test)



#algoritmo Random Forest
rand_for = RandomForestClassifier(max_depth=3, min_samples_split=2, n_estimators=50, random_state=1)
rand_for.fit(X_train, y_train)
y_predict = rand_for.predict(X_test)
metric_random = metrics.accuracy_score(y_predict, y_test)


#Adaboost
adaBoost =  AdaBoostClassifier(DecisionTreeClassifier(max_depth=1),
                         algorithm="SAMME",
                         n_estimators=200)

adaBoost.fit(X_train,y_train)
adaBoost.score(X_test,y_test) 
metric_ada = metrics.accuracy_score(y_predict, y_test)



pd.DataFrame([['Support Vector Machine', metrica_svm],
                             ['Decision Tree', metrica_dctree], ['Random Forest', metric_random],['AdaBoost',metric_ada]],
                                 columns=['Model', 'Accuracy'])


# In[95]:


matriz_confusion = confusion_matrix(y_test,y_predict)
plt.subplots(figsize=(12, 8))
sns.heatmap(matriz_confusion.T, square=True, annot=True, fmt='d', cbar=False)
plt.xlabel('label')
plt.ylabel('predicted label')


# In[96]:


print(confusion_matrix(y_test,classificador.predict(X_test)))
print(confusion_matrix(y_test,gausiNb.predict(X_test)))
print(confusion_matrix(y_test,adaBoost.predict(X_test)))


# In[98]:


print(classification_report(y_test,y_predict))

