# EXEMPLO PARA CORRELAÇÃO COM VALIDAÇÃO (Momento de Pearson)
###############################################################
import numpy as np
import pandas as pd
import collections
import warnings
import pickle

from sklearn.cluster import KMeans
from scipy import stats

#Pega as colunas especificas
df = pd.read_csv("base/pm25.csv", sep=";")
df_normalize = pd.get_dummies(df)
#print(df_normalize)
#df_normalize.to_csv("C:\\Users\\JhonyZam\\Desktop\\UNIVALI\\Escobar - Aprendizado de maquina\\python\\pm25\\base\\df_normalize.csv", index = None, header=True)

kmeans = KMeans(n_clusters=4).fit(df_normalize)

# Salvar modelo
# Finalizar o modelo
filename = 'models/pm25_kmeans.sav'
pickle.dump(kmeans, open(filename, 'wb'))
