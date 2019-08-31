import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import pickle
from random import randint

    
def geraArrayRandom(n):
	r_array = []
	previous = 0

	for i in range(n):
		value = randint(0, 1)

		if previous == 1 & i % 2 == 1:
			value = 0
		
		previous = value
		r_array.append(value)

	return r_array

model_file = 'models/pm25_kmeans.sav'

kmeans_vote = pickle.load(open(model_file, 'rb'))

result = kmeans_vote.predict(
    [
        [2010,1,2,0,-16,-4,1020.0,1.79,0,0,0,0,1,0,1,0,0,0],
        [2014,1,2,0,-10,-5,120.0,1.05,0,0,0,0,1,0,0,1,0,0]
    ]
 )

print(result)