# EXEMPLO PARA CORRELAÇÃO COM VALIDAÇÃO (Momento de Pearson)
###############################################################
import numpy as np
import pandas as pd
import collections
import warnings

from scipy import stats

#Funcao para correlação usando metodo de pearson
def pearsonr_ci(x,y,alpha=0.05):
   r, p = stats.pearsonr(x, y)
   r_z = np.arctanh(r)
   se = 1 / np.sqrt(x.size - 3)
   z = stats.norm.ppf(1 - alpha / 2)
   lo_z, hi_z = r_z - z * se, r_z + z * se
   lo, hi = np.tanh((lo_z, hi_z))
 
   return r, p, lo, hi

#Funcao verifica se o fator de correlação esta dentro dos parametros esperados MAIOR/MENOR
def checa_correlacao(r, p, lo, hi):
    #base = 0.058865348804654646
    if (r > lo) & (r < hi) :
        #if (r == base):
        val_valida = valida_forca_correlacao(r, lo, hi)
        return r, lo, hi, val_valida

    return 0, 0, 0, 0


#Funcao verifica a força da correlação entre os campos
def valida_forca_correlacao(r, lo, hi):        

    neutro = (lo+ ((hi-lo) / 2))
    parte = (neutro - lo) / 3

    #print("NEUTRO ORIGINAL ", neutro, "PARTE ",parte)

    menor = lo  
    menor2 = menor + parte
    menor1 = menor2 + parte  
    neutro =  menor1 + parte
    maior1 = neutro + parte  
    maior2 =  maior1 + parte
    maior = hi

    #print("MENOR ", menor)  
    #print("MENOR2 ", menor2)
    #print("MENOR 1", menor1)
    #print("NEUTRO", neutro)
    #print("MAIOR 1", maior1)
    #print("MAIOR 2", maior2)
    #print("MAIOR", maior)
    
    if(r >= menor) & (r < menor2):      
        return "RELACAO FORTE NEGATIVO"
    elif(r >= menor2) & (r < neutro):  
        return "RELACAO FRACA NEGATIVO"
    elif(r <= maior) & (r >= maior2):   
        return "RELACAO FORTE POSITIVO"
    elif(r < maior2) & (r >= neutro):   
        return "RELACAO FRACA POSITIVO"


#Pega as colunas especificas do pm25
df = pd.read_csv("base/pm25.csv", sep=";", usecols=["year","month","day","hour","DEWP","TEMP","PRES","Iws","Is","Ir"])

#Buscando a correlção
c1 = 0
for i, j in df.iteritems():
    c2 = 0
    for i2, j2 in df.iteritems():
        r, p, lo, hi = pearsonr_ci(df[i], df[i2])

        val_r, val_lo, val_hi, val_valida = checa_correlacao(r, p, lo, hi)
        if(val_r > 0) & (i != i2):
            print(i,"/",i2," => ", val_r, val_lo, val_hi, val_valida)

        c2 += 1

    c1 += 1