import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.linear_model import LogisticRegression
#from sklearn.model_selection import train_test_split

df = pd.read_csv("heart.csv")

'''Tratamento dos dados'''
#print(df.head())


#print(df.shape)

#df.info()

#print(df.describe())

##Visualizar valores vazios 
#print(df.isnull().sum()) #resultado não tem valores nulos

'''Parece que os dados já estão tratados'''

'''Exploração de Dados'''

##contando os valores de sim e não da coluna target(alvo)
print(df.target.value_counts())

## gráfico de contagem (count plot) que mostra a distribuição das classes(sim e não) na coluna "target" 
#sns.countplot(x="target", data=df, hue="target", palette="bwr", legend=False)
#plt.xlabel("Target (0 = Não, 1= Sim)")
#plt.show()

#calcula e imprime a porcentagem de pacientes que têm e não têm doença cardíaca
'''
countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(df.target))*100))) '''

## gráfico de contagem (count plot) que mostra a distribuição das classes(female e male) na coluna "sex" 
#sns.countplot(x="sex", data=df, hue="sex", palette="mako_r", legend=True)
#plt.xlabel("Sex (0 = female, 1= male)")
#plt.show()