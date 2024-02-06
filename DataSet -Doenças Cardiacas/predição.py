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
#print(df.isnull().sum()) #RESULTADO: não tem valores nulos

#Verificar dados duplicados
#dupl=df.duplicated().sum() #RESULTADO: Existem linhas duplicadas
#print(dupl)

#Eliminar duplicados
dropd = df.drop_duplicates()
print(dropd)


'''Parece que os dados já estão tratados'''

'''Exploração de Dados'''

''' Target

##contando os valores de sim e não da coluna target(alvo)
print(df.target.value_counts())

## gráfico de contagem (count plot) que mostra a distribuição das classes(sim e não) na coluna "target" 
sns.countplot(x="target", data=df, hue="target", palette="bwr", legend=False)
plt.xlabel("Target (0 = Não, 1= Sim)")
plt.show()

#calcula e imprime a porcentagem de pacientes que têm e não têm doença cardíaca

countNoDisease = len(df[df.target == 0])
countHaveDisease = len(df[df.target == 1])
print("Porcentagem de pacientes que não têm doença cardíaca: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Porcentagem de pacientes que têm doença cardíaca: {:.2f}%".format((countHaveDisease / (len(df.target))*100))) 
'''

''' Sexo
## gráfico de contagem (count plot) que mostra a distribuição das classes(female e male) na coluna "sex" 
sns.countplot(x="sex", data=df, hue="sex", palette="mako_r", legend=True)
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

countFemale = len(df[df.sex == 0])
countMale = len(df[df.sex == 1])

print("Percentagem de pacientes do sexo feminino: {:.2f}%".format((countFemale/(len(df.sex))*100)))
print("Percentagem de pacientes do sexo masculino: {:.2f}%".format((countMale / (len(df.sex))*100)))

'''


#df.groupby('target').mean()