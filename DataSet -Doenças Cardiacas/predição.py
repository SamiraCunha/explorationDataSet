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
#print(df.duplicated().sum()) #RESULTADO: Existem linhas duplicadas


#Eliminar duplicados \ dc = doenças cardiacas
dc = df.drop_duplicates()

#print(dc.shape)

#print(dc.describe())


#plt.figure(figsize=(16,6))
#sns.heatmap(dc.corr(), annot=True, cmap='Blues')
#plt.show() 


'''Exploração de Dados'''

''' Target

##contando os valores de sim e não da coluna target(alvo)
print(dc.target.value_counts())

## gráfico de contagem (count plot) que mostra a distribuição das classes(sim e não) na coluna "target" 
sns.countplot(x="target", data=dc, hue="target", palette="bwr", legend=False)
plt.xlabel("Target (0 = Não, 1= Sim)")
plt.show()

#calcula e imprime a porcentagem de pacientes que têm e não têm doença cardíaca

countNoDisease = len(dc[dc.target == 0])
countHaveDisease = len(dc[dc.target == 1])
print("Porcentagem de pacientes que não têm doença cardíaca: {:.2f}%".format((countNoDisease / (len(df.target))*100)))
print("Porcentagem de pacientes que têm doença cardíaca: {:.2f}%".format((countHaveDisease / (len(df.target))*100))) 
'''

''' Sexo
## gráfico de contagem (count plot) que mostra a distribuição das classes(female e male) na coluna "sex" 
sns.countplot(x="sex", data=dc, hue="sex", palette="mako_r", legend=True)
plt.xlabel("Sex (0 = female, 1= male)")
plt.show()

countFemale = len(dc[dc.sex == 0])
countMale = len(dc[dc.sex == 1])

print("Percentagem de pacientes do sexo feminino: {:.2f}%".format((countFemale/(len(df.sex))*100)))
print("Percentagem de pacientes do sexo masculino: {:.2f}%".format((countMale / (len(df.sex))*100)))

'''

#print(dc.groupby('target').mean())

#for c in dc.columns:
 #   print(f"{c}: {dc[c].nunique()}")
    
categoricas = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
numericas = ['age','trestbps','chol','thalach','oldpeak']

#Visualização da distribuição das variáveis 

'''Distribuição dos dados para variáveis categóricas usando gráficos de contagem e para variáveis numéricas usando histogramas.
Isso permite uma rápida análise visual das características dos dados e sua relação com a variável alvo '''

'''
for c in dc.columns:
    plt.figure(figsize=(8,4))
    plt.title(f"Coluna avaliada: {c}", fontsize = 16)
    if c in categoricas:
        sns.countplot(x = dc[c], hue = dc['target'])
    if c in numericas:
        sns.histplot(dc[c], kde = True)
    plt.show()
'''    
#Removendo outliers
'''ver outliers 
plt.figure(figsize=(16, 6))
sns.boxplot(data=dc, orient='h')  # Plotar o boxplot horizontalmente  
plt.title('Boxplot das Variáveis no DataFrame')
plt.xlabel('Valores')
plt.ylabel('Variáveis')
plt.show()

#atributos com outliers: trestbps, chol, thalach

print(dc.describe())
'''

# terstbps
#testar quanto ficaria se tirasse 1 % 
print(dc['trestbps'].quantile(0.99)) 

dc1 = dc[dc["trestbps"] < dc['trestbps'].quantile(0.99)]
'''
plt.figure(figsize=(16, 6))
sns.boxplot(data=dc1, orient='h')  # Plotar o boxplot horizontalmente  
plt.title('Boxplot das Variáveis no DataFrame')
plt.xlabel('Valores')
plt.ylabel('Variáveis')
plt.show()
'''

print(dc1.describe())


# chol
#print(dc1['chol'].quantile(0.98))

dc2 = dc1[dc1["chol"] < dc1['chol'].quantile(0.98)]
#print(dc2.describe())
'''''
plt.figure(figsize=(16, 6))
sns.boxplot(data=dc2, orient='h')  # Plotar o boxplot horizontalmente  
plt.title('Boxplot das Variáveis no DataFrame')
plt.xlabel('Valores')
plt.ylabel('Variáveis')
plt.show()
'''

#nesse caso tira -se o excesso do inicio
print(dc2['thalach'].quantile(0.005))

