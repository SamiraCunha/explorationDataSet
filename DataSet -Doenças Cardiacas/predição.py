import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

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
#print(dc['trestbps'].quantile(0.99)) 

dc1 = dc[dc["trestbps"] < dc['trestbps'].quantile(0.99)]
'''
plt.figure(figsize=(16, 6))
sns.boxplot(data=dc1, orient='h')  # Plotar o boxplot horizontalmente  
plt.title('Boxplot das Variáveis no DataFrame')
plt.xlabel('Valores')
plt.ylabel('Variáveis')
plt.show()
'''

#print(dc1.describe())


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
# print(dc2['thalach'].quantile(0.005))

dc3 = dc2[dc2["thalach"] > dc2['thalach'].quantile(0.005)]

#print(dc3.describe())

categoricas = ['sex','cp','fbs','restecg','exang','slope','ca','thal']
numericas = ['age','trestbps','chol','thalach','oldpeak']

dc4 = pd.get_dummies(dc3, columns = ['sex','cp','fbs','restecg','exang','slope','ca','thal'])
#print(dc4)

#Creating Model for Logistic Regression

y = dc4.target.values
x_data = dc4.drop(['target'], axis = 1)

# Normalize
x = (x_data - np.min(x_data)) / (np.max(x_data) - np.min(x_data))

#20% dos dados serão usados como conjunto de teste
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2,random_state=0)

#Aqui, os conjuntos de treinamento e teste são transpostos, ou seja,
# as linhas são transformadas em colunas e vice-versa.
x_train = x_train.T
y_train = y_train.T
x_test = x_test.T
y_test = y_test.T

#peso = 0,01 e bias = 0,0

#initialize
def initialize(dimension):
    
    weight = np.full((dimension,1),0.01)
    bias = 0.0
    return weight,bias

def sigmoid(z):
    
    y_head = 1/(1+ np.exp(-z))
    return y_head

def forwardBackward(weight,bias,x_train,y_train):
    # Forward
    
    y_head = sigmoid(np.dot(weight.T,x_train) + bias)
    loss = -(y_train*np.log(y_head) + (1-y_train)*np.log(1-y_head))
    cost = np.sum(loss) / x_train.shape[1]
    
    # Backward
    derivative_weight = np.dot(x_train,((y_head-y_train).T))/x_train.shape[1]
    derivative_bias = np.sum(y_head-y_train)/x_train.shape[1]
    gradients = {"Derivative Weight" : derivative_weight, "Derivative Bias" : derivative_bias}
    
    return cost,gradients

def update(weight,bias,x_train,y_train,learningRate,iteration) :
    costList = []
    index = []
    
    #for each iteration, update weight and bias values
    for i in range(iteration):
        cost,gradients = forwardBackward(weight,bias,x_train,y_train)
        weight = weight - learningRate * gradients["Derivative Weight"]
        bias = bias - learningRate * gradients["Derivative Bias"]
        
        costList.append(cost)
        index.append(i)

    parameters = {"weight": weight,"bias": bias}
    
    print("iteration:",iteration)
    print("cost:",cost)

    plt.plot(index,costList)
    plt.xlabel("Number of Iteration")
    plt.ylabel("Cost")
    plt.show()

    return parameters, gradients

def predict(weight,bias,x_test):
    z = np.dot(weight.T,x_test) + bias
    y_head = sigmoid(z)

    y_prediction = np.zeros((1,x_test.shape[1]))
    
    for i in range(y_head.shape[1]):
        if y_head[0,i] <= 0.5:
            y_prediction[0,i] = 0
        else:
            y_prediction[0,i] = 1
    return y_prediction

def logistic_regression(x_train,y_train,x_test,y_test,learningRate,iteration):
    dimension = x_train.shape[0]
    weight,bias = initialize(dimension)
    
    parameters, gradients = update(weight,bias,x_train,y_train,learningRate,iteration)

    y_prediction = predict(parameters["weight"],parameters["bias"],x_test)
    
    print("Manuel Test Accuracy: {:.2f}%".format((100 - np.mean(np.abs(y_prediction - y_test))*100)))
    
logistic_regression(x_train,y_train,x_test,y_test,1,100)
    
#Sklearn Logistic Regression
accuracies = {}

lr = LogisticRegression()
lr.fit(x_train.T,y_train.T)
acc = lr.score(x_test.T,y_test.T)*100

accuracies['Logistic Regression'] = acc
print("Test Accuracy {:.2f}%".format(acc))

''' 
# KNN Model
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 2)  # n_neighbors means k
knn.fit(x_train.T, y_train.T)
prediction = knn.predict(x_test.T)

print("{} NN Score: {:.2f}%".format(2, knn.score(x_test.T, y_test.T)*100))
'''

#Support Vector Machine (SVM) Algorithm
svm = SVC(random_state = 1)
svm.fit(x_train.T, y_train.T)

acc = svm.score(x_test.T,y_test.T)*100
accuracies['SVM'] = acc
print("Test Accuracy of SVM Algorithm: {:.2f}%".format(acc))

#Naive Bayes Algorithm
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train.T, y_train.T)

acc = nb.score(x_test.T,y_test.T)*100
accuracies['Naive Bayes'] = acc
print("Accuracy of Naive Bayes: {:.2f}%".format(acc))

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(x_train.T, y_train.T)

acc = dtc.score(x_test.T, y_test.T)*100
accuracies['Decision Tree'] = acc
print("Decision Tree Test Accuracy {:.2f}%".format(acc))

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators = 1000, random_state = 1)
rf.fit(x_train.T, y_train.T)

acc = rf.score(x_test.T,y_test.T)*100
accuracies['Random Forest'] = acc
print("Random Forest Algorithm Accuracy Score : {:.2f}%".format(acc))


#Comparando modelos
colors = ["purple", "green", "orange", "magenta","#CFC60E"]

sns.set_style("whitegrid")
plt.figure(figsize=(16,5))
plt.yticks(np.arange(0,100,10))
plt.ylabel("Accuracy %")
plt.xlabel("Algorithms")
sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()), palette=colors, hue=list(accuracies.keys()))
plt.show()

# Valores previstos
y_head_lr = lr.predict(x_test.T)
y_head_svm = svm.predict(x_test.T)
y_head_nb = nb.predict(x_test.T)
y_head_dtc = dtc.predict(x_test.T)
y_head_rf = rf.predict(x_test.T)

from sklearn.metrics import confusion_matrix

cm_lr = confusion_matrix(y_test,y_head_lr)
cm_svm = confusion_matrix(y_test,y_head_svm)
cm_nb = confusion_matrix(y_test,y_head_nb)
cm_dtc = confusion_matrix(y_test,y_head_dtc)
cm_rf = confusion_matrix(y_test,y_head_rf)