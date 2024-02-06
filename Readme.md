
# Bibliotecas instaladas por enquanto

pip install matplotlib
pip install pandas
pip install numpy

#  Definição do Problema

A base de dados tem 14 atributos.
Temos dados que classificam se os pacientes têm doenças cardíacas ou não de acordo com suas características. Tentaremos usar esses dados para criar um modelo que tente prever se um paciente tem ou não esta doença.
Usaremos algoritmo de regressão logística (classificação).

# Data contains

age - age in years
sex - (1 = male; 0 = female)
cp - chest pain type
trestbps - resting blood pressure (in mm Hg on admission to the hospital)
chol - serum cholestoral in mg/dl
fbs - (fasting blood sugar > 120 mg/dl) (1 = true; 0 = false)
restecg - resting electrocardiographic results
thalach - maximum heart rate achieved
exang - exercise induced angina (1 = yes; 0 = no)
oldpeak - ST depression induced by exercise relative to rest
slope - the slope of the peak exercise ST segment
ca - number of major vessels (0-3) colored by flourosopy
thal - 3 = normal; 6 = fixed defect; 7 = reversable defect
target - have disease or not (1=yes, 0=no)

# target
df.target.value_counts()

1  -  526 sim
0  -  499 não

# Formato inicial dos dados
print(df.shape)
(l,c)
(1025, 14)

# Os dados contêm:

idade - idade em anos
sexo - (1 = masculino; 0 = feminino)
cp - tipo de dor no peito
trestbps - pressão arterial em repouso (em mm Hg na admissão ao hospital)
col - colesterol sérico em mg/dl
fbs - (glicemia em jejum > 120 mg/dl) (1 = verdadeiro; 0 = falso)
retecg - resultados eletrocardiográficos em repouso
thalach - frequência cardíaca máxima alcançada
exang - angina induzida por exercício (1 = sim; 0 = não)
oldpeak - depressão do segmento ST induzida pelo exercício em relação ao repouso
inclinação - a inclinação do pico do segmento ST do exercício
ca - número de vasos principais (0-3) coloridos por fluorosopia
tal - 3 = normal; 6 = defeito fixo; 7 = defeito reversível
alvo - ter doença ou não (1=sim, 0=não)

# forma antiga
#sns.countplot(x="target", data=df, palette="bwr")
# forma nova
sns.countplot(x="target", data=df, hue="target", palette="bwr", legend=False)
# explorationDataSet
