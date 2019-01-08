# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 23:53:54 2019

@author: Ivan Alves
"""

import pandas as pd
base  = pd.read_csv('credit-data.csv')
#mostrando valores de idade negativa
base.loc[base['age'] < 0]
#apagar a coluna -> coluna age, 1 apagar coluna completa, inplace=true  sem retorno
base.drop('age', 1, inplace=True)
#apagar somente os registros inconsistentes
base.drop(base[base.age < 0].index, inplace=True)
#preenchimento manual
#preencher valores utilzando a média
base.mean()
#pegando média de idade do arquivo original
mediaIdadeOriginal = base['age'].mean()
#pegando média de idades corretas
mediaIdade = base['age'][base.age > 0].mean()
#atualizando valores negativos, para a media de idades
base.loc[base.age < 0, 'age' ] = mediaIdade

#Tratamento de valores faltantes 
pd.isnull(base['age'])
#pegando todos valores NaN da coluna age
base.loc[pd.isnull(base['age'])]

#pegando previsores e classes   
previsores = base.iloc[:, 1:4].values
classe = base.iloc[:, 4].values

#tratamento de valores faltantes
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(previsores[:, 0:3]) 
previsores[:, 0:3] = imputer.transform(previsores[:, 0:3])