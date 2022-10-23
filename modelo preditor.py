import pandas as pd
import matplotlib.pyplot as plt
import time
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from imblearn import under_sampling,over_sampling
from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

# carregando os dados
df = pd.read_csv('dados.csv',sep=';')

# analise exploratoria
print(df.shape)
print(df.head(),'\n')
print(df.info(),'\n')

# periodo de coleta dos dados
inicio = pd.to_datetime(df['DT_AQUISICAO']).dt.date.min()
fim = pd.to_datetime(df['DT_AQUISICAO']).dt.date.max()
print(f'Período dos dados - de {inicio} até {fim}','\n')

print(df.describe(),'\n')
# verificando a presença de dados missing
print(df.isna().sum(),'\n')

# verificando os valores exclusivos de cada variavel
print(df.nunique(),'\n')

# analisando as variaveis categoricas
print(df.groupby(['FORMA_AQUISICAO']).size(),'\n')

print(df.groupby(['SEXO']).size(),'\n')

print(df.groupby(['DURACAO_CONTRATO']).size(),'\n')

print(df.groupby(['NOME_PRODUTO']).size(),'\n')

print(df.groupby(['SITUACAO']).size(),'\n')

#GERANDO UM GRAFICO DAs VARIAVEis FORMA_AQUISICAO E SEXO em relação a variavel alvo SITUAÇÃO
plt.rcParams['figure.figsize'] = [12.00,3.50]
plt.rcParams['figure.autolayout'] = True
f, axes = plt.subplots(1,2)
sns.countplot(data=df,x='SEXO',hue='SITUACAO',ax = axes[0])
sns.countplot(data=df,x='FORMA_AQUISICAO',hue='SITUACAO',ax = axes[1])
plt.show()

#GERANDO UM GRAFICO DA VARIAVEl duração_contrato em relação a variavel alvo SITUAÇÃO
plt.rcParams['figure.figsize'] = [15.00,5.0]
plt.rcParams['figure.autolayout'] = True
sns.countplot(data=df,x='DURACAO_CONTRATO',hue='SITUACAO')
plt.show()

#GERANDO UM GRAFICO DA VARIAVEl NOME_PRODUTO em relação a variavel alvo SITUAÇÃO
plt.rcParams['figure.figsize'] = [15.00,5.0]
plt.rcParams['figure.autolayout'] = True
sns.countplot(data=df,x='NOME_PRODUTO',hue='SITUACAO')
plt.show()

#GERANDO UM GRAFICO DA VARIAVEl de saida SITUAÇÃO
plt.rcParams['figure.figsize'] = [15.00,5.0]
plt.rcParams['figure.autolayout'] = True
df.SITUACAO.value_counts().plot(kind='bar',title='Clientes ativos x cancelados',color=['blue','orange'])
plt.show()

# analisando as variaveis numericas
variaveis_numericas = []
for i in df.columns[1:24].tolist():
    if df.dtypes[i] == 'int64' or df.dtypes[i] == 'float64':
        print(i,' : ',df.dtypes[i])
        variaveis_numericas.append(i)

print(len(variaveis_numericas))


# plotando os graficos de boxplot das variaveis numericas para verificar a presença de outliers
plt.rcParams['figure.figsize'] = [14.00,24.00]
plt.rcParams['figure.autolayout'] = True
f, axes = plt.subplots(4,4)
linha = 0
coluna= 0
for i in variaveis_numericas:
    sns.boxplot(data=df,y=i,ax=axes[linha][coluna])
    coluna += 1
    if coluna == 4:
        linha +=1
        coluna = 0

plt.show()

# as variaveis que possuem outliers mais significativos são QT_FILHOS, QT_PC_PAGAS e QT_PC_PAGA_EM_DIA, logo , irão passar por um tratamento


# é possivel notar que apenas 5 registros são outliers no contexto de QT_FILHOS e podem ser removidos do conjunto de dados
print(df.groupby(['QT_FILHOS']).size(),'\n')
print(df.loc[df["QT_FILHOS"] > 2])




#### tratando os dados
# removendo os outliers da coluna QT_FILHOS
dados = df.drop(df.loc[df["QT_FILHOS"] > 2].index)
print(dados.shape)
print(dados.groupby(['QT_FILHOS']).size(),'\n')

#substituindo os dados nulls
print('media de filhos: ',dados['QT_FILHOS'].mean())
print('mediana de filhos: ',dados['QT_FILHOS'].median())
print('moda de filhos: ',dados['QT_FILHOS'].mode())
dados['QT_FILHOS'] = dados['QT_FILHOS'].fillna(dados['QT_FILHOS'].median())
print(dados.isna().sum())

# transformando dados categoricos em numericos nesta variavel DURACAO_CONTRATO
dados['DURACAO_CONTRATO'] = dados['DURACAO_CONTRATO'].replace(['12 Meses'],12)
dados['DURACAO_CONTRATO'] = dados['DURACAO_CONTRATO'].replace(['24 Meses'],24)
dados['DURACAO_CONTRATO'] = dados['DURACAO_CONTRATO'].replace(['36 Meses'],36)
dados['DURACAO_CONTRATO'] = dados['DURACAO_CONTRATO'].replace(['48 Meses'],48)

print(dados.info())

# identificando os valores maximos e minimos para tratar os outliers de QT_PC_PAGAS e QT_PC_PAGA_EM_DIA
print('\n',dados['QT_PC_PAGAS'].max())
print(dados['QT_PC_PAGA_EM_DIA'].max())
# substituindo os dados outliers dessas colunas
dados.loc[dados.QT_PC_PAGAS > dados.DURACAO_CONTRATO,'QT_PC_PAGAS'] = dados.DURACAO_CONTRATO
dados.loc[dados.QT_PC_PAGA_EM_DIA > dados.DURACAO_CONTRATO,'QT_PC_PAGA_EM_DIA'] = dados.DURACAO_CONTRATO

# observando os novos limites ajustados
print('\n',dados['QT_PC_PAGAS'].max())
print(dados['QT_PC_PAGA_EM_DIA'].max(),'\n')

## engenharia de atributos

#novo atributo que indica o nivel de pagamento dos clientes de acordo com as parcelas pagas
intervalos = [-10,3,6,12,48]
nomes = ['RUIM','MEDIO','BOM','OTIMO']
dados['NIVEL_PAGAMENTO'] = pd.cut(dados['QT_PC_PAGAS'],bins=intervalos,labels=nomes)
print(dados['NIVEL_PAGAMENTO'].value_counts())

# criando um objetor que irá transformar os dados categoricos em numericos das demais variaveis categoricas
lb = LabelEncoder()

dados['SEXO'] = lb.fit_transform(dados['SEXO'])
dados['FORMA_AQUISICAO'] = lb.fit_transform(dados['FORMA_AQUISICAO'])
dados['NOME_PRODUTO'] = lb.fit_transform(dados['NOME_PRODUTO'])
dados['NIVEL_PAGAMENTO'] = lb.fit_transform(dados['NIVEL_PAGAMENTO'])

print(dados.sample(15))

# variaveis de entrada e saida que serão utilizadas pelo modelo de aprendizado de maquina
colunas =[i for i in dados.columns if i not in ('ID_CLIENTE','DT_AQUISICAO','DT_CANCELAMENTO','SITUACAO')]
print(colunas)

dados2 = pd.DataFrame(dados,columns=colunas)
print(dados2.info())

#GERANDO UM GRAFICO DA VARIAVEl de saida
plt.rcParams['figure.figsize'] = [15.00,5.0]
plt.rcParams['figure.autolayout'] = True
dados2['COD_SITUACAO'].value_counts().plot(kind='bar',title='Clientes ativos x cancelados',color=['blue','orange'])
plt.show()

# SEPARANDO OS DADOS DE ENTRADA E DE SAIDA

entradas = dados2.loc[:,dados2.columns != 'COD_SITUACAO']
alvo = dados2['COD_SITUACAO']

print(entradas.info())
print(alvo.sample(5))

# criar o objeto balanceador e fazendo o balanceamento dos dados
balanceador = SMOTE(random_state=100)
entradas_baca, alvo_baca = balanceador.fit_resample(entradas,alvo)

#GERANDO UM GRAFICO DA VARIAVEl de saida balanceada
plt.rcParams['figure.figsize'] = [15.00,5.0]
plt.rcParams['figure.autolayout'] = True
alvo_baca.value_counts().plot(kind='bar',title='Clientes ativos x cancelados',color=['blue','orange'])
plt.show()

print(entradas_baca.shape)
print(alvo_baca.shape)

# separando os dados de treino e teste
x_treino,x_teste,y_treino,y_teste = train_test_split(entradas_baca,alvo_baca,test_size=0.3,random_state=42)

# Padronizando os dados
Padrao = StandardScaler()
x_treino_padro = Padrao.fit_transform(x_treino)
x_teste_padro = Padrao.fit_transform(x_teste)

# determinando o melhor modelo classificador: KNeighborsClassifier
kvals = range(3,10,2)

acuracia = []

start = time.time()
for k in kvals:
    modeloKNN = KNeighborsClassifier(n_neighbors=k)
    modeloKNN.fit(x_treino_padro,y_treino)
    # avaliando os modelos
    predito = modeloKNN.predict(x_teste_padro)
    acur = accuracy_score(y_teste,predito)
    acuracia.append(acur)
    print(f'Com o valor de k = {k}, a acurácia é {acur*100:.2f}%')

fim = time.time()
print('\n',f'Tempo de treinamento dos modelos: {fim-start}s')

## criando o modelo final
# obtendo o valor de k do modelo com o melhor desempenho
f = np.argmax(acuracia)
modeloKNN_final = KNeighborsClassifier(n_neighbors=kvals[f])
modeloKNN_final.fit(x_treino_padro,y_treino)
previsao = modeloKNN_final.predict(x_teste_padro)

print(f'Acurácia do modelo final é {accuracy_score(y_teste,previsao):.3f}')
