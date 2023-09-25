# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 08:30:18 2023

@author: jeferson.dantas
"""


# %% carregar pacotes
from DadosAbertosBrasil import ibge
from DadosAbertosBrasil import ipea
import pandas as pd
from IPython.display import display
from datetime import timedelta
from numpy import sqrt
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib as mpl
import pandas as pd
import os
import numpy as np
import gc

# %% Coletar Informações do DadosAbertosBrasil

# determinando local Sergipe
local = ipea.lista_territorios()
local = local.loc[local['nome']=='Sergipe']['codigo']
idlocal = int(local.iloc[0])


# %% Lê os arquivos .csv coletados e cria dataframes pré-tratados. Estes dataframes são salvos em arquivos CSV para seremusados posteriormente.
## Esta etapa só precisa ser executada a primeira vez, pos os dados coletados são armazenados nos arquivos  e reutilizados.


# Defina o caminho da pasta
path = "dados brutos ans/"


# Liste todos os arquivos na pasta
all_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

# Crie uma lista para armazenar cada DataFrame
dfs = []

# Leia cada arquivo e adicione-o à lista
for file in all_files:
    try:
        df = pd.read_csv(file, encoding='ISO-8859-1', delimiter=';')
        dfs.append(df)
        print(f"arquivo {file} carregado.")
    except Exception as e:
        print(f"Erro ao ler o arquivo {file}: {e}")

# Concatene todos os DataFrames na lista em um único DataFrame
final_df = pd.concat(dfs, ignore_index=True)

# Filtrar o dataframe com base nos valores da coluna MODALIDADE_OPERADORA
filtered_df = final_df[final_df['MODALIDADE_OPERADORA'].isin(['COOPERATIVA MÉDICA', 'MEDICINA DE GRUPO', 'SEGURADORA ESPECIALIZADA EM SAÚDE'])]
filtered_df = filtered_df[filtered_df['COBERTURA_ASSIST_PLAN'].isin(['Médico-hospitalar'])]

# Agrupar e somar
grouped_df = filtered_df.groupby(['#ID_CMPT_MOVEL', 'NM_RAZAO_SOCIAL', 'CD_PLANO'])['QT_BENEFICIARIO_ATIVO'].sum().reset_index()

# Renomear as colunas
ANS_planos = grouped_df.rename(columns={'#ID_CMPT_MOVEL': 'Ano_mes', 'NM_RAZAO_SOCIAL': 'Plano', 'CD_PLANO': 'cod_plano', 'QT_BENEFICIARIO_ATIVO': 'Benef_ativos'})


#ANS_planos = ANS_planos.merge(df_preco[['Ano_mes', 'cod_plano', 'preco_medio']], 
#                              on=['Ano_mes', 'cod_plano'], 
#                              how='left')


############################################################
############################################################
############################################################
############################################################
## Filtrar dataset 
ANS_planos = ANS_planos[ANS_planos['Ano_mes']>=201807]
ANS_planos = ANS_planos.reset_index(drop=True)
############################################################
############################################################
############################################################
############################################################
############################################################
print(ANS_planos.describe())
print(ANS_planos['Ano_mes'].value_counts())
print(ANS_planos['Ano_mes'].nunique())
print(ANS_planos['Plano'].nunique())

# Ordenar o dataframe com base na coluna Ano_mes
ANS_anomes = ANS_planos.groupby(['Ano_mes']).agg({'Benef_ativos': 'sum'}).reset_index()

ANS_anomes = ANS_anomes.sort_values(by='Ano_mes', ascending=True)
ANS_anomes['seq_tempo'] = range(1, len(ANS_anomes) + 1)

# Filtrar os 12 últimos valores
ANS_anomes_last_12 = ANS_anomes.tail(12)

ANS_base_Abc = ANS_planos[ANS_planos['Ano_mes'].isin(ANS_anomes_last_12['Ano_mes'])]
ANS_base_Abc = ANS_base_Abc.groupby(['Plano'])['Benef_ativos'].sum().reset_index()
ANS_base_Abc = ANS_base_Abc.sort_values(by='Benef_ativos', ascending=False)
ANS_base_Abc['Benef_ativos_acumulado'] = ANS_base_Abc['Benef_ativos'].cumsum()
total_benef_ativos = ANS_base_Abc['Benef_ativos'].sum()
ANS_base_Abc['percentual_acumulado'] = (ANS_base_Abc['Benef_ativos_acumulado'] / total_benef_ativos) * 100
ANS_base_Abc.loc[ANS_base_Abc['percentual_acumulado'] > 90, 'Plano'] = 'Geral'
ANS_base_Abc= ANS_base_Abc.groupby(['Plano'])['Benef_ativos'].sum().reset_index()
ANS_base_Abc['coluna_ordem'] = np.where(ANS_base_Abc['Plano'] == 'Geral', 0, ANS_base_Abc['Benef_ativos'])
ANS_base_Abc = ANS_base_Abc.sort_values(by='coluna_ordem', ascending=False)
ANS_base_Abc['idplano'] = range(1, len(ANS_base_Abc) + 1)
ANS_base_Abc = ANS_base_Abc.reset_index(drop=True)

# Realiza o merge com base na coluna 'Ano_mes'
merged_df = ANS_planos.merge(ANS_anomes[['Ano_mes', 'seq_tempo']], on='Ano_mes', how='left')
# Atualiza o dataframe ANS_planos com a coluna 'seq_tempo' do dataframe merged_df
ANS_planos['seq_tempo'] = merged_df['seq_tempo']



# Verifica se os valores em 'Plano' de ANS_planos estão em 'Plano' de ANS_base_Abc
mask = ANS_planos['Plano'].isin(ANS_base_Abc['Plano'])
# atualiza a coluna plano com base na condição
ANS_planos['Plano'] = np.where(mask, ANS_planos['Plano'], 'Geral')

# Realiza o merge com base na coluna 'Plano'
merged_df = ANS_planos.merge(ANS_base_Abc[['Plano', 'idplano']], on='Plano', how='left')
# Atualiza o dataframe ANS_planos com a coluna 'idplano' do dataframe merged_df
ANS_planos['idplano'] = merged_df['idplano']
ANS_planos = ANS_planos.groupby(['seq_tempo', 'idplano']).agg({'Benef_ativos': 'sum'}).reset_index()

#ANS_planos = ANS_planos.groupby(['seq_tempo', 'idplano'])['Benef_ativos'].sum().reset_index()
ANS_planos = ANS_planos.sort_values(by=['seq_tempo','idplano'] , ascending=[True, True])



###### Preencher dados faltantes com Zero
all_combinations = pd.merge(ANS_base_Abc.assign(key=1), ANS_anomes.assign(key=1), on='key').drop('key', axis=1)
all_combinations = all_combinations[['seq_tempo','idplano']]
all_combinations = all_combinations.sort_values(by=['seq_tempo', 'idplano'])
# Realizar o merge com o dataframe original
ANS_planos = pd.merge(all_combinations, ANS_planos, on=['idplano', 'seq_tempo'], how='left')
# Preencher os valores faltantes (por exemplo, com 0)
ANS_planos['Benef_ativos'].fillna(0, inplace=True)
ANS_planos = ANS_planos.sort_values(by=['seq_tempo', 'idplano'])
ANS_planos = ANS_planos.reset_index(drop=True)
##############


## Salvando arquivos
ANS_planos.to_csv("dados tratados/tabela_anomes_planos.csv")
ANS_base_Abc.to_csv("dados tratados/tabela_planos.csv")
ANS_anomes.to_csv("dados tratados/tabela_anomes.csv")


del df
del final_df 
del filtered_df
del grouped_df
del total_benef_ativos
del mask
del merged_df
gc.collect()




# %% carregar arquivos ANS_PLANOS

ANS_base_Abc = pd.read_csv("dados tratados/tabela_planos.csv", encoding='ISO-8859-1', delimiter=',') 
ANS_anomes =  pd.read_csv("dados tratados/tabela_anomes.csv", encoding='ISO-8859-1', delimiter=',') 
ANS_planos =  pd.read_csv("dados tratados/tabela_anomes_planos.csv", encoding='ISO-8859-1', delimiter=',') 

#%% Funções de defasagem dos dados socieconômicos (se necessário).

def defasar_1_mes(anomes):
    year = int(anomes / 100)
    month = anomes % 100
    month += 1
    if month > 12:
        month -= 12
        year += 1
    return year * 100 + month


def defasar_2_meses(anomes):
    year = int(anomes / 100)
    month = anomes % 100
    month += 2
    if month > 12:
        month -= 12
        year += 1
    return year * 100 + month

def defasar_3_meses(anomes):
    year = int(anomes / 100)
    month = anomes % 100
    month += 3
    if month > 12:
        month -= 12
        year += 1
    return year * 100 + month

def defasar_4_meses(anomes):
    year = int(anomes / 100)
    month = anomes % 100
    month += 4
    if month > 12:
        month -= 12
        year += 1
    return year * 100 + month

def defasar_5_meses(anomes):
    year = int(anomes / 100)
    month = anomes % 100
    month += 5
    if month > 12:
        month -= 12
        year += 1
    return year * 100 + month


def defasar_6_meses(anomes):
    year = int(anomes / 100)
    month = anomes % 100
    month += 6
    if month > 12:
        month -= 12
        year += 1
    return year * 100 + month

def nao_defasar(anomes):
    return anomes


# %%IPCA coletando dados do IPCA nos DadosAbertosBrasil

serieIPCA = ipea.Serie('BM12_IPCA2012').valores

serieIPCA['data'] = pd.to_datetime(serieIPCA['data'])
serieIPCA = serieIPCA.loc[serieIPCA['data']>='2013-01-01',['data', 'valor']]
# Calculando o valor acumulado dos últimos 12 meses
serieIPCA['valor_ac12'] = serieIPCA['valor'].rolling(window=12).sum()
serieIPCA['valor_ac12'].fillna(0, inplace=True)
serieIPCA['Ano_mes_original'] = serieIPCA['data'].dt.strftime('%Y%m').astype('int64')
serieIPCA['Ano_mes'] = serieIPCA['Ano_mes_original'].apply(nao_defasar)
serieIPCA = serieIPCA.reset_index(drop=True)
merged_df = serieIPCA.merge(ANS_anomes[['Ano_mes', 'seq_tempo']], on='Ano_mes', how='left')
serieIPCA['seq_tempo'] = merged_df['seq_tempo']
serieIPCA = serieIPCA.dropna(subset=['seq_tempo'])
serieIPCA.rename(columns={'valor_ac12': 'ipca'}, inplace=True)
serieIPCA = serieIPCA.reset_index(drop=True)

# Salva arquivo para usu posterior
serieIPCA.to_csv("dados brutos/ipca.csv")



# %% Lendo dados dos nascimentos de arquivos baixados no datasus e armazendandos em pasta

path = "dados saudegov/nascimentos"

# Liste todos os arquivos na pasta
all_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

# Crie uma lista para armazenar cada DataFrame
dfs = []

# Leia cada arquivo e adicione-o à lista
for file in all_files:
    try:
        df = pd.read_csv(file, encoding='ISO-8859-1', delimiter=';')
        dfs.append(df)
        print(f"arquivo {file} carregado.")
    except Exception as e:
        print(f"Erro ao ler o arquivo {file}: {e}")

# Concatene todos os DataFrames na lista em um único DataFrame
serienascimentos = pd.concat(dfs, ignore_index=True)
serienascimentos.drop(serienascimentos.columns[0], axis=1, inplace=True)
serienascimentos = serienascimentos.melt(var_name='Ano_mes', value_name='nascimentos')
serienascimentos = serienascimentos.dropna(subset=['nascimentos'])
serienascimentos = serienascimentos.groupby(['Ano_mes'])['nascimentos'].sum().reset_index()

# Dividir a coluna 'ano_mes' em duas colunas 'mes' e 'ano'
serienascimentos[['mes', 'ano']] = serienascimentos['Ano_mes'].str.split('/', expand=True)
# Ajustar o formato do ano para ter 4 dígitos
serienascimentos['ano'] = '20' + serienascimentos['ano']
# Concatenar 'ano' e 'mes' para obter o formato YYYYMM
serienascimentos['Ano_mes_original'] = serienascimentos['ano'] + serienascimentos['mes']
# Converter a coluna 'ano_mes' para inteiro
serienascimentos['Ano_mes_original'] = serienascimentos['Ano_mes_original'].astype(int)
# Opcional: remover as colunas 'mes' e 'ano' se não forem mais necessárias
serienascimentos.drop(columns=['mes', 'ano'], inplace=True)
serienascimentos['Ano_mes'] = serienascimentos['Ano_mes_original'].apply(defasar_1_mes)

serienascimentos = serienascimentos.merge(ANS_anomes[['Ano_mes', 'seq_tempo']], on='Ano_mes', how='left')
serienascimentos = serienascimentos[['seq_tempo', 'nascimentos']]
serienascimentos = serienascimentos.dropna(subset=['seq_tempo'])
serienascimentos = serienascimentos.sort_values(by='seq_tempo', ascending=True)

# Encontrar os valores de seq_tempo que estão em ANS_anomes mas não em serienascimentos
missing_seq_tempo = set(ANS_anomes['seq_tempo']) - set(serienascimentos['seq_tempo'])

for seq in missing_seq_tempo:
    # Encontre o último valor de nascimento antes do seq_tempo ausente
    last_value = serienascimentos[serienascimentos['seq_tempo'] < seq]['nascimentos'].iloc[-1]
    
    # Adicione a nova linha ao dataframe serienascimentos
    new_row = pd.DataFrame({'seq_tempo': [seq], 'nascimentos': [last_value]})
    serienascimentos = pd.concat([serienascimentos, new_row], ignore_index=True)

# Ordene o dataframe serienascimentos por seq_tempo
serienascimentos = serienascimentos.sort_values(by='seq_tempo').reset_index(drop=True)

#salva arquivo para uso posterior
serienascimentos.to_csv("dados brutos/nascimentos.csv")

# %% Lendo dados dos óbitos de arquivos baixados no datasus e armazendandos em pasta

path = "dados saudegov/obitos"

# Liste todos os arquivos na pasta
all_files = [os.path.join(path, file) for file in os.listdir(path) if file.endswith('.csv')]

# Crie uma lista para armazenar cada DataFrame
dfs = []

# Leia cada arquivo e adicione-o à lista
for file in all_files:
    try:
        df = pd.read_csv(file, encoding='ISO-8859-1', delimiter=';')
        dfs.append(df)
        print(f"arquivo {file} carregado.")
    except Exception as e:
        print(f"Erro ao ler o arquivo {file}: {e}")

# Concatene todos os DataFrames na lista em um único DataFrame
serieobitos = pd.concat(dfs, ignore_index=True)
serieobitos.drop(serieobitos.columns[0], axis=1, inplace=True)
serieobitos = serieobitos.melt(var_name='Ano_mes', value_name='obitos')
serieobitos = serieobitos.dropna(subset=['obitos'])
serieobitos = serieobitos.groupby(['Ano_mes'])['obitos'].sum().reset_index()

# Dividir a coluna 'ano_mes' em duas colunas 'mes' e 'ano'
serieobitos[['mes', 'ano']] = serieobitos['Ano_mes'].str.split('/', expand=True)
# Ajustar o formato do ano para ter 4 dígitos
serieobitos['ano'] = '20' + serieobitos['ano']
# Concatenar 'ano' e 'mes' para obter o formato YYYYMM
serieobitos['Ano_mes_original'] = serieobitos['ano'] + serieobitos['mes']
# Converter a coluna 'ano_mes' para inteiro
serieobitos['Ano_mes_original'] = serieobitos['Ano_mes_original'].astype(int)
serieobitos['Ano_mes'] = serieobitos['Ano_mes_original'].apply(defasar_1_mes)
# Opcional: remover as colunas 'mes' e 'ano' se não forem mais necessárias
serieobitos.drop(columns=['mes', 'ano'], inplace=True)
serieobitos = serieobitos.merge(ANS_anomes[['Ano_mes', 'seq_tempo']], on='Ano_mes', how='left')
serieobitos = serieobitos[['seq_tempo', 'obitos']]
serieobitos = serieobitos.dropna(subset=['seq_tempo'])
serieobitos = serieobitos.sort_values(by='seq_tempo', ascending=True)

# Encontrar os valores de seq_tempo que estão em ANS_anomes mas não em serienascimentos
missing_seq_tempo = set(ANS_anomes['seq_tempo']) - set(serieobitos['seq_tempo'])

for seq in missing_seq_tempo:
    # Encontre o último valor de nascimento antes do seq_tempo ausente
    last_value = serieobitos[serieobitos['seq_tempo'] < seq]['obitos'].iloc[-1]
    
    # Adicione a nova linha ao dataframe serienascimentos
    new_row = pd.DataFrame({'seq_tempo': [seq], 'obitos': [last_value]})
    serieobitos = pd.concat([serieobitos, new_row], ignore_index=True)

# Ordene o dataframe serienascimentos por seq_tempo
serieobitos = serieobitos.sort_values(by='seq_tempo').reset_index(drop=True)

#salva arquivo para uso posterior
serieobitos.to_csv("dados brutos/serieobitos.csv")

# %% DESEMPREGO Sergipe - lendo arquivo PNAD TRIMESTRAL (arquivo gerado manualmente em planilha)

seriedesempregotrimestre = pd.read_csv("dados penad/penadcontinuatrimestral.csv", encoding='ISO-8859-1', delimiter=';')

# Criar um novo dataframe vazio
seriedesemprego = pd.DataFrame(columns=['ano', 'mes', 'txdesemprego'])

# Mapear trimestre para meses
trimestre_to_meses = {
    1: [1, 2, 3],
    2: [4, 5, 6],
    3: [7, 8, 9],
    4: [10, 11, 12]
}

# Inicializar a taxa de desemprego do trimestre anterior
tx_anterior = seriedesempregotrimestre.iloc[0]['txdesemprego']

for index, row in seriedesempregotrimestre.iterrows():
    meses = trimestre_to_meses[row['trimestre']]
    diferenca = (row['txdesemprego'] - tx_anterior) / 3
    
    for i, mes in enumerate(meses):
        # Distribuir txdesemprego linearmente entre os meses do trimestre
        tx_mensal = tx_anterior + diferenca * (i + 1)
        new_row = pd.DataFrame({
            'ano': [row['ano']],
            'mes': [mes],
            'txdesemprego': [tx_mensal]
        })
        seriedesemprego = pd.concat([seriedesemprego, new_row], ignore_index=True)
    
    tx_anterior = row['txdesemprego']

seriedesemprego['Ano_mes_original'] = seriedesemprego['ano']*100 + seriedesemprego['mes']
seriedesemprego['Ano_mes'] = seriedesemprego['Ano_mes_original'].apply(nao_defasar)
#seriedesemprego['Ano_mes'] = seriedesemprego['Ano_mes_original'].apply(defasar_2_meses)


seriedesemprego = seriedesemprego.merge(ANS_anomes[['Ano_mes', 'seq_tempo']], on='Ano_mes', how='left')
seriedesemprego = seriedesemprego[['seq_tempo', 'txdesemprego','Ano_mes','Ano_mes_original']]
seriedesemprego = seriedesemprego.sort_values(by='seq_tempo', ascending=True)
seriedesemprego = seriedesemprego.dropna(subset=['seq_tempo'])

missing_seq_tempo = set(ANS_anomes['seq_tempo']) - set(seriedesemprego['seq_tempo'])

for seq in missing_seq_tempo:
    # Encontre o último valor de nascimento antes do seq_tempo ausente
    last_value = seriedesemprego[seriedesemprego['seq_tempo'] < seq]['txdesemprego'].iloc[-1]
    
    # Adicione a nova linha ao dataframe serienascimentos
    new_row = pd.DataFrame({'seq_tempo': [seq], 'txdesemprego': [last_value]})
    seriedesemprego = pd.concat([seriedesemprego, new_row], ignore_index=True)

# Ordene o dataframe serienascimentos por seq_tempo
seriedesemprego = seriedesemprego.sort_values(by='seq_tempo').reset_index(drop=True)

#apmazenando arquivo para uso posterior
seriedesempregotrimestre.to_csv("dados brutos/seriedesempregotrimestre.csv")
seriedesemprego.to_csv("dados brutos/seriedesemprego.csv")    


 

# %%Criar tabela de dados socioeconomicos

dados_socioeconomicos = ANS_anomes[['seq_tempo']]

dados_socioeconomicos = dados_socioeconomicos.merge(serienascimentos[['nascimentos', 'seq_tempo']], on='seq_tempo', how='left')
dados_socioeconomicos = dados_socioeconomicos.merge(serieobitos[['obitos', 'seq_tempo']], on='seq_tempo', how='left')
dados_socioeconomicos = dados_socioeconomicos.merge(seriedesemprego[['txdesemprego', 'seq_tempo']], on='seq_tempo', how='left')
dados_socioeconomicos = dados_socioeconomicos.merge(serieIPCA[['ipca', 'seq_tempo']], on='seq_tempo', how='left')
dados_socioeconomicos.to_csv("dados tratados/tabela_socioeconomicos.csv")

dados_planos_socioeconomicos = ANS_planos[['Benef_ativos', 'idplano','seq_tempo']]
dados_planos_socioeconomicos = dados_planos_socioeconomicos.merge(dados_socioeconomicos[['nascimentos','obitos', 'txdesemprego', 'ipca', 'seq_tempo']], on='seq_tempo', how='left')

dados_planos_socioeconomicos.to_csv("dados tratados/tabela_planos_socioeconomicos.csv")

dados_beneficiarios_socioeconomicos = ANS_anomes.merge(dados_socioeconomicos[['seq_tempo','nascimentos','obitos', 'txdesemprego', 'ipca']], on='seq_tempo', how='left') 
dados_beneficiarios_socioeconomicos = dados_beneficiarios_socioeconomicos[['seq_tempo','Benef_ativos','nascimentos','obitos', 'txdesemprego', 'ipca']]

dados_beneficiarios_socioeconomicos.to_csv("dados tratados/tabela_beneficiarios_socioeconomicos.csv")



